using System.Text;
using System.Text.Json;
using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Api.Services;

public sealed class FeedbackCorpusService(EvalFeedbackService evalFeedbackService)
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    public async Task<BuildFeedbackCorpusResponse> BuildAsync(
        BuildFeedbackCorpusRequest request,
        CancellationToken cancellationToken)
    {
        var maxItems = Math.Clamp(request.MaxItems, 1, 5_000);
        var threshold = Math.Clamp(request.LowScoreThreshold, 0, 100);
        var paths = evalFeedbackService.GetFeedbackPaths();
        Directory.CreateDirectory(paths.Directory);

        var generationLines = await evalFeedbackService.ReadJsonLinesAsync(paths.GenerationPath, cancellationToken);
        var evalLines = await evalFeedbackService.ReadJsonLinesAsync(paths.EvalRunsPath, cancellationToken);

        var selectedSamples = new List<string>();
        var sourceGenerationEntries = 0;
        var sourceEvalEntries = 0;

        foreach (var line in generationLines)
        {
            if (selectedSamples.Count >= maxItems)
            {
                break;
            }

            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            sourceGenerationEntries++;
            var verificationPassed = root.TryGetProperty("verificationPassed", out var passNode) &&
                                     passNode.ValueKind == JsonValueKind.True;
            if (!request.IncludePassingSamples && verificationPassed)
            {
                continue;
            }

            if (!root.TryGetProperty("task", out var taskNode) || taskNode.ValueKind != JsonValueKind.String)
            {
                continue;
            }

            var task = taskNode.GetString() ?? string.Empty;
            var outputLength = root.TryGetProperty("output", out var outputNode) && outputNode.ValueKind == JsonValueKind.String
                ? (outputNode.GetString() ?? string.Empty).Length
                : 0;
            var errors = root.TryGetProperty("errors", out var errorNode) && errorNode.ValueKind == JsonValueKind.Array
                ? string.Join("; ", errorNode.EnumerateArray()
                    .Take(3)
                    .Select(e => e.GetString())
                    .Where(e => !string.IsNullOrWhiteSpace(e)))
                : "No diagnostics captured.";
            var notes = $"VerificationPassed={verificationPassed}; OutputLength={outputLength}.";
            selectedSamples.Add(BuildSample(task, errors, notes));
        }

        foreach (var line in evalLines)
        {
            if (selectedSamples.Count >= maxItems)
            {
                break;
            }

            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            if (!root.TryGetProperty("results", out var resultsNode) || resultsNode.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            foreach (var caseNode in resultsNode.EnumerateArray())
            {
                if (selectedSamples.Count >= maxItems)
                {
                    break;
                }

                sourceEvalEntries++;
                var score = caseNode.TryGetProperty("score", out var scoreNode) && scoreNode.ValueKind == JsonValueKind.Number
                    ? scoreNode.GetInt32()
                    : 0;
                var verificationPassed = caseNode.TryGetProperty("verificationPassed", out var passNode) &&
                                         passNode.ValueKind == JsonValueKind.True;
                if (!request.IncludePassingSamples && verificationPassed && score >= threshold)
                {
                    continue;
                }

                var prompt = caseNode.TryGetProperty("prompt", out var promptNode) && promptNode.ValueKind == JsonValueKind.String
                    ? promptNode.GetString() ?? string.Empty
                    : string.Empty;
                if (string.IsNullOrWhiteSpace(prompt))
                {
                    continue;
                }

                var notes = caseNode.TryGetProperty("notes", out var notesNode) && notesNode.ValueKind == JsonValueKind.String
                    ? notesNode.GetString() ?? "No notes."
                    : "No notes.";

                var diagnostics = $$"""
                    Score={{score}}
                    VerificationPassed={{verificationPassed}}
                    Notes={{notes}}
                    """;
                selectedSamples.Add(BuildSample(prompt, diagnostics, "Evaluation case feedback."));
            }
        }

        string? jsonlPath = null;
        string? textPath = null;
        if (request.IncludeJsonl)
        {
            jsonlPath = Path.Combine(paths.Directory, "feedback_corpus.jsonl");
            await WriteJsonlAsync(jsonlPath, selectedSamples, cancellationToken);
        }

        if (request.IncludeText)
        {
            textPath = Path.Combine(paths.Directory, "feedback_corpus.txt");
            await File.WriteAllTextAsync(textPath, string.Join(Environment.NewLine + Environment.NewLine, selectedSamples), cancellationToken);
        }

        return new BuildFeedbackCorpusResponse(
            Success: true,
            SourceGenerationEntries: sourceGenerationEntries,
            SourceEvalEntries: sourceEvalEntries,
            SelectedItems: selectedSamples.Count,
            JsonlPath: jsonlPath,
            TextPath: textPath,
            Message: $"Built feedback corpus with {selectedSamples.Count} samples.");
    }

    private static async Task WriteJsonlAsync(string path, IReadOnlyList<string> samples, CancellationToken cancellationToken)
    {
        await using var stream = File.Create(path);
        await using var writer = new StreamWriter(stream, Encoding.UTF8);
        foreach (var sample in samples)
        {
            var line = JsonSerializer.Serialize(new { text = sample }, JsonOptions);
            await writer.WriteLineAsync(line.AsMemory(), cancellationToken);
        }
    }

    private static string BuildSample(string prompt, string diagnostics, string guidance)
    {
        var expectedOutput = BuildExpectedOutput(prompt);
        return $$"""
            [PROMPT]
            {{prompt}}

            [DIAGNOSTICS]
            {{diagnostics}}

            [EXPECTED_OUTPUT]
            {{expectedOutput}}

            [EXPECTED_BEHAVIOR]
            Return compile-ready, requirement-matching code with no hallucinated APIs.
            Do not output placeholders, disclaimers, or low-confidence fallback text.
            Keep the implementation concrete and technology-aligned with the prompt.

            [GUIDANCE]
            {{guidance}}
            """;
    }

    private static string BuildExpectedOutput(string prompt)
    {
        var lower = prompt.ToLowerInvariant();
        if (lower.Contains("minimal api", StringComparison.Ordinal) &&
            lower.Contains("sqlite", StringComparison.Ordinal))
        {
            var dataSource = lower.Contains("hello.db", StringComparison.Ordinal)
                ? "Data Source=hello.db"
                : "Data Source=app.db";
            return $$"""
                ```csharp
                using Microsoft.EntityFrameworkCore;

                var builder = WebApplication.CreateBuilder(args);
                builder.Services.AddDbContext<TodoDbContext>(options =>
                    options.UseSqlite("{{dataSource}}"));

                var app = builder.Build();
                app.MapGet("/todos", async (TodoDbContext db) =>
                    await db.Todos.AsNoTracking().ToListAsync());
                app.MapPost("/todos", async (TodoItem todo, TodoDbContext db) =>
                {
                    db.Todos.Add(todo);
                    await db.SaveChangesAsync();
                    return Results.Created($"/todos/{todo.Id}", todo);
                });
                app.Run();

                public sealed class TodoDbContext(DbContextOptions<TodoDbContext> options) : DbContext(options)
                {
                    public DbSet<TodoItem> Todos => Set<TodoItem>();
                }

                public sealed class TodoItem
                {
                    public int Id { get; set; }
                    public string Title { get; set; } = string.Empty;
                    public bool IsCompleted { get; set; }
                }
                ```
                """;
        }

        if (lower.Contains("minimal api", StringComparison.Ordinal) &&
            (lower.Contains("postgres", StringComparison.Ordinal) || lower.Contains("npgsql", StringComparison.Ordinal)))
        {
            return """
                ```csharp
                using Microsoft.EntityFrameworkCore;

                var builder = WebApplication.CreateBuilder(args);
                builder.Services.AddDbContext<TodoDbContext>(options =>
                    options.UseNpgsql(builder.Configuration.GetConnectionString("DefaultConnection")));

                var app = builder.Build();
                app.MapGet("/todos", async (TodoDbContext db) => await db.Todos.AsNoTracking().ToListAsync());
                app.Run();

                public sealed class TodoDbContext(DbContextOptions<TodoDbContext> options) : DbContext(options)
                {
                    public DbSet<TodoItem> Todos => Set<TodoItem>();
                }

                public sealed class TodoItem
                {
                    public int Id { get; set; }
                    public string Title { get; set; } = string.Empty;
                    public bool IsCompleted { get; set; }
                }
                ```
                """;
        }

        if (lower.Contains("xunit", StringComparison.Ordinal) || lower.Contains("[fact]", StringComparison.Ordinal))
        {
            return """
                ```csharp
                using Xunit;

                public sealed class SampleTests
                {
                    [Fact]
                    public void Works()
                    {
                        var result = 2 + 2;
                        Assert.Equal(4, result);
                    }
                }
                ```
                """;
        }

        if (lower.Contains("httpclient", StringComparison.Ordinal) ||
            (lower.Contains("http", StringComparison.Ordinal) &&
             (lower.Contains("retry", StringComparison.Ordinal) ||
              lower.Contains("retri", StringComparison.Ordinal) ||
              lower.Contains("backoff", StringComparison.Ordinal))))
        {
            return """
                ```csharp
                using System.Net.Http;
                using Microsoft.Extensions.Logging;

                public sealed class ResilientHttpClient(HttpClient httpClient, ILogger<ResilientHttpClient> logger)
                {
                    public async Task<string> GetWithRetryAsync(
                        string url,
                        int maxRetries = 3,
                        CancellationToken cancellationToken = default)
                    {
                        var delay = TimeSpan.FromMilliseconds(200);
                        for (var attempt = 1; ; attempt++)
                        {
                            try
                            {
                                using var response = await httpClient.GetAsync(url, cancellationToken);
                                response.EnsureSuccessStatusCode();
                                return await response.Content.ReadAsStringAsync(cancellationToken);
                            }
                            catch (Exception ex) when (attempt <= maxRetries && ex is HttpRequestException or TaskCanceledException)
                            {
                                logger.LogWarning(ex, "Transient HTTP failure on attempt {Attempt}", attempt);
                                await Task.Delay(delay, cancellationToken);
                                delay = TimeSpan.FromMilliseconds(delay.TotalMilliseconds * 2);
                            }
                        }
                    }
                }
                ```
                """;
        }

        if (lower.Contains("todo", StringComparison.Ordinal) &&
            lower.Contains("console", StringComparison.Ordinal))
        {
            return """
                ```csharp
                using System;
                using System.Collections.Generic;
                using System.Linq;

                var todos = new List<TodoItem>();
                var nextId = 1;

                while (true)
                {
                    Console.WriteLine("1) Add  2) List  3) Complete  4) Delete  0) Exit");
                    var choice = Console.ReadLine();
                    if (choice == "0") break;

                    if (choice == "1")
                    {
                        Console.Write("Title: ");
                        var title = Console.ReadLine()?.Trim();
                        if (string.IsNullOrWhiteSpace(title))
                        {
                            Console.WriteLine("Title is required.");
                            continue;
                        }

                        todos.Add(new TodoItem { Id = nextId++, Title = title, IsCompleted = false, CreatedAt = DateTime.UtcNow });
                    }
                    else if (choice == "2")
                    {
                        foreach (var todo in todos.OrderBy(t => t.Id))
                        {
                            Console.WriteLine($"{todo.Id}: {todo.Title} [{(todo.IsCompleted ? "x" : " ")}]");
                        }
                    }
                    else if (choice == "3")
                    {
                        Console.Write("Id: ");
                        if (int.TryParse(Console.ReadLine(), out var id))
                        {
                            var todo = todos.FirstOrDefault(t => t.Id == id);
                            if (todo is not null) todo.IsCompleted = true;
                        }
                    }
                    else if (choice == "4")
                    {
                        Console.Write("Id: ");
                        if (int.TryParse(Console.ReadLine(), out var id))
                        {
                            todos.RemoveAll(t => t.Id == id);
                        }
                    }
                }

                public sealed class TodoItem
                {
                    public int Id { get; set; }
                    public string Title { get; set; } = string.Empty;
                    public bool IsCompleted { get; set; }
                    public DateTime CreatedAt { get; set; }
                }
                ```
                """;
        }

        if (lower.Contains("blazor", StringComparison.Ordinal) &&
            (lower.Contains("javascript", StringComparison.Ordinal) || lower.Contains("js interop", StringComparison.Ordinal)))
        {
            return """
                ```csharp
                @inject IJSRuntime JS

                <button @onclick="SaveAsync">Save</button>

                @code {
                    private async Task SaveAsync()
                    {
                        await JS.InvokeVoidAsync("todoStorage.save", "draft", "hello");
                    }
                }
                ```

                ```javascript
                window.todoStorage = {
                  save: (key, value) => localStorage.setItem(key, value),
                  load: (key) => localStorage.getItem(key)
                };
                ```
                """;
        }

        return """
            ```csharp
            using System;

            public static class Solution
            {
                public static void Run()
                {
                    Console.WriteLine("Implement according to prompt requirements.");
                }
            }
            ```
            """;
    }
}
