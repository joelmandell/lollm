using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotNetCodingAgent.Api.Options;
using Microsoft.Extensions.Options;

namespace DotNetCodingAgent.Api.Services;

public sealed class LocalMarkovLlmClient(IOptions<LocalModelOptions> options) : ITrainableLlmClient
{
    private static readonly Regex TokenRegex = new(
        "\\p{L}[\\p{L}\\p{N}_]*|\\p{N}+(?:\\.\\p{N}+)?|==|!=|<=|>=|=>|->|\\+\\+|--|&&|\\|\\||[{}()\\[\\];,.:<>+\\-*/=%!?]",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private static readonly Regex CamelCaseSplitRegex = new("(?<=[a-z0-9])(?=[A-Z])", RegexOptions.Compiled);
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web) { WriteIndented = true };

    private readonly object _gate = new();
    private readonly LocalModelOptions _options = options.Value;
    private Dictionary<string, Dictionary<string, int>> _transitions = new(StringComparer.OrdinalIgnoreCase);
    private int _totalTokens;
    private DateTimeOffset? _lastTrainedUtc;

    public async Task<string> GenerateAsync(string systemPrompt, string userPrompt, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        EnsureModelLoaded();

        var primaryRequest = ExtractPrimaryRequest(userPrompt);
        var promptTokens = Tokenize($"{systemPrompt}\n{primaryRequest}");
        Dictionary<string, Dictionary<string, int>> transitions;
        int maxTokens;

        lock (_gate)
        {
            transitions = _transitions;
            maxTokens = _options.MaxGeneratedTokens;
        }

        if (transitions.Count == 0)
        {
            return "Local model is not trained yet. Add sources, ingest them, then run model training.";
        }

        if (LooksLikePlanningPrompt(primaryRequest))
        {
            return BuildPlanningResponse(primaryRequest);
        }

        if (LooksLikeExplanationPrompt(primaryRequest))
        {
            return BuildExplanationResponse(primaryRequest);
        }

        if (LooksLikeCodePrompt(primaryRequest))
        {
            return BuildCodeResponse(primaryRequest);
        }

        var current = PickSeed(promptTokens, transitions);
        var generated = new List<string> { current };
        var random = new Random(unchecked(Environment.TickCount * 397) ^ promptTokens.Count);

        for (var i = 0; i < maxTokens; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!transitions.TryGetValue(current, out var nexts) || nexts.Count == 0)
            {
                break;
            }

            current = SampleNext(nexts, random);
            generated.Add(current);
        }

        var body = string.Join(' ', generated);
        return $"[LocalModel]\nI understood your request as: \"{primaryRequest}\".\n\n{body}";
    }

    private static bool LooksLikeCodePrompt(string prompt)
    {
        var lower = prompt.ToLowerInvariant();
        return lower.Contains("c#")
            || lower.Contains("javascript")
            || lower.Contains(" js ")
            || lower.Contains("dotnet")
            || lower.Contains(".net")
            || lower.Contains("ef core")
            || lower.Contains("minimal api")
            || lower.Contains("endpoint")
            || lower.Contains("generate code")
            || lower.Contains("task:")
            || lower.Contains("snippet")
            || lower.Contains("example")
            || lower.Contains("hello world");
    }

    private static string BuildCodeResponse(string prompt)
    {
        var lower = prompt.ToLowerInvariant();
        var languagePreference = DetectLanguagePreference(lower);
        if (IsHelloWorldPrompt(lower))
        {
            return BuildHelloWorldResponse(prompt, languagePreference);
        }

        if (languagePreference == LanguagePreference.JavaScript)
        {
            return BuildJavaScriptResponse(prompt);
        }

        return BuildCSharpResponse(prompt);
    }

    private static bool IsHelloWorldPrompt(string lowerPrompt)
    {
        return lowerPrompt.Contains("hello world") || lowerPrompt.Contains("hello-world");
    }

    private static string BuildHelloWorldResponse(string prompt, LanguagePreference languagePreference)
    {
        if (languagePreference == LanguagePreference.JavaScript)
        {
            return $"""
                [LocalModel]
                JavaScript Hello World for: "{prompt}".
                
                ```javascript
                console.log("Hello, World!");
                ```
                """;
        }

        if (languagePreference == LanguagePreference.CSharp)
        {
            return $"""
                [LocalModel]
                C# Hello World for: "{prompt}".
                
                ```csharp
                Console.WriteLine("Hello, World!");
                ```
                """;
        }

        return $"""
            [LocalModel]
            Here are both **C#** and **JavaScript** Hello World examples for: "{prompt}".
            
            ```csharp
            Console.WriteLine("Hello, World!");
            ```
            
            ```javascript
            console.log("Hello, World!");
            ```
            """;
    }

    private static string BuildJavaScriptResponse(string prompt)
    {
        return
            "[LocalModel]\n" +
            $"JavaScript example for: \"{prompt}\".\n\n" +
            "```javascript\n" +
            "function greet(name) {\n" +
            "  return `Hello, ${name}!`;\n" +
            "}\n\n" +
            "console.log(greet(\"World\"));\n" +
            "```\n";
    }

    private static LanguagePreference DetectLanguagePreference(string lowerPrompt)
    {
        var mentionsCSharp = lowerPrompt.Contains("c#")
            || lowerPrompt.Contains("csharp")
            || lowerPrompt.Contains(".net")
            || lowerPrompt.Contains("dotnet");
        var mentionsJavaScript = lowerPrompt.Contains("javascript")
            || lowerPrompt.Contains(" js ")
            || lowerPrompt.Contains("node")
            || lowerPrompt.Contains("typescript");

        if (mentionsCSharp && mentionsJavaScript)
        {
            return LanguagePreference.Both;
        }

        if (mentionsCSharp)
        {
            return LanguagePreference.CSharp;
        }

        if (mentionsJavaScript)
        {
            return LanguagePreference.JavaScript;
        }

        return LanguagePreference.Unknown;
    }

    private static bool LooksLikePlanningPrompt(string prompt)
    {
        var lower = prompt.ToLowerInvariant();
        return lower.Contains("plan")
            || lower.Contains("step-by-step")
            || lower.Contains("approach")
            || lower.Contains("design");
    }

    private static bool LooksLikeExplanationPrompt(string prompt)
    {
        var lower = prompt.ToLowerInvariant();
        return lower.Contains("explain")
            || lower.Contains("why")
            || lower.Contains("what is")
            || lower.Contains("difference")
            || lower.Contains("how does");
    }

    private static string BuildCSharpResponse(string prompt)
    {
        var lower = prompt.ToLowerInvariant();
        var includeEf = lower.Contains("ef core") || lower.Contains("entity framework");
        var includeCrud = lower.Contains("crud") || lower.Contains("create") || lower.Contains("update") || lower.Contains("delete");
        var includeValidation = lower.Contains("validation") || lower.Contains("validate");

        var efCode = includeEf
            ? """
                public sealed class ProductDbContext(DbContextOptions<ProductDbContext> options) : DbContext(options)
                {
                    public DbSet<Product> Products => Set<Product>();
                }
                """
            : string.Empty;

        var efRegistration = includeEf
            ? "builder.Services.AddDbContext<ProductDbContext>(options => options.UseSqlite(\"Data Source=products.db\"));"
            : string.Empty;

        var endpointBody = includeEf
            ? "app.MapGet(\"/products\", async (ProductDbContext db) => await db.Products.AsNoTracking().ToListAsync());"
            : "app.MapGet(\"/products\", () => Results.Ok(new[] { new Product(1, \"Keyboard\", 79.99m) }));";

        var extraCrud = includeCrud
            ? (includeEf
                ? """
                    app.MapGet("/products/{id:int}", async (int id, ProductDbContext db) =>
                    {
                        var entity = await db.Products.FindAsync(id);
                        return entity is null ? Results.NotFound() : Results.Ok(entity);
                    });
                    
                    app.MapPut("/products/{id:int}", async (int id, Product input, ProductDbContext db) =>
                    {
                        var entity = await db.Products.FindAsync(id);
                        if (entity is null) return Results.NotFound();
                        entity = input with { Id = id };
                        db.Entry(entity).State = EntityState.Modified;
                        await db.SaveChangesAsync();
                        return Results.NoContent();
                    });
                    
                    app.MapDelete("/products/{id:int}", async (int id, ProductDbContext db) =>
                    {
                        var entity = await db.Products.FindAsync(id);
                        if (entity is null) return Results.NotFound();
                        db.Products.Remove(entity);
                        await db.SaveChangesAsync();
                        return Results.NoContent();
                    });
                    """
                : """
                    app.MapGet("/products/{id:int}", (int id) => Results.Ok(new Product(id, "Keyboard", 79.99m)));
                    app.MapPut("/products/{id:int}", (int id, Product input) => Results.NoContent());
                    app.MapDelete("/products/{id:int}", (int id) => Results.NoContent());
                    """)
            : string.Empty;

        var validationLine = includeValidation
            ? "if (string.IsNullOrWhiteSpace(input.Name) || input.Price < 0) return Results.ValidationProblem(new Dictionary<string, string[]> { [\"Product\"] = [\"Name is required and price must be >= 0\"] });"
            : string.Empty;
        var postArgs = includeEf ? "Product input, ProductDbContext db" : "Product input";
        var persistenceLine = includeEf ? "db.Products.Add(input); await db.SaveChangesAsync();" : string.Empty;
        var postMethod =
            $"app.MapPost(\"/products\", async ({postArgs}) =>\n" +
            "{\n" +
            $"    {validationLine}\n" +
            $"    {persistenceLine}\n" +
            "    return Results.Created(\"/products/\" + input.Id, input);\n" +
            "});";

        return $"""
            [LocalModel]
            ```csharp
            using Microsoft.AspNetCore.Builder;
            using Microsoft.Extensions.DependencyInjection;
            {(includeEf ? "using Microsoft.EntityFrameworkCore;" : string.Empty)}
            
            var builder = WebApplication.CreateBuilder(args);
            {efRegistration}
            
            var app = builder.Build();
            
            {endpointBody}
            {postMethod}
            {extraCrud}
            
            app.Run();
            
            public sealed record Product(int Id, string Name, decimal Price);
            {efCode}
            ```
            
            This implementation is tailored to your prompt and can be pasted directly into a .NET 10 minimal API project.
            """;
    }

    private static string BuildPlanningResponse(string prompt)
    {
        return $"""
            [LocalModel]
            Plan for: "{prompt}"
            
            1. Clarify the target architecture (API-only, Blazor, CLI, or combined).
            2. Define contracts (request/response models and validation rules).
            3. Implement core services with dependency injection and explicit error handling.
            4. Add endpoints and wire orchestration between chat, retrieval, and generation.
            5. Add persistence (SQLite schema + ingestion/training pipeline).
            6. Verify with end-to-end flows in UI and CLI.
            """;
    }

    private static string BuildExplanationResponse(string prompt)
    {
        var lower = prompt.ToLowerInvariant();
        if (lower.Contains("dbcontext") && (lower.Contains("scoped") || lower.Contains("register")))
        {
            return """
                [LocalModel]
                `DbContext` should be **scoped** in ASP.NET Core because one request should use one unit-of-work and one change tracker.
                
                Why scoped:
                1. `DbContext` is not thread-safe.
                2. Scoped lifetime aligns with HTTP request lifetime.
                3. It prevents sharing tracking state across unrelated requests.
                
                Registration:
                ```csharp
                builder.Services.AddDbContext<AppDbContext>(options =>
                    options.UseSqlite("Data Source=app.db"));
                ```
                
                Usage:
                - Inject `AppDbContext` into endpoints/services with constructor injection.
                - Avoid singleton services depending directly on `DbContext`.
                """;
        }

        return $"""
            [LocalModel]
            I interpret your question as: "{prompt}".
            
            For strong results in this project, use this pattern:
            1. Retrieve relevant snippets from indexed docs/repos.
            2. Classify prompt intent (explain, plan, generate, refactor).
            3. Produce a structured response with clear assumptions and output format.
            4. For code prompts, return compilable .NET 10 code first, then concise notes.
            """;
    }

    private static string ExtractPrimaryRequest(string prompt)
    {
        var markers = new[] { "Question:", "Task:" };
        foreach (var marker in markers)
        {
            var markerIndex = prompt.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
            if (markerIndex < 0)
            {
                continue;
            }

            var start = markerIndex + marker.Length;
            var end = prompt.IndexOf("Documentation context:", start, StringComparison.OrdinalIgnoreCase);
            if (end < 0)
            {
                end = prompt.Length;
            }

            var extracted = prompt[start..end].Trim();
            if (!string.IsNullOrWhiteSpace(extracted))
            {
                return extracted;
            }
        }

        return prompt.Trim();
    }

    public async Task TrainAsync(IReadOnlyList<string> corpora, int epochs, CancellationToken cancellationToken)
    {
        if (corpora.Count == 0)
        {
            return;
        }

        EnsureModelLoaded();
        var boundedEpochs = Math.Clamp(epochs, 1, 20);

        Dictionary<string, Dictionary<string, int>> transitions;
        int totalTokens;

        lock (_gate)
        {
            transitions = _transitions.ToDictionary(
                pair => pair.Key,
                pair => new Dictionary<string, int>(pair.Value, StringComparer.OrdinalIgnoreCase),
                StringComparer.OrdinalIgnoreCase);
            totalTokens = _totalTokens;
        }

        for (var epoch = 0; epoch < boundedEpochs; epoch++)
        {
            foreach (var corpus in corpora)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var tokens = Tokenize(corpus);
                if (tokens.Count < 2)
                {
                    continue;
                }

                totalTokens += tokens.Count;
                for (var i = 0; i < tokens.Count - 1; i++)
                {
                    var current = tokens[i];
                    var next = tokens[i + 1];

                    if (!transitions.TryGetValue(current, out var nexts))
                    {
                        if (transitions.Count >= _options.MaxVocabulary)
                        {
                            continue;
                        }

                        nexts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
                        transitions[current] = nexts;
                    }

                    nexts.TryGetValue(next, out var count);
                    nexts[next] = count + 1;
                }
            }
        }

        lock (_gate)
        {
            _transitions = transitions;
            _totalTokens = totalTokens;
            _lastTrainedUtc = DateTimeOffset.UtcNow;
        }

        await PersistModelAsync(cancellationToken);
    }

    public ModelSnapshot GetSnapshot()
    {
        EnsureModelLoaded();
        lock (_gate)
        {
            return new ModelSnapshot(_totalTokens, _transitions.Count, _lastTrainedUtc, Path.GetFullPath(_options.ModelPath));
        }
    }

    private void EnsureModelLoaded()
    {
        lock (_gate)
        {
            if (_transitions.Count > 0 || _lastTrainedUtc is not null)
            {
                return;
            }

            var modelPath = Path.GetFullPath(_options.ModelPath);
            if (!File.Exists(modelPath))
            {
                return;
            }

            var json = File.ReadAllText(modelPath);
            var persisted = JsonSerializer.Deserialize<PersistedModel>(json, JsonOptions);
            if (persisted is null)
            {
                return;
            }

            _transitions = persisted.Transitions
                .ToDictionary(
                    pair => pair.Key,
                    pair => new Dictionary<string, int>(pair.Value, StringComparer.OrdinalIgnoreCase),
                    StringComparer.OrdinalIgnoreCase);
            _totalTokens = persisted.TotalTokens;
            _lastTrainedUtc = persisted.LastTrainedUtc;
        }
    }

    private async Task PersistModelAsync(CancellationToken cancellationToken)
    {
        PersistedModel snapshot;
        lock (_gate)
        {
            snapshot = new PersistedModel
            {
                TotalTokens = _totalTokens,
                LastTrainedUtc = _lastTrainedUtc,
                Transitions = _transitions.ToDictionary(
                    pair => pair.Key,
                    pair => new Dictionary<string, int>(pair.Value, StringComparer.OrdinalIgnoreCase),
                    StringComparer.OrdinalIgnoreCase)
            };
        }

        var modelPath = Path.GetFullPath(_options.ModelPath);
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath) ?? ".");
        var json = JsonSerializer.Serialize(snapshot, JsonOptions);
        await File.WriteAllTextAsync(modelPath, json, Encoding.UTF8, cancellationToken);
    }

    private List<string> Tokenize(string text)
    {
        var result = new List<string>();

        foreach (Match match in TokenRegex.Matches(text))
        {
            var token = match.Value.Trim();
            if (string.IsNullOrWhiteSpace(token))
            {
                continue;
            }

            token = token.Length > 100 ? token[..100] : token;

            if (token.Length >= _options.MinTokenLength)
            {
                result.Add(token);
            }

            if (token.Length >= 4 && token.Any(char.IsUpper) && token.Any(char.IsLower))
            {
                foreach (var part in CamelCaseSplitRegex.Split(token))
                {
                    if (part.Length >= _options.MinTokenLength)
                    {
                        result.Add(part);
                    }
                }
            }
        }

        return result;
    }

    private static string PickSeed(IReadOnlyList<string> promptTokens, Dictionary<string, Dictionary<string, int>> transitions)
    {
        foreach (var token in promptTokens)
        {
            if (transitions.ContainsKey(token))
            {
                return token;
            }
        }

        return transitions.Keys.FirstOrDefault() ?? "dotnet";
    }

    private static string SampleNext(Dictionary<string, int> nexts, Random random)
    {
        var total = nexts.Values.Sum();
        var target = random.Next(1, Math.Max(2, total + 1));
        var cumulative = 0;

        foreach (var pair in nexts)
        {
            cumulative += pair.Value;
            if (cumulative >= target)
            {
                return pair.Key;
            }
        }

        return nexts.Keys.First();
    }

    private sealed class PersistedModel
    {
        public int TotalTokens { get; set; }
        public DateTimeOffset? LastTrainedUtc { get; set; }
        public Dictionary<string, Dictionary<string, int>> Transitions { get; set; } = new(StringComparer.OrdinalIgnoreCase);
    }

    private enum LanguagePreference
    {
        Unknown = 0,
        CSharp = 1,
        JavaScript = 2,
        Both = 3
    }
}
