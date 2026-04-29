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
            var output = root.TryGetProperty("output", out var outputNode) && outputNode.ValueKind == JsonValueKind.String
                ? outputNode.GetString() ?? string.Empty
                : string.Empty;
            var errors = root.TryGetProperty("errors", out var errorNode) && errorNode.ValueKind == JsonValueKind.Array
                ? string.Join("; ", errorNode.EnumerateArray()
                    .Take(3)
                    .Select(e => e.GetString())
                    .Where(e => !string.IsNullOrWhiteSpace(e)))
                : "No diagnostics captured.";

            selectedSamples.Add(BuildSample(task, output, errors));
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

                var target = $$"""
                    Score={{score}}
                    VerificationPassed={{verificationPassed}}
                    Notes={{notes}}
                    """;
                selectedSamples.Add(BuildSample(prompt, target, notes));
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

    private static string BuildSample(string prompt, string badOutput, string diagnostics)
    {
        return $$"""
            [PROMPT]
            {{prompt}}

            [BAD_OUTPUT_OR_NOTES]
            {{badOutput}}

            [DIAGNOSTICS]
            {{diagnostics}}

            [EXPECTED_BEHAVIOR]
            Return compile-ready, requirement-matching code with no hallucinated APIs.
            """;
    }
}
