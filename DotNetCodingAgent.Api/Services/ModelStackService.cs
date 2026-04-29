using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotNetCodingAgent.Api.Options;
using DotNetCodingAgent.Contracts;
using Microsoft.Extensions.Options;

namespace DotNetCodingAgent.Api.Services;

public sealed class ModelStackService(
    IKnowledgeRepository repository,
    IOptions<ModelBackendOptions> options,
    IHostEnvironment hostEnvironment,
    TransformerServiceLlmClient transformerClient)
{
    private static readonly Regex FenceRegex = new("```(?:csharp|cs)?\\s*(.*?)```", RegexOptions.Singleline | RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex MultiWhitespaceRegex = new("\\s+", RegexOptions.Compiled);
    private static readonly HashSet<string> NoiseLines =
    [
        "table of contents",
        "additional resources",
        "this browser is no longer supported.",
        "skip to main content",
        "feedback",
        "theme",
        "light",
        "dark",
        "high contrast"
    ];

    private readonly ModelBackendOptions _options = options.Value;

    public async Task<ExportCorpusResponse> ExportCorpusAsync(ExportCorpusRequest request, CancellationToken cancellationToken)
    {
        var rawChunks = await repository.GetAllChunkContentsAsync(cancellationToken);
        var chunks = PrepareTrainingCorpus(rawChunks);
        var directory = Path.GetFullPath(
            Path.IsPathRooted(_options.ExportDataDirectory)
                ? _options.ExportDataDirectory
                : Path.Combine(hostEnvironment.ContentRootPath, _options.ExportDataDirectory));

        Directory.CreateDirectory(directory);

        string? jsonlPath = null;
        string? textPath = null;

        if (request.IncludeJsonl)
        {
            jsonlPath = Path.Combine(directory, "corpus.jsonl");
            await using var stream = File.Create(jsonlPath);
            await using var writer = new StreamWriter(stream, Encoding.UTF8);
            foreach (var chunk in chunks)
            {
                var line = JsonSerializer.Serialize(new { text = chunk });
                await writer.WriteLineAsync(line);
            }
        }

        if (request.IncludeText)
        {
            textPath = Path.Combine(directory, "corpus.txt");
            await File.WriteAllTextAsync(textPath, string.Join(Environment.NewLine + Environment.NewLine, chunks), cancellationToken);
        }

        return new ExportCorpusResponse(
            Success: true,
            ChunkCount: chunks.Count,
            JsonlPath: jsonlPath,
            TextPath: textPath,
            Message: $"Exported {chunks.Count} knowledge chunks.");
    }

    public async Task<BackendStatusResponse> GetBackendStatusAsync(CancellationToken cancellationToken)
    {
        var healthy = await transformerClient.IsHealthyAsync(cancellationToken);
        return new BackendStatusResponse(
            Provider: _options.Provider,
            TransformerHealthy: healthy,
            TransformerBaseUrl: transformerClient.BaseUrl);
    }

    private static List<string> PrepareTrainingCorpus(IReadOnlyList<string> rawChunks)
    {
        var results = new List<string>(rawChunks.Count);
        var dedupe = new HashSet<string>(StringComparer.Ordinal);
        foreach (var chunk in rawChunks)
        {
            var cleaned = CleanChunk(chunk);
            if (!string.IsNullOrWhiteSpace(cleaned) && dedupe.Add(cleaned))
            {
                results.Add(cleaned);
            }

            foreach (Match match in FenceRegex.Matches(chunk))
            {
                var code = CleanChunk(match.Groups[1].Value);
                if (!string.IsNullOrWhiteSpace(code) && dedupe.Add(code))
                {
                    // Upweight concrete code signal during training.
                    results.Add(code);
                    results.Add(code);
                }
            }
        }

        return results;
    }

    private static string CleanChunk(string text)
    {
        var lines = text
            .Replace("\r\n", "\n", StringComparison.Ordinal)
            .Replace('\r', '\n')
            .Split('\n');

        var builder = new StringBuilder();
        var blankCount = 0;
        foreach (var rawLine in lines)
        {
            var line = rawLine.Trim();
            if (line.Length == 0)
            {
                blankCount++;
                if (blankCount <= 1)
                {
                    builder.AppendLine();
                }

                continue;
            }

            blankCount = 0;
            var lower = line.ToLowerInvariant();
            if (NoiseLines.Contains(lower))
            {
                continue;
            }

            if (line.StartsWith('|') && line.Contains("---", StringComparison.Ordinal))
            {
                continue;
            }

            var alphaCount = line.Count(char.IsLetter);
            var looksCode = line.IndexOfAny(['{', '}', '(', ')', ';', '<', '>']) >= 0;
            if (alphaCount < 2 && !looksCode)
            {
                continue;
            }

            var compact = MultiWhitespaceRegex.Replace(line, " ").Trim();
            if (compact.Length > 1_200)
            {
                compact = compact[..1_200];
            }

            builder.AppendLine(compact);
        }

        return builder.ToString().Trim();
    }
}
