using System.Text;
using System.Text.Json;
using DotNetCodingAgent.Api.Options;
using DotNetCodingAgent.Contracts;
using Microsoft.Extensions.Options;

namespace DotNetCodingAgent.Api.Services;

public sealed class EvalFeedbackService(
    IOptions<ModelBackendOptions> options,
    IHostEnvironment hostEnvironment)
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly ModelBackendOptions _options = options.Value;

    private string FeedbackDirectory => Path.GetFullPath(
        Path.IsPathRooted(_options.ExportDataDirectory)
            ? _options.ExportDataDirectory
            : Path.Combine(hostEnvironment.ContentRootPath, _options.ExportDataDirectory));

    private string GenerationFeedbackPath => Path.Combine(FeedbackDirectory, "generation_feedback.jsonl");
    private string EvalRunsPath => Path.Combine(FeedbackDirectory, "eval_runs.jsonl");

    public async Task RecordGenerationAsync(
        GenerateCodeRequest request,
        CodeGenerationMetrics metrics,
        string output,
        CancellationToken cancellationToken)
    {
        var entry = new
        {
            timestampUtc = DateTimeOffset.UtcNow,
            type = "generation",
            task = request.Task,
            request.Language,
            request.UseKnowledge,
            request.MaxKnowledgeSnippets,
            metrics.VerificationPassed,
            metrics.VerificationAttempts,
            metrics.RepairIterationsUsed,
            errors = metrics.LastVerificationErrors,
            output = Truncate(output, 2400)
        };

        await AppendJsonLineAsync(GenerationFeedbackPath, entry, cancellationToken);
    }

    public async Task RecordEvalRunAsync(
        CodingEvalRequest request,
        CodingEvalResponse response,
        CancellationToken cancellationToken)
    {
        var entry = new
        {
            timestampUtc = DateTimeOffset.UtcNow,
            type = "eval-run",
            request.UseKnowledge,
            request.MaxKnowledgeSnippets,
            promptCount = request.Prompts?.Count ?? 0,
            response.AverageScore,
            response.CaseCount,
            results = response.Results
        };

        await AppendJsonLineAsync(EvalRunsPath, entry, cancellationToken);
    }

    public async Task<FeedbackStatusResponse> GetStatusAsync(CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(FeedbackDirectory);
        var generationCount = await CountJsonLinesAsync(GenerationFeedbackPath, cancellationToken);
        var evalRunCount = await CountJsonLinesAsync(EvalRunsPath, cancellationToken);
        var lastWrite = GetLastWriteUtc(GenerationFeedbackPath, EvalRunsPath);

        return new FeedbackStatusResponse(
            FeedbackDirectory: FeedbackDirectory,
            GenerationFeedbackPath: GenerationFeedbackPath,
            EvalRunsPath: EvalRunsPath,
            GenerationFeedbackEntries: generationCount,
            EvalRunEntries: evalRunCount,
            LastUpdatedUtc: lastWrite);
    }

    private async Task AppendJsonLineAsync(string path, object payload, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(FeedbackDirectory);
        var line = JsonSerializer.Serialize(payload, JsonOptions);

        await _gate.WaitAsync(cancellationToken);
        try
        {
            await using var stream = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read);
            await using var writer = new StreamWriter(stream, Encoding.UTF8);
            await writer.WriteLineAsync(line);
        }
        finally
        {
            _gate.Release();
        }
    }

    private static async Task<int> CountJsonLinesAsync(string path, CancellationToken cancellationToken)
    {
        if (!File.Exists(path))
        {
            return 0;
        }

        var count = 0;
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        using var reader = new StreamReader(stream, Encoding.UTF8);
        while (await reader.ReadLineAsync(cancellationToken) is not null)
        {
            count++;
        }

        return count;
    }

    private static DateTimeOffset? GetLastWriteUtc(params string[] paths)
    {
        var writes = paths
            .Where(File.Exists)
            .Select(path => (DateTimeOffset?)File.GetLastWriteTimeUtc(path))
            .Where(date => date is not null)
            .ToList();
        return writes.Count == 0 ? null : writes.Max();
    }

    private static string Truncate(string input, int maxChars)
    {
        if (string.IsNullOrWhiteSpace(input) || input.Length <= maxChars)
        {
            return input;
        }

        return input[..maxChars];
    }
}
