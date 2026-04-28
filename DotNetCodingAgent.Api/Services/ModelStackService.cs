using System.Text;
using System.Text.Json;
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
    private readonly ModelBackendOptions _options = options.Value;

    public async Task<ExportCorpusResponse> ExportCorpusAsync(ExportCorpusRequest request, CancellationToken cancellationToken)
    {
        var chunks = await repository.GetAllChunkContentsAsync(cancellationToken);
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
}
