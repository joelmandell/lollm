using System.Security.Cryptography;
using System.Text;
using DotNetCodingAgent.Contracts;
using Microsoft.Extensions.Logging;

namespace DotNetCodingAgent.Api.Services;

public sealed class KnowledgeIngestionService(
    IKnowledgeRepository repository,
    IWebContentFetcher contentFetcher,
    ILogger<KnowledgeIngestionService> logger)
{
    public async Task<OperationResponse> IngestAsync(string? url, bool force, CancellationToken cancellationToken)
    {
        var sources = await ResolveSourcesAsync(url, cancellationToken);
        if (sources.Count == 0)
        {
            return new OperationResponse(false, "No knowledge sources available.");
        }

        var refreshed = 0;
        var skipped = 0;
        var failed = 0;

        foreach (var source in sources)
        {
            try
            {
                if (!IsWebSource(source.Url))
                {
                    skipped++;
                    continue;
                }

                var (title, text) = await contentFetcher.FetchAsync(source.Url, cancellationToken);
                var hash = ComputeHash(text);
                if (!force && source.LastContentHash == hash)
                {
                    skipped++;
                    continue;
                }

                var chunks = TextChunker.Chunk(text);
                await repository.UpdateSourceChunksAsync(
                    source.Id,
                    title,
                    hash,
                    DateTimeOffset.UtcNow,
                    chunks,
                    cancellationToken);

                refreshed++;
            }
            catch (Exception ex)
            {
                failed++;
                logger.LogWarning(ex, "Failed to ingest source {SourceUrl}", source.Url);
            }
        }

        var success = failed == 0;
        var message = $"Indexed: {refreshed}, skipped: {skipped}, failed: {failed}.";
        return new OperationResponse(success, message);
    }

    private async Task<IReadOnlyList<KnowledgeSource>> ResolveSourcesAsync(string? url, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(url))
        {
            return await repository.GetActiveSourcesAsync(cancellationToken);
        }

        var source = await repository.AddSourceAsync(url, null, cancellationToken);
        return [source];
    }

    private static string ComputeHash(string text)
    {
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(text));
        return Convert.ToHexString(bytes);
    }

    private static bool IsWebSource(string url)
    {
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri))
        {
            return false;
        }

        return uri.Scheme is "http" or "https";
    }
}
