namespace DotNetCodingAgent.Api.Services;

public interface IKnowledgeRepository
{
    Task InitializeAsync(CancellationToken cancellationToken);
    Task<IReadOnlyList<KnowledgeSource>> GetSourcesAsync(CancellationToken cancellationToken);
    Task<IReadOnlyList<KnowledgeSource>> GetActiveSourcesAsync(CancellationToken cancellationToken);
    Task<KnowledgeSource> AddSourceAsync(string url, string? title, CancellationToken cancellationToken);
    Task<IReadOnlyList<KnowledgeSnippet>> SearchAsync(string query, int limit, CancellationToken cancellationToken);
    Task<IReadOnlyList<string>> GetAllChunkContentsAsync(CancellationToken cancellationToken);
    Task<IReadOnlyList<KnowledgeSnippet>> SearchBySourceUrlPrefixAsync(string query, string sourceUrlPrefix, int limit, CancellationToken cancellationToken);
    Task<IReadOnlyList<string>> GetChunkContentsBySourceUrlPrefixAsync(string sourceUrlPrefix, CancellationToken cancellationToken);
    Task UpdateSourceChunksAsync(
        long sourceId,
        string title,
        string contentHash,
        DateTimeOffset indexedAtUtc,
        IReadOnlyList<string> chunks,
        CancellationToken cancellationToken);
}
