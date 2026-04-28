namespace DotNetCodingAgent.Api.Services;

public sealed record KnowledgeSource(
    long Id,
    string Url,
    string Title,
    DateTimeOffset? LastIndexedUtc,
    string? LastContentHash,
    bool IsActive);

public sealed record KnowledgeSnippet(
    string SourceUrl,
    string SourceTitle,
    string Content);
