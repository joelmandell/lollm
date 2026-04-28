namespace DotNetCodingAgent.Contracts;

public sealed record ChatRequest(
    string Prompt,
    string? ConversationId = null,
    bool UseKnowledge = true,
    int MaxKnowledgeSnippets = 6);

public sealed record ChatResponse(
    string Answer,
    string ConversationId,
    IReadOnlyList<string> UsedSources,
    string? AgentNotes = null);

public sealed record GenerateCodeRequest(
    string Task,
    string Language = "csharp",
    bool UseKnowledge = true,
    int MaxKnowledgeSnippets = 8);

public sealed record GenerateCodeResponse(
    string Plan,
    string Code,
    string Explanation,
    IReadOnlyList<string> UsedSources);

public sealed record AddKnowledgeSourceRequest(
    string Url,
    string? Title = null);

public sealed record KnowledgeSourceDto(
    long Id,
    string Url,
    string Title,
    DateTimeOffset? LastIndexedUtc,
    bool IsActive);

public sealed record IngestKnowledgeRequest(
    string? Url = null,
    bool Force = false);

public sealed record OperationResponse(
    bool Success,
    string Message);
