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

public sealed record TrainModelRequest(
    int Epochs = 1);

public sealed record TrainModelResponse(
    bool Success,
    string Message,
    int TotalTokens,
    int VocabularySize,
    DateTimeOffset? LastTrainedUtc);

public sealed record ModelStatusResponse(
    int TotalTokens,
    int VocabularySize,
    DateTimeOffset? LastTrainedUtc,
    string ModelPath);

public sealed record BootstrapTrainingRequest(
    bool SeedDefaults = true,
    bool ForceIngest = true,
    int Epochs = 1);

public sealed record BootstrapTrainingResponse(
    bool Success,
    string Message,
    int SourceCount,
    OperationResponse Ingestion,
    TrainModelResponse Training);

public sealed record ModelBenchmarkRequest(
    int MaxCases = 10);

public sealed record ModelBenchmarkCaseResult(
    string Prompt,
    int Score,
    string Notes);

public sealed record ModelBenchmarkResponse(
    int AverageScore,
    int CaseCount,
    IReadOnlyList<ModelBenchmarkCaseResult> Results);

public sealed record ExportCorpusRequest(
    bool IncludeJsonl = true,
    bool IncludeText = true);

public sealed record ExportCorpusResponse(
    bool Success,
    int ChunkCount,
    string? JsonlPath,
    string? TextPath,
    string Message);

public sealed record BackendStatusResponse(
    string Provider,
    bool TransformerHealthy,
    string TransformerBaseUrl);
