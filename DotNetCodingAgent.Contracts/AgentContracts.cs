namespace DotNetCodingAgent.Contracts;

public sealed record ChatRequest(
    string Prompt,
    string? ConversationId = null,
    bool UseKnowledge = true,
    int MaxKnowledgeSnippets = 6,
    string? ProjectTag = null,
    bool RubberDuckMode = false);

public sealed record ChatResponse(
    string Answer,
    string ConversationId,
    IReadOnlyList<string> UsedSources,
    string? AgentNotes = null);

public sealed record GenerateCodeRequest(
    string Task,
    string Language = "csharp",
    bool UseKnowledge = true,
    int MaxKnowledgeSnippets = 8,
    string? ProjectTag = null,
    bool RubberDuckMode = false);

public sealed record GenerateCodeResponse(
    string Plan,
    string Code,
    string Explanation,
    IReadOnlyList<string> UsedSources,
    CodeGenerationMetrics? Metrics = null);

public sealed record CodeGenerationMetrics(
    int VerificationAttempts,
    int RepairIterationsUsed,
    bool VerificationPassed,
    IReadOnlyList<string> LastVerificationErrors);

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

public sealed record CodingEvalRequest(
    IReadOnlyList<string>? Prompts = null,
    bool UseKnowledge = true,
    int MaxKnowledgeSnippets = 8);

public sealed record CodingEvalCaseResult(
    string Prompt,
    int Score,
    bool VerificationPassed,
    int VerificationAttempts,
    int RepairIterationsUsed,
    string Notes);

public sealed record CodingEvalResponse(
    int AverageScore,
    int CaseCount,
    IReadOnlyList<CodingEvalCaseResult> Results);

public sealed record FeedbackStatusResponse(
    string FeedbackDirectory,
    string GenerationFeedbackPath,
    string EvalRunsPath,
    int GenerationFeedbackEntries,
    int EvalRunEntries,
    DateTimeOffset? LastUpdatedUtc);

public sealed record BuildFeedbackCorpusRequest(
    int MaxItems = 500,
    int LowScoreThreshold = 65,
    bool IncludePassingSamples = false,
    bool IncludeJsonl = true,
    bool IncludeText = true);

public sealed record BuildFeedbackCorpusResponse(
    bool Success,
    int SourceGenerationEntries,
    int SourceEvalEntries,
    int SelectedItems,
    string? JsonlPath,
    string? TextPath,
    string Message);

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

public sealed record ProjectZipTrainingResponse(
    bool Success,
    string ProjectTag,
    int FilesIndexed,
    int ChunkCount,
    int TrainedChunks,
    TrainModelResponse Training,
    string Message);
