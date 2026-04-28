namespace DotNetCodingAgent.Api.Services;

public interface ITrainableLlmClient : ILlmClient
{
    Task TrainAsync(IReadOnlyList<string> corpora, int epochs, CancellationToken cancellationToken);
    ModelSnapshot GetSnapshot();
}

public sealed record ModelSnapshot(
    int TotalTokens,
    int VocabularySize,
    DateTimeOffset? LastTrainedUtc,
    string ModelPath);
