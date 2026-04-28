using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Api.Services;

public sealed class ModelTrainingService(IKnowledgeRepository knowledgeRepository, ITrainableLlmClient model)
{
    public async Task<TrainModelResponse> TrainAsync(int epochs, CancellationToken cancellationToken)
    {
        var corpora = await knowledgeRepository.GetAllChunkContentsAsync(cancellationToken);
        if (corpora.Count == 0)
        {
            var empty = model.GetSnapshot();
            return new TrainModelResponse(
                false,
                "No indexed corpus found. Ingest web or repository sources first.",
                empty.TotalTokens,
                empty.VocabularySize,
                empty.LastTrainedUtc);
        }

        await model.TrainAsync(corpora, epochs, cancellationToken);
        var trained = model.GetSnapshot();
        return new TrainModelResponse(
            true,
            $"Model trained with {corpora.Count} corpus chunks.",
            trained.TotalTokens,
            trained.VocabularySize,
            trained.LastTrainedUtc);
    }

    public ModelStatusResponse GetStatus()
    {
        var snapshot = model.GetSnapshot();
        return new ModelStatusResponse(
            snapshot.TotalTokens,
            snapshot.VocabularySize,
            snapshot.LastTrainedUtc,
            snapshot.ModelPath);
    }
}
