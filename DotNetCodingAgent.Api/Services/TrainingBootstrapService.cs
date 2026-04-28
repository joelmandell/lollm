using DotNetCodingAgent.Api.Options;
using DotNetCodingAgent.Contracts;
using Microsoft.Extensions.Options;

namespace DotNetCodingAgent.Api.Services;

public sealed class TrainingBootstrapService(
    IKnowledgeRepository knowledgeRepository,
    KnowledgeIngestionService ingestionService,
    ModelTrainingService modelTrainingService,
    IOptions<KnowledgeOptions> knowledgeOptions)
{
    public async Task<BootstrapTrainingResponse> BootstrapAsync(
        BootstrapTrainingRequest request,
        CancellationToken cancellationToken)
    {
        if (request.SeedDefaults)
        {
            foreach (var url in knowledgeOptions.Value.SeedUrls)
            {
                await knowledgeRepository.AddSourceAsync(url, null, cancellationToken);
            }
        }

        var sources = await knowledgeRepository.GetSourcesAsync(cancellationToken);
        var ingestion = await ingestionService.IngestAsync(null, request.ForceIngest, cancellationToken);
        var training = await modelTrainingService.TrainAsync(request.Epochs, cancellationToken);

        var success = ingestion.Success && training.Success;
        var message = $"Sources: {sources.Count}. Ingestion: {ingestion.Message} Training: {training.Message}";
        return new BootstrapTrainingResponse(success, message, sources.Count, ingestion, training);
    }
}
