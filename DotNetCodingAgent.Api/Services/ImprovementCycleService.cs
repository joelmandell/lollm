using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Api.Services;

public sealed class ImprovementCycleService(
    CodingEvalService codingEvalService,
    FeedbackCorpusService feedbackCorpusService,
    EvalFeedbackService evalFeedbackService)
{
    public async Task<RunImprovementCycleResponse> RunAsync(
        RunImprovementCycleRequest request,
        CancellationToken cancellationToken)
    {
        var evalRequest = new CodingEvalRequest(
            Prompts: request.Prompts,
            UseKnowledge: request.UseKnowledge,
            MaxKnowledgeSnippets: request.MaxKnowledgeSnippets);
        var evaluation = await codingEvalService.RunAsync(evalRequest, cancellationToken);

        var corpusRequest = new BuildFeedbackCorpusRequest(
            MaxItems: request.FeedbackCorpusMaxItems,
            LowScoreThreshold: request.FeedbackLowScoreThreshold,
            IncludePassingSamples: request.IncludePassingSamplesInCorpus,
            IncludeJsonl: true,
            IncludeText: true);
        var feedbackCorpus = await feedbackCorpusService.BuildAsync(corpusRequest, cancellationToken);
        var feedbackStatus = await evalFeedbackService.GetStatusAsync(cancellationToken);

        var success = evaluation.CaseCount > 0 && feedbackCorpus.Success;
        return new RunImprovementCycleResponse(
            Success: success,
            Evaluation: evaluation,
            FeedbackCorpus: feedbackCorpus,
            FeedbackStatus: feedbackStatus,
            Message: $"Improvement cycle completed. Eval avg={evaluation.AverageScore}, corpus items={feedbackCorpus.SelectedItems}.");
    }
}
