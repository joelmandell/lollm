using System.Net.Http.Json;
using DotNetCodingAgent.Contracts;
using Microsoft.AspNetCore.Components.Forms;

namespace DotNetCodingAgent.Blazor.Services;

public sealed class AgentApiClient(HttpClient httpClient)
{
    public async Task<ChatResponse?> ChatAsync(ChatRequest request, CancellationToken cancellationToken = default)
    {
        var model = request.RubberDuckMode ? "lollm-rubberduck-001" : "lollm-coder-001";
        var userPrompt = BuildPromptWithContext(
            request.Prompt,
            request.ProjectTag,
            request.UseKnowledge,
            request.MaxKnowledgeSnippets);
        var content = await RequestAssistantTextAsync(
            model,
            "You are a .NET 10 coding assistant. Return concrete, compile-ready C# guidance.",
            userPrompt,
            cancellationToken);

        return new ChatResponse(
            Answer: string.IsNullOrWhiteSpace(content) ? "No answer returned." : content,
            ConversationId: request.ConversationId ?? Guid.NewGuid().ToString("N"),
            UsedSources: [],
            AgentNotes: "Response via OpenAI-compatible endpoint.");
    }

    public async Task<GenerateCodeResponse?> GenerateCodeAsync(
        GenerateCodeRequest request,
        CancellationToken cancellationToken = default)
    {
        var model = request.RubberDuckMode ? "lollm-rubberduck-001" : "lollm-coder-001";
        var userTask = BuildPromptWithContext(
            request.Task,
            request.ProjectTag,
            request.UseKnowledge,
            request.MaxKnowledgeSnippets);

        var content = await RequestAssistantTextAsync(
            model,
            "You are a .NET 10 coding assistant. Return compile-ready C# only for requested scope.",
            """
            Generate solution output in this exact markdown shape:
            ## Plan
            - concise implementation steps
            ## Code
            ```csharp
            // complete code
            ```
            ## Explanation
            - key behavior notes

            Task:
            """ + Environment.NewLine + userTask,
            cancellationToken);

        var (plan, code, explanation) = ParseCodeResponse(content);
        return new GenerateCodeResponse(
            Plan: plan,
            Code: code,
            Explanation: explanation,
            UsedSources: [],
            Metrics: null);
    }

    public async Task<IReadOnlyList<KnowledgeSourceDto>> GetSourcesAsync(CancellationToken cancellationToken = default)
    {
        var response = await httpClient.GetFromJsonAsync<IReadOnlyList<KnowledgeSourceDto>>("/api/knowledge/sources", cancellationToken);
        return response ?? [];
    }

    public async Task<KnowledgeSourceDto?> AddSourceAsync(string url, CancellationToken cancellationToken = default)
    {
        var response = await httpClient.PostAsJsonAsync("/api/knowledge/sources", new AddKnowledgeSourceRequest(url), cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<KnowledgeSourceDto>(cancellationToken);
    }

    public async Task<OperationResponse?> IngestAsync(string? url, bool force = false, CancellationToken cancellationToken = default)
    {
        var response = await httpClient.PostAsJsonAsync("/api/knowledge/ingest", new IngestKnowledgeRequest(url, force), cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<OperationResponse>(cancellationToken);
    }

    public async Task<ModelStatusResponse?> GetModelStatusAsync(CancellationToken cancellationToken = default)
    {
        return await httpClient.GetFromJsonAsync<ModelStatusResponse>("/api/model/status", cancellationToken);
    }

    public async Task<TrainModelResponse?> TrainModelAsync(int epochs = 1, CancellationToken cancellationToken = default)
    {
        var response = await httpClient.PostAsJsonAsync("/api/model/train", new TrainModelRequest(epochs), cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<TrainModelResponse>(cancellationToken);
    }

    public async Task<OperationResponse?> SeedDefaultSourcesAsync(CancellationToken cancellationToken = default)
    {
        var response = await httpClient.PostAsync("/api/knowledge/seed-defaults", null, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<OperationResponse>(cancellationToken);
    }

    public async Task<BootstrapTrainingResponse?> BootstrapTrainingAsync(
        BootstrapTrainingRequest request,
        CancellationToken cancellationToken = default)
    {
        var response = await httpClient.PostAsJsonAsync("/api/model/bootstrap", request, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<BootstrapTrainingResponse>(cancellationToken);
    }

    public async Task<ModelBenchmarkResponse?> BenchmarkAsync(int maxCases = 10, CancellationToken cancellationToken = default)
    {
        var response = await httpClient.PostAsJsonAsync("/api/model/benchmark", new ModelBenchmarkRequest(maxCases), cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ModelBenchmarkResponse>(cancellationToken);
    }

    public async Task<CodingEvalResponse?> EvaluateCodingAsync(
        CodingEvalRequest? request = null,
        CancellationToken cancellationToken = default)
    {
        request ??= new CodingEvalRequest();
        var response = await httpClient.PostAsJsonAsync("/api/model/evaluate-coding", request, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<CodingEvalResponse>(cancellationToken);
    }

    public async Task<BackendStatusResponse?> GetBackendStatusAsync(CancellationToken cancellationToken = default)
    {
        return await httpClient.GetFromJsonAsync<BackendStatusResponse>("/api/model/backend-status", cancellationToken);
    }

    public async Task<FeedbackStatusResponse?> GetFeedbackStatusAsync(CancellationToken cancellationToken = default)
    {
        return await httpClient.GetFromJsonAsync<FeedbackStatusResponse>("/api/model/feedback-status", cancellationToken);
    }

    public async Task<BuildFeedbackCorpusResponse?> BuildFeedbackCorpusAsync(
        BuildFeedbackCorpusRequest? request = null,
        CancellationToken cancellationToken = default)
    {
        request ??= new BuildFeedbackCorpusRequest();
        var response = await httpClient.PostAsJsonAsync("/api/model/build-feedback-corpus", request, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<BuildFeedbackCorpusResponse>(cancellationToken);
    }

    public async Task<RunImprovementCycleResponse?> RunImprovementCycleAsync(
        RunImprovementCycleRequest? request = null,
        CancellationToken cancellationToken = default)
    {
        request ??= new RunImprovementCycleRequest();
        var response = await httpClient.PostAsJsonAsync("/api/model/run-improvement-cycle", request, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<RunImprovementCycleResponse>(cancellationToken);
    }

    public async Task<RunImprovementTrainingResponse?> RunImprovementTrainingAsync(
        RunImprovementTrainingRequest? request = null,
        CancellationToken cancellationToken = default)
    {
        request ??= new RunImprovementTrainingRequest();
        var response = await httpClient.PostAsJsonAsync("/api/model/run-improvement-training", request, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<RunImprovementTrainingResponse>(cancellationToken);
    }

    public async Task<ImprovementTrainingStatusResponse?> GetImprovementTrainingStatusAsync(
        CancellationToken cancellationToken = default)
    {
        return await httpClient.GetFromJsonAsync<ImprovementTrainingStatusResponse>("/api/model/improvement-training-status", cancellationToken);
    }

    public async Task<ImprovementTrainingLogResponse?> GetImprovementTrainingLogAsync(
        int tailLines = 200,
        CancellationToken cancellationToken = default)
    {
        return await httpClient.GetFromJsonAsync<ImprovementTrainingLogResponse>($"/api/model/improvement-training-log?tailLines={Math.Clamp(tailLines, 1, 2000)}", cancellationToken);
    }

    public async Task<ImprovementTrainingStopResponse?> StopImprovementTrainingAsync(
        CancellationToken cancellationToken = default)
    {
        var response = await httpClient.PostAsync("/api/model/stop-improvement-training", null, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ImprovementTrainingStopResponse>(cancellationToken);
    }

    public async Task<ExportCorpusResponse?> ExportCorpusAsync(
        bool includeJsonl = true,
        bool includeText = true,
        CancellationToken cancellationToken = default)
    {
        var request = new ExportCorpusRequest(includeJsonl, includeText);
        var response = await httpClient.PostAsJsonAsync("/api/model/export-corpus", request, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ExportCorpusResponse>(cancellationToken);
    }

    public async Task<ProjectZipTrainingResponse?> TrainProjectZipAsync(
        string projectTag,
        IBrowserFile zipFile,
        int epochs = 1,
        CancellationToken cancellationToken = default)
    {
        await using var stream = zipFile.OpenReadStream(maxAllowedSize: 250 * 1024 * 1024, cancellationToken);
        using var streamContent = new StreamContent(stream);
        var contentType = string.IsNullOrWhiteSpace(zipFile.ContentType) ? "application/zip" : zipFile.ContentType;
        streamContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(contentType);

        using var formData = new MultipartFormDataContent
        {
            { new StringContent(projectTag), "projectTag" },
            { new StringContent(Math.Max(1, epochs).ToString()), "epochs" },
            { streamContent, "zipFile", zipFile.Name }
        };

        var response = await httpClient.PostAsync("/api/model/train-project-zip", formData, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ProjectZipTrainingResponse>(cancellationToken);
    }

    private sealed record OpenAiChatMessage(string Role, string Content);

    private sealed record OpenAiChatCompletionsRequest(
        string Model,
        IReadOnlyList<OpenAiChatMessage> Messages);

    private sealed record OpenAiChatChoice(OpenAiChatMessage Message);

    private sealed record OpenAiChatCompletionsResponse(IReadOnlyList<OpenAiChatChoice> Choices);

    private async Task<string> RequestAssistantTextAsync(
        string model,
        string systemPrompt,
        string userPrompt,
        CancellationToken cancellationToken)
    {
        var openAiRequest = new OpenAiChatCompletionsRequest(
            Model: model,
            Messages:
            [
                new OpenAiChatMessage("system", systemPrompt),
                new OpenAiChatMessage("user", userPrompt)
            ]);

        var response = await httpClient.PostAsJsonAsync("/v1/chat/completions", openAiRequest, cancellationToken);
        response.EnsureSuccessStatusCode();
        var completion = await response.Content.ReadFromJsonAsync<OpenAiChatCompletionsResponse>(cancellationToken);
        return completion?.Choices?.FirstOrDefault()?.Message?.Content?.Trim() ?? string.Empty;
    }

    private static string BuildPromptWithContext(
        string prompt,
        string? projectTag,
        bool useKnowledge,
        int maxKnowledgeSnippets)
    {
        var parts = new List<string>();
        if (!string.IsNullOrWhiteSpace(projectTag))
        {
            parts.Add($"ProjectTag: {projectTag}");
        }

        parts.Add($"UseKnowledge: {useKnowledge}");
        parts.Add($"MaxKnowledgeSnippets: {Math.Max(1, maxKnowledgeSnippets)}");
        parts.Add(prompt);
        return string.Join(Environment.NewLine + Environment.NewLine, parts);
    }

    private static (string Plan, string Code, string Explanation) ParseCodeResponse(string content)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            return ("No plan returned.", "No code returned.", "No explanation returned.");
        }

        var plan = ExtractSection(content, "## Plan", "## Code");
        var codeSection = ExtractSection(content, "## Code", "## Explanation");
        var explanation = ExtractSection(content, "## Explanation", null);
        var code = ExtractCodeBlock(codeSection);

        return (
            string.IsNullOrWhiteSpace(plan) ? "No plan returned." : plan,
            string.IsNullOrWhiteSpace(code) ? content : code,
            string.IsNullOrWhiteSpace(explanation) ? "No explanation returned." : explanation);
    }

    private static string ExtractSection(string text, string startMarker, string? endMarker)
    {
        var start = text.IndexOf(startMarker, StringComparison.OrdinalIgnoreCase);
        if (start < 0)
        {
            return string.Empty;
        }

        start += startMarker.Length;
        var end = endMarker is null
            ? -1
            : text.IndexOf(endMarker, start, StringComparison.OrdinalIgnoreCase);
        var section = end >= 0 ? text[start..end] : text[start..];
        return section.Trim();
    }

    private static string ExtractCodeBlock(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        var start = text.IndexOf("```", StringComparison.Ordinal);
        if (start < 0)
        {
            return text.Trim();
        }

        var afterFence = text.IndexOf('\n', start);
        if (afterFence < 0)
        {
            return text.Trim();
        }

        var end = text.IndexOf("```", afterFence + 1, StringComparison.Ordinal);
        if (end < 0)
        {
            return text[(afterFence + 1)..].Trim();
        }

        return text[(afterFence + 1)..end].Trim();
    }
}
