using System.Net.Http.Json;
using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Blazor.Services;

public sealed class AgentApiClient(HttpClient httpClient)
{
    public async Task<ChatResponse?> ChatAsync(ChatRequest request, CancellationToken cancellationToken = default)
    {
        var response = await httpClient.PostAsJsonAsync("/api/chat", request, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ChatResponse>(cancellationToken);
    }

    public async Task<GenerateCodeResponse?> GenerateCodeAsync(
        GenerateCodeRequest request,
        CancellationToken cancellationToken = default)
    {
        var response = await httpClient.PostAsJsonAsync("/api/agent/generate-code", request, cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<GenerateCodeResponse>(cancellationToken);
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
}
