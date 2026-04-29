using System.Net.Http.Json;
using DotNetCodingAgent.Contracts;
using Microsoft.AspNetCore.Components.Forms;

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

    public async Task<ModelBenchmarkResponse?> BenchmarkAsync(int maxCases = 10, CancellationToken cancellationToken = default)
    {
        var response = await httpClient.PostAsJsonAsync("/api/model/benchmark", new ModelBenchmarkRequest(maxCases), cancellationToken);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ModelBenchmarkResponse>(cancellationToken);
    }

    public async Task<BackendStatusResponse?> GetBackendStatusAsync(CancellationToken cancellationToken = default)
    {
        return await httpClient.GetFromJsonAsync<BackendStatusResponse>("/api/model/backend-status", cancellationToken);
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
}
