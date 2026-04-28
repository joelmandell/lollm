using System.Net.Http.Json;
using System.Text.Json.Serialization;
using DotNetCodingAgent.Api.Options;
using Microsoft.Extensions.Options;

namespace DotNetCodingAgent.Api.Services;

public sealed class TransformerServiceLlmClient(HttpClient httpClient, IOptions<ModelBackendOptions> options)
{
    private readonly ModelBackendOptions _options = options.Value;

    public async Task<string> GenerateAsync(string systemPrompt, string userPrompt, CancellationToken cancellationToken)
    {
        var request = new TransformerGenerateRequest(systemPrompt, userPrompt, 600, 0.2);
        var response = await httpClient.PostAsJsonAsync("/generate", request, cancellationToken);
        var body = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException($"Transformer backend failed ({response.StatusCode}): {body}");
        }

        var payload = System.Text.Json.JsonSerializer.Deserialize<TransformerGenerateResponse>(
            body,
            new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        if (string.IsNullOrWhiteSpace(payload?.Text))
        {
            throw new InvalidOperationException("Transformer backend returned empty output.");
        }

        return payload.Text;
    }

    public async Task<bool> IsHealthyAsync(CancellationToken cancellationToken)
    {
        try
        {
            var response = await httpClient.GetAsync("/health", cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public string BaseUrl => _options.TransformerBaseUrl;

    private sealed record TransformerGenerateRequest(
        [property: JsonPropertyName("system_prompt")] string SystemPrompt,
        [property: JsonPropertyName("user_prompt")] string UserPrompt,
        [property: JsonPropertyName("max_tokens")] int MaxTokens,
        [property: JsonPropertyName("temperature")] double Temperature);

    private sealed record TransformerGenerateResponse(string Text);
}
