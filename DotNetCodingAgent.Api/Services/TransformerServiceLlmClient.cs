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
        var request = new TransformerGenerateRequest(systemPrompt, userPrompt, DetermineMaxTokens(userPrompt), 0.2);
        HttpResponseMessage response;
        string body;
        try
        {
            response = await httpClient.PostAsJsonAsync("/generate", request, cancellationToken);
            body = await response.Content.ReadAsStringAsync(cancellationToken);
        }
        catch (OperationCanceledException ex) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException(
                $"Transformer backend request timed out after {_options.TransformerTimeoutSeconds}s " +
                $"(baseUrl={_options.TransformerBaseUrl}, endpoint=/generate).",
                ex);
        }
        catch (HttpRequestException ex)
        {
            throw new InvalidOperationException(
                $"Transformer backend request failed (baseUrl={_options.TransformerBaseUrl}, endpoint=/generate): {ex.Message}",
                ex);
        }

        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException(
                $"Transformer backend failed ({response.StatusCode}) " +
                $"[baseUrl={_options.TransformerBaseUrl}, endpoint=/generate]: {body}");
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

    private static int DetermineMaxTokens(string userPrompt)
    {
        if (string.IsNullOrWhiteSpace(userPrompt))
        {
            return 400;
        }

        var lower = userPrompt.ToLowerInvariant();
        var latencySensitive = userPrompt.Length <= 1200
                               && (lower.Contains("/hello-world", StringComparison.Ordinal)
                                   || lower.Contains("hello-world", StringComparison.Ordinal)
                                   || lower.Contains("minimal api", StringComparison.Ordinal)
                                   || lower.Contains("endpoint", StringComparison.Ordinal));
        return latencySensitive ? 240 : 600;
    }
}
