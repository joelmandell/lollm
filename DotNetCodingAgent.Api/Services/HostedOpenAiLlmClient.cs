using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using DotNetCodingAgent.Api.Options;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace DotNetCodingAgent.Api.Services;

public sealed class HostedOpenAiLlmClient(
    HttpClient httpClient,
    IOptions<ModelBackendOptions> options,
    ILogger<HostedOpenAiLlmClient> logger)
{
    private static readonly TimeSpan ChatCompletionsAttemptTimeout = TimeSpan.FromSeconds(4);
    private static readonly TimeSpan GenerateAttemptTimeout = TimeSpan.FromSeconds(14);
    private readonly ModelBackendOptions _options = options.Value;

    public async Task<string> GenerateAsync(string systemPrompt, string userPrompt, CancellationToken cancellationToken)
    {
        var messages = new[]
        {
            new { role = "system", content = systemPrompt },
            new { role = "user", content = userPrompt }
        };

        using var request = new HttpRequestMessage(HttpMethod.Post, "chat/completions");
        var apiKey = ResolveApiKey();
        if (!string.IsNullOrWhiteSpace(apiKey))
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
        }

        request.Content = JsonContent.Create(new
        {
            model = _options.OpenAiModel,
            messages,
            temperature = 0.2
        });

        HttpResponseMessage response;
        string body;
        try
        {
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(ChatCompletionsAttemptTimeout);
            response = await httpClient.SendAsync(request, timeoutCts.Token);
            body = await response.Content.ReadAsStringAsync(timeoutCts.Token);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            logger.LogWarning(
                "Hosted OpenAI chat/completions timed out after {TimeoutSeconds}s; using transformer fallback.",
                ChatCompletionsAttemptTimeout.TotalSeconds);
            return await GenerateViaTransformerEndpointAsync(systemPrompt, userPrompt, cancellationToken);
        }

        using (response)
        {
        if (!response.IsSuccessStatusCode)
        {
            if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
            {
                return await GenerateViaTransformerEndpointAsync(systemPrompt, userPrompt, cancellationToken);
            }

            throw new InvalidOperationException($"Hosted OpenAI backend failed ({response.StatusCode}): {body}");
        }

        using var document = JsonDocument.Parse(body);
        var root = document.RootElement;
        var content = root.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString();
        if (string.IsNullOrWhiteSpace(content))
        {
            throw new InvalidOperationException("Hosted OpenAI backend returned empty output.");
        }

        return content;
        }
    }

    private async Task<string> GenerateViaTransformerEndpointAsync(string systemPrompt, string userPrompt, CancellationToken cancellationToken)
    {
        string first;
        try
        {
            first = await GenerateViaTransformerOnceAsync(
                systemPrompt,
                userPrompt,
                maxTokens: 320,
                temperature: 0.25,
                topK: 72,
                repetitionPenalty: 1.12,
                minNewTokens: 48,
                cancellationToken);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            logger.LogWarning(
                "Transformer /generate attempt timed out after {TimeoutSeconds}s (first pass).",
                GenerateAttemptTimeout.TotalSeconds);
            return BuildTimeoutRescueResponse(
                "transformer.generate.first-pass-timeout",
                $"Timeout after {GenerateAttemptTimeout.TotalSeconds:0}s while calling /generate.");
        }

        if (!IsLowConfidenceFallback(first))
        {
            return first;
        }

        var retryPrompt = $"""
            {userPrompt}

            Hard requirements:
            - Do not emit fallback/disclaimer text.
            - Return concrete, directly usable output.
            - Prefer concise, specific code or steps.
            """;
        string second;
        try
        {
            second = await GenerateViaTransformerOnceAsync(
                systemPrompt,
                retryPrompt,
                maxTokens: 360,
                temperature: 0.2,
                topK: 64,
                repetitionPenalty: 1.10,
                minNewTokens: 40,
                cancellationToken);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            logger.LogWarning(
                "Transformer /generate attempt timed out after {TimeoutSeconds}s (retry pass).",
                GenerateAttemptTimeout.TotalSeconds);
            return BuildTimeoutRescueResponse(
                "transformer.generate.retry-timeout",
                $"Timeout after {GenerateAttemptTimeout.TotalSeconds:0}s while calling /generate retry.");
        }

        if (!IsLowConfidenceFallback(second))
        {
            return second;
        }

        return FormatModelError(
            "transformer.generate.low-confidence",
            "The local model is currently under-confident for this prompt. Please retry with tighter constraints (language, framework, output format) while retraining continues.");
    }

    private async Task<string> GenerateViaTransformerOnceAsync(
        string systemPrompt,
        string userPrompt,
        int maxTokens,
        double temperature,
        int topK,
        double repetitionPenalty,
        int minNewTokens,
        CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "generate");
        request.Content = JsonContent.Create(new
        {
            system_prompt = systemPrompt,
            user_prompt = userPrompt,
            max_tokens = maxTokens,
            temperature,
            top_k = topK,
            repetition_penalty = repetitionPenalty,
            min_new_tokens = minNewTokens
        });

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutCts.CancelAfter(GenerateAttemptTimeout);
        using var response = await httpClient.SendAsync(request, timeoutCts.Token);
        var body = await response.Content.ReadAsStringAsync(timeoutCts.Token);
        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException($"Hosted backend fallback /generate failed ({response.StatusCode}): {body}");
        }

        using var document = JsonDocument.Parse(body);
        var text = document.RootElement.GetProperty("text").GetString();
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new InvalidOperationException("Hosted backend fallback /generate returned empty output.");
        }

        return text;
    }

    private static bool IsLowConfidenceFallback(string text)
    {
        return text.Contains("Unable to produce a high-confidence answer", StringComparison.OrdinalIgnoreCase);
    }

    private string BuildTimeoutRescueResponse(string stage, string technicalCause)
    {
        return FormatModelError(
            stage,
            "The model backend timed out while generating. Retry with a tighter prompt (language/framework/output format).",
            technicalCause);
    }

    private string? ResolveApiKey()
    {
        return _options.OpenAiApiKey
               ?? Environment.GetEnvironmentVariable("OPENAI_API_KEY");
    }

    private string FormatModelError(string stage, string userMessage, string? technicalCause = null)
    {
        if (!_options.IncludeDetailedModelErrors)
        {
            return userMessage;
        }

        var details = technicalCause ?? "No additional details.";
        return $$"""
            {{userMessage}}
            [ModelError]
            Stage={{stage}}
            OpenAiBaseUrl={{_options.OpenAiBaseUrl}}
            OpenAiModel={{_options.OpenAiModel}}
            Details={{details}}
            """;
    }
}
