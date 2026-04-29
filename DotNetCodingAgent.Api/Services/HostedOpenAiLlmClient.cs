using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using DotNetCodingAgent.Api.Options;
using Microsoft.Extensions.Options;

namespace DotNetCodingAgent.Api.Services;

public sealed class HostedOpenAiLlmClient(HttpClient httpClient, IOptions<ModelBackendOptions> options)
{
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

        using var response = await httpClient.SendAsync(request, cancellationToken);
        var body = await response.Content.ReadAsStringAsync(cancellationToken);
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

    private async Task<string> GenerateViaTransformerEndpointAsync(string systemPrompt, string userPrompt, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "generate");
        request.Content = JsonContent.Create(new
        {
            system_prompt = systemPrompt,
            user_prompt = userPrompt,
            max_tokens = 512,
            temperature = 0.3,
            top_k = 96,
            repetition_penalty = 1.14,
            min_new_tokens = 96
        });

        using var response = await httpClient.SendAsync(request, cancellationToken);
        var body = await response.Content.ReadAsStringAsync(cancellationToken);
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

    private string? ResolveApiKey()
    {
        return _options.OpenAiApiKey
               ?? Environment.GetEnvironmentVariable("OPENAI_API_KEY");
    }
}
