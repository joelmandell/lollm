using DotNetCodingAgent.Api.Options;
using Microsoft.Extensions.Options;

namespace DotNetCodingAgent.Api.Services;

public sealed class RoutedLlmClient(
    IOptions<ModelBackendOptions> options,
    LocalMarkovLlmClient localClient,
    TransformerServiceLlmClient transformerClient,
    HostedOpenAiLlmClient hostedOpenAiClient) : ILlmClient
{
    private readonly ModelBackendOptions _options = options.Value;

    public async Task<string> GenerateAsync(string systemPrompt, string userPrompt, CancellationToken cancellationToken = default)
    {
        var provider = (_options.Provider ?? "hybrid").Trim().ToLowerInvariant();

        if (provider == "local")
        {
            return await localClient.GenerateAsync(systemPrompt, userPrompt, cancellationToken);
        }

        if (provider == "transformer")
        {
            return await transformerClient.GenerateAsync(systemPrompt, userPrompt, cancellationToken);
        }

        if (provider == "openai")
        {
            return await hostedOpenAiClient.GenerateAsync(systemPrompt, userPrompt, cancellationToken);
        }

        // Hybrid mode is transformer-first without local fallback.
        if (provider == "hybrid")
        {
            return await transformerClient.GenerateAsync(systemPrompt, userPrompt, cancellationToken);
        }

        throw new InvalidOperationException(
            $"Unsupported ModelBackend:Provider '{_options.Provider}'. Supported values: local, transformer, hybrid, openai.");
    }
}
