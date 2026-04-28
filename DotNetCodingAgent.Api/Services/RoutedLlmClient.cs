using DotNetCodingAgent.Api.Options;
using Microsoft.Extensions.Options;

namespace DotNetCodingAgent.Api.Services;

public sealed class RoutedLlmClient(
    IOptions<ModelBackendOptions> options,
    LocalMarkovLlmClient localClient,
    TransformerServiceLlmClient transformerClient) : ILlmClient
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

        try
        {
            return await transformerClient.GenerateAsync(systemPrompt, userPrompt, cancellationToken);
        }
        catch
        {
            return await localClient.GenerateAsync(systemPrompt, userPrompt, cancellationToken);
        }
    }
}
