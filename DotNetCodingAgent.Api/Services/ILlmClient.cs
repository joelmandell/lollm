namespace DotNetCodingAgent.Api.Services;

public interface ILlmClient
{
    Task<string> GenerateAsync(string systemPrompt, string userPrompt, CancellationToken cancellationToken);
}
