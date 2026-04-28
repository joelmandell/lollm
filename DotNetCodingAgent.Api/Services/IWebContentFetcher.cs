namespace DotNetCodingAgent.Api.Services;

public interface IWebContentFetcher
{
    Task<(string Title, string Text)> FetchAsync(string url, CancellationToken cancellationToken);
}
