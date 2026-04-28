using System.Net;
using System.Text.RegularExpressions;

namespace DotNetCodingAgent.Api.Services;

public sealed class WebContentFetcher(HttpClient httpClient) : IWebContentFetcher
{
    private static readonly Regex ScriptOrStyleRegex = new(
        "<(script|style)[^>]*?>.*?</\\1>",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    private static readonly Regex TagRegex = new(
        "<[^>]+>",
        RegexOptions.Compiled | RegexOptions.Singleline);

    private static readonly Regex WhitespaceRegex = new(
        "\\s+",
        RegexOptions.Compiled | RegexOptions.Singleline);

    private static readonly Regex TitleRegex = new(
        "<title[^>]*>(.*?)</title>",
        RegexOptions.Compiled | RegexOptions.IgnoreCase | RegexOptions.Singleline);

    public async Task<(string Title, string Text)> FetchAsync(string url, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.UserAgent.ParseAdd("DotNetCodingAgent/1.0");
        using var response = await httpClient.SendAsync(request, cancellationToken);
        response.EnsureSuccessStatusCode();

        var html = await response.Content.ReadAsStringAsync(cancellationToken);
        var title = ExtractTitle(html, url);
        var text = HtmlToText(html);
        return (title, text);
    }

    private static string ExtractTitle(string html, string fallback)
    {
        var match = TitleRegex.Match(html);
        if (!match.Success)
        {
            return fallback;
        }

        var title = WebUtility.HtmlDecode(match.Groups[1].Value).Trim();
        return string.IsNullOrWhiteSpace(title) ? fallback : title;
    }

    private static string HtmlToText(string html)
    {
        var withoutScripts = ScriptOrStyleRegex.Replace(html, " ");
        var withoutTags = TagRegex.Replace(withoutScripts, " ");
        var decoded = WebUtility.HtmlDecode(withoutTags);
        return WhitespaceRegex.Replace(decoded, " ").Trim();
    }
}
