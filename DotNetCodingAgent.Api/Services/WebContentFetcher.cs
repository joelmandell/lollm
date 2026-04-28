using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.IO.Compression;

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

    private static readonly HashSet<string> CodeExtensions =
    [
        ".cs", ".csx", ".vb", ".fs", ".json", ".xml", ".yml", ".yaml", ".md", ".txt",
        ".razor", ".ts", ".tsx", ".js", ".jsx", ".sql", ".props", ".targets", ".sln", ".csproj"
    ];

    public async Task<(string Title, string Text)> FetchAsync(string url, CancellationToken cancellationToken)
    {
        if (TryParseGitHubRepository(url, out var owner, out var repo, out var branch, out var rootPath))
        {
            return await FetchGitHubRepositoryAsync(owner, repo, branch, rootPath, cancellationToken);
        }

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

    private async Task<(string Title, string Text)> FetchGitHubRepositoryAsync(
        string owner,
        string repo,
        string? branch,
        string? rootPath,
        CancellationToken cancellationToken)
    {
        var effectiveBranch = string.IsNullOrWhiteSpace(branch)
            ? await ResolveDefaultBranchAsync(owner, repo, cancellationToken)
            : branch;

        var archiveUrl = $"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{effectiveBranch}";
        using var request = new HttpRequestMessage(HttpMethod.Get, archiveUrl);
        request.Headers.UserAgent.ParseAdd("DotNetCodingAgent/1.0");
        using var response = await httpClient.SendAsync(request, cancellationToken);
        response.EnsureSuccessStatusCode();

        await using var archiveStream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var zipArchive = new ZipArchive(archiveStream, ZipArchiveMode.Read, leaveOpen: false);

        var builder = new StringBuilder();
        var added = 0;
        foreach (var entry in zipArchive.Entries.OrderBy(e => e.FullName))
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (added >= 5000)
            {
                break;
            }

            if (entry.Length == 0 || entry.Length > 250_000)
            {
                continue;
            }

            var extension = Path.GetExtension(entry.FullName);
            if (!CodeExtensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
            {
                continue;
            }

            if (!string.IsNullOrWhiteSpace(rootPath) &&
                !entry.FullName.Contains($"/{rootPath.Trim('/')}/", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            await using var fileStream = entry.Open();
            using var reader = new StreamReader(fileStream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            var content = await reader.ReadToEndAsync(cancellationToken);
            if (string.IsNullOrWhiteSpace(content))
            {
                continue;
            }

            builder.AppendLine($"FILE: {entry.FullName}");
            builder.AppendLine(content.Length > 8_000 ? content[..8_000] : content);
            builder.AppendLine();
            added++;
        }

        var title = string.IsNullOrWhiteSpace(rootPath)
            ? $"{owner}/{repo}"
            : $"{owner}/{repo}:{rootPath}";
        var text = builder.Length == 0
            ? $"Repository {owner}/{repo} contains no supported files for ingestion."
            : builder.ToString();
        return (title, text);
    }

    private async Task<string> ResolveDefaultBranchAsync(string owner, string repo, CancellationToken cancellationToken)
    {
        var repoApiUrl = $"https://api.github.com/repos/{owner}/{repo}";
        using var request = new HttpRequestMessage(HttpMethod.Get, repoApiUrl);
        request.Headers.UserAgent.ParseAdd("DotNetCodingAgent/1.0");
        request.Headers.Accept.ParseAdd("application/vnd.github+json");
        using var response = await httpClient.SendAsync(request, cancellationToken);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var document = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);
        if (document.RootElement.TryGetProperty("default_branch", out var defaultBranch))
        {
            return defaultBranch.GetString() ?? "main";
        }

        return "main";
    }

    private static bool TryParseGitHubRepository(
        string url,
        out string owner,
        out string repo,
        out string? branch,
        out string? rootPath)
    {
        owner = string.Empty;
        repo = string.Empty;
        branch = null;
        rootPath = null;

        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri))
        {
            return false;
        }

        if (!uri.Host.Equals("github.com", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        var segments = uri.AbsolutePath.Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length < 2)
        {
            return false;
        }

        owner = segments[0];
        repo = segments[1].EndsWith(".git", StringComparison.OrdinalIgnoreCase)
            ? segments[1][..^4]
            : segments[1];

        if (segments.Length >= 4 && segments[2].Equals("tree", StringComparison.OrdinalIgnoreCase))
        {
            branch = segments[3];
            if (segments.Length >= 5)
            {
                rootPath = string.Join('/', segments.Skip(4));
            }
        }

        return true;
    }
}
