namespace DotNetCodingAgent.Api.Options;

public sealed class KnowledgeOptions
{
    public string DatabasePath { get; set; } = "knowledge.db";
    public int RefreshMinutes { get; set; } = 120;
    public List<string> SeedUrls { get; set; } =
    [
        "https://learn.microsoft.com/dotnet/core/whats-new/dotnet-10/overview",
        "https://learn.microsoft.com/dotnet/csharp/language-reference/language-specification/introduction",
        "https://learn.microsoft.com/dotnet/csharp/",
        "https://learn.microsoft.com/dotnet/core/",
        "https://learn.microsoft.com/dotnet/api/",
        "https://learn.microsoft.com/ef/core/",
        "https://learn.microsoft.com/dotnet/aspire/",
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
        "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference",
        "https://github.com/dotnet/runtime",
        "https://github.com/dotnet/efcore",
        "https://github.com/dotnet/aspire",
        "https://github.com/mdn/content/tree/main/files/en-us/web/javascript"
    ];
}
