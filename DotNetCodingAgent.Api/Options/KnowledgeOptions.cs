namespace DotNetCodingAgent.Api.Options;

public sealed class KnowledgeOptions
{
    public string DatabasePath { get; set; } = "knowledge.db";
    public int RefreshMinutes { get; set; } = 120;
    public List<string> SeedUrls { get; set; } =
    [
        "https://learn.microsoft.com/dotnet/csharp/language-reference/language-specification/introduction",
        "https://learn.microsoft.com/dotnet/csharp/",
        "https://learn.microsoft.com/dotnet/core/",
        "https://learn.microsoft.com/dotnet/api/"
    ];
}
