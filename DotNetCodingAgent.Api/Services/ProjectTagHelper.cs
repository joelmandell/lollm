using System.Text.RegularExpressions;

namespace DotNetCodingAgent.Api.Services;

public static partial class ProjectTagHelper
{
    [GeneratedRegex("[^a-z0-9\\-_]+", RegexOptions.Compiled)]
    private static partial Regex InvalidCharsRegex();

    public static string Normalize(string projectTag)
    {
        if (string.IsNullOrWhiteSpace(projectTag))
        {
            throw new ArgumentException("Project tag is required.", nameof(projectTag));
        }

        var normalized = InvalidCharsRegex()
            .Replace(projectTag.Trim().ToLowerInvariant(), "-")
            .Trim('-');

        if (string.IsNullOrWhiteSpace(normalized))
        {
            throw new ArgumentException("Project tag must include alphanumeric characters.", nameof(projectTag));
        }

        return normalized;
    }

    public static string ToSourcePrefix(string projectTag) => $"project://{Normalize(projectTag)}/";
}
