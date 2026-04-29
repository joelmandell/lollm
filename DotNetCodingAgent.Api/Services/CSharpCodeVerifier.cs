using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace DotNetCodingAgent.Api.Services;

public sealed class CSharpCodeVerifier
{
    public CodeVerificationResult Verify(string task, string candidate)
    {
        var code = ExtractCode(candidate);
        if (string.IsNullOrWhiteSpace(code))
        {
            return new CodeVerificationResult(false, "Missing csharp code block.", []);
        }

        var syntaxErrors = ParseSyntaxErrors(code);
        var requirementErrors = CollectTaskConstraintErrors(task, code);
        var errors = syntaxErrors.Concat(requirementErrors).ToList();
        return new CodeVerificationResult(errors.Count == 0, BuildSummary(errors), errors);
    }

    private static List<string> ParseSyntaxErrors(string code)
    {
        var parseOptions = new CSharpParseOptions(LanguageVersion.Preview, DocumentationMode.Parse, SourceCodeKind.Regular);
        var syntaxTree = CSharpSyntaxTree.ParseText(code, parseOptions);
        return syntaxTree.GetDiagnostics()
            .Where(d => d.Severity == DiagnosticSeverity.Error)
            .Take(8)
            .Select(d => d.ToString())
            .ToList();
    }

    private static List<string> CollectTaskConstraintErrors(string task, string code)
    {
        var errors = new List<string>();
        var taskLower = task.ToLowerInvariant();
        var codeLower = code.ToLowerInvariant();

        var requestsEndpoint = taskLower.Contains("endpoint")
                               || taskLower.Contains("route")
                               || taskLower.Contains("mapget")
                               || taskLower.Contains("/hello-world")
                               || taskLower.Contains("hello-world");
        if (requestsEndpoint && !codeLower.Contains("mapget("))
        {
            errors.Add("Task requires a GET endpoint (MapGet missing).");
        }

        if (taskLower.Contains("minimal api") && !codeLower.Contains("mapget("))
        {
            errors.Add("Task requires minimal API endpoints (MapGet missing).");
        }

        if ((taskLower.Contains("/hello-world") || taskLower.Contains("hello-world")) &&
            !codeLower.Contains("/hello-world"))
        {
            errors.Add("Task requires an endpoint mapped exactly to /hello-world.");
        }

        if ((taskLower.Contains("query param") || taskLower.Contains("query parameter") || taskLower.Contains("get param")) &&
            !(codeLower.Contains("(string ") || codeLower.Contains("request.query")))
        {
            errors.Add("Task requires binding a query parameter in the endpoint handler.");
        }

        if (taskLower.Contains("hello world") && !codeLower.Contains("hello world"))
        {
            errors.Add("Task requires returning text that includes 'hello world'.");
        }

        if ((taskLower.Contains("ef core") || taskLower.Contains("entity framework")) &&
            (!codeLower.Contains("dbcontext") || !codeLower.Contains("adddbcontext")))
        {
            errors.Add("Task requires EF Core DbContext registration.");
        }

        if (taskLower.Contains("sqlite") && !codeLower.Contains("usesqlite("))
        {
            errors.Add("Task requires UseSqlite(...) configuration.");
        }

        if ((taskLower.Contains("postgres") || taskLower.Contains("postgresql") || taskLower.Contains("npgsql")) &&
            !codeLower.Contains("usenpgsql("))
        {
            errors.Add("Task requires UseNpgsql(...) configuration.");
        }

        if (taskLower.Contains("hello.db") && !codeLower.Contains("hello.db"))
        {
            errors.Add("Task requires the hello.db datasource.");
        }

        return errors;
    }

    private static string BuildSummary(IReadOnlyList<string> errors)
    {
        if (errors.Count == 0)
        {
            return "No compile plausibility issues found.";
        }

        var builder = new StringBuilder();
        for (var i = 0; i < errors.Count; i++)
        {
            builder.Append(i + 1).Append(". ").AppendLine(errors[i]);
        }

        return builder.ToString().TrimEnd();
    }

    private static string ExtractCode(string candidate)
    {
        var marker = "```csharp";
        var start = candidate.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (start < 0)
        {
            return candidate;
        }

        var codeStart = start + marker.Length;
        var codeFenceEnd = candidate.IndexOf("```", codeStart, StringComparison.OrdinalIgnoreCase);
        if (codeFenceEnd < 0)
        {
            return candidate[codeStart..].Trim();
        }

        return candidate[codeStart..codeFenceEnd].Trim();
    }
}

public sealed record CodeVerificationResult(
    bool IsValid,
    string Summary,
    IReadOnlyList<string> Errors);
