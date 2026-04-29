using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Api.Services;

public sealed class CodingEvalService(
    AgentOrchestrator orchestrator,
    CSharpCodeVerifier codeVerifier,
    EvalFeedbackService evalFeedbackService)
{
    private static readonly string[] DefaultPrompts =
    [
        "Create a .NET 10 minimal API for Todo items using EF Core with SQLite provider and data source hello.db. Return complete Program.cs only.",
        "Create a .NET 10 minimal API for Todo items using EF Core with PostgreSQL using UseNpgsql. Return complete Program.cs only.",
        "Create a Blazor JS interop sample that calls window.localStorage and include C# + JavaScript snippets."
    ];

    public async Task<CodingEvalResponse> RunAsync(CodingEvalRequest request, CancellationToken cancellationToken)
    {
        var prompts = request.Prompts is { Count: > 0 }
            ? request.Prompts.Where(p => !string.IsNullOrWhiteSpace(p)).ToList()
            : DefaultPrompts.ToList();

        var results = new List<CodingEvalCaseResult>(prompts.Count);
        foreach (var prompt in prompts)
        {
            var codeResponse = await orchestrator.GenerateCodeAsync(
                new GenerateCodeRequest(
                    Task: prompt,
                    Language: "csharp",
                    UseKnowledge: request.UseKnowledge,
                    MaxKnowledgeSnippets: request.MaxKnowledgeSnippets),
                cancellationToken);

            var metrics = codeResponse.Metrics ?? new CodeGenerationMetrics(0, 0, false, []);
            var verification = codeVerifier.Verify(prompt, codeResponse.Code);
            var score = ScoreEval(prompt, codeResponse.Code, verification, metrics);
            var notes = metrics.VerificationPassed
                ? "Verifier loop passed."
                : $"Verifier failed: {string.Join("; ", metrics.LastVerificationErrors.Take(2))}";

            results.Add(new CodingEvalCaseResult(
                Prompt: prompt,
                Score: score,
                VerificationPassed: verification.IsValid && metrics.VerificationPassed,
                VerificationAttempts: Math.Max(metrics.VerificationAttempts, 1),
                RepairIterationsUsed: metrics.RepairIterationsUsed,
                Notes: notes));
        }

        var average = results.Count == 0 ? 0 : (int)Math.Round(results.Average(r => r.Score));
        var response = new CodingEvalResponse(average, results.Count, results);
        await evalFeedbackService.RecordEvalRunAsync(request, response, cancellationToken);
        return response;
    }

    private static int ScoreEval(
        string prompt,
        string output,
        CodeVerificationResult verification,
        CodeGenerationMetrics metrics)
    {
        var score = 40;
        if (verification.IsValid)
        {
            score += 35;
        }
        else
        {
            score -= Math.Min(35, verification.Errors.Count * 10);
        }

        if (output.Contains("```csharp", StringComparison.OrdinalIgnoreCase))
        {
            score += 10;
        }

        if (prompt.Contains("sqlite", StringComparison.OrdinalIgnoreCase) &&
            output.Contains("UseSqlite(", StringComparison.OrdinalIgnoreCase))
        {
            score += 8;
        }

        if ((prompt.Contains("postgres", StringComparison.OrdinalIgnoreCase) ||
             prompt.Contains("npgsql", StringComparison.OrdinalIgnoreCase)) &&
            output.Contains("UseNpgsql(", StringComparison.OrdinalIgnoreCase))
        {
            score += 8;
        }

        if (metrics.RepairIterationsUsed <= 1)
        {
            score += 4;
        }

        return Math.Clamp(score, 0, 100);
    }
}
