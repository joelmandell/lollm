using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Api.Services;

public sealed class ModelBenchmarkService(ILlmClient llmClient, PromptIntelligenceService promptIntelligence)
{
    private static readonly string[] DefaultPrompts =
    [
        "hello world in c#",
        "show javascript hello world",
        "Generate .NET 10 minimal API product CRUD with EF Core and validation.",
        "Explain why DbContext should be scoped in ASP.NET Core."
    ];

    public async Task<ModelBenchmarkResponse> RunAsync(int maxCases, CancellationToken cancellationToken)
    {
        var prompts = DefaultPrompts.Take(Math.Max(1, Math.Min(maxCases, DefaultPrompts.Length))).ToList();
        var results = new List<ModelBenchmarkCaseResult>();

        foreach (var prompt in prompts)
        {
            var analysis = promptIntelligence.Analyze(prompt);
            var response = await llmClient.GenerateAsync("You are a coding model.", prompt, cancellationToken);
            var (score, notes) = ScoreResponse(analysis, response);
            results.Add(new ModelBenchmarkCaseResult(prompt, score, notes));
        }

        var average = results.Count == 0 ? 0 : (int)Math.Round(results.Average(r => r.Score));
        return new ModelBenchmarkResponse(average, results.Count, results);
    }

    private static (int Score, string Notes) ScoreResponse(PromptAnalysis analysis, string response)
    {
        var score = 0;
        var notes = new List<string>();
        var lower = response.ToLowerInvariant();

        if (!string.IsNullOrWhiteSpace(response))
        {
            score += 20;
            notes.Add("non-empty");
        }

        if (analysis.Intent == PromptIntent.CodeGeneration || analysis.WantsHelloWorld)
        {
            if (response.Contains("```"))
            {
                score += 20;
                notes.Add("code-fence");
            }
        }

        if (analysis.Language == PromptLanguage.CSharp)
        {
            if (lower.Contains("```csharp") || lower.Contains("console.writeline"))
            {
                score += 35;
                notes.Add("csharp-detected");
            }

            if (!analysis.WantsHelloWorld || !lower.Contains("```javascript"))
            {
                score += 15;
                notes.Add("language-precision");
            }
        }

        if (analysis.Language == PromptLanguage.JavaScript)
        {
            if (lower.Contains("```javascript") || lower.Contains("console.log"))
            {
                score += 35;
                notes.Add("javascript-detected");
            }
        }

        if (analysis.Intent == PromptIntent.Explanation && lower.Contains("because"))
        {
            score += 10;
            notes.Add("explanation-structure");
        }

        return (Math.Min(100, score), string.Join(", ", notes));
    }
}
