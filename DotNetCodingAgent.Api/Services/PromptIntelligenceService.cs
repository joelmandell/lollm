namespace DotNetCodingAgent.Api.Services;

public sealed class PromptIntelligenceService
{
    public PromptAnalysis Analyze(string prompt)
    {
        var normalized = prompt.Trim();
        var lower = normalized.ToLowerInvariant();

        var intent = PromptIntent.General;
        if (lower.Contains("generate code") || lower.Contains("write code") || lower.Contains("snippet") || lower.Contains("example"))
        {
            intent = PromptIntent.CodeGeneration;
        }
        else if (lower.Contains("plan") || lower.Contains("step-by-step") || lower.Contains("approach"))
        {
            intent = PromptIntent.Planning;
        }
        else if (lower.Contains("explain") || lower.Contains("why") || lower.Contains("what is") || lower.Contains("how does"))
        {
            intent = PromptIntent.Explanation;
        }

        var wantsHelloWorld = lower.Contains("hello world") || lower.Contains("hello-world");
        var wantsCrud = lower.Contains("crud");
        var wantsValidation = lower.Contains("validation") || lower.Contains("validate");
        var wantsEfCore = lower.Contains("ef core") || lower.Contains("entity framework");

        var mentionsCSharp = lower.Contains("c#") || lower.Contains("csharp") || lower.Contains(".net") || lower.Contains("dotnet");
        var mentionsJavaScript = lower.Contains("javascript") || lower.Contains(" js ") || lower.Contains("node") || lower.Contains("typescript");

        var language = PromptLanguage.Unknown;
        if (mentionsCSharp && mentionsJavaScript)
        {
            language = PromptLanguage.Both;
        }
        else if (mentionsCSharp)
        {
            language = PromptLanguage.CSharp;
        }
        else if (mentionsJavaScript)
        {
            language = PromptLanguage.JavaScript;
        }

        return new PromptAnalysis(
            normalized,
            lower,
            intent,
            language,
            wantsHelloWorld,
            wantsCrud,
            wantsValidation,
            wantsEfCore);
    }
}

public sealed record PromptAnalysis(
    string OriginalPrompt,
    string LowerPrompt,
    PromptIntent Intent,
    PromptLanguage Language,
    bool WantsHelloWorld,
    bool WantsCrud,
    bool WantsValidation,
    bool WantsEfCore);

public enum PromptIntent
{
    General = 0,
    Explanation = 1,
    Planning = 2,
    CodeGeneration = 3
}

public enum PromptLanguage
{
    Unknown = 0,
    CSharp = 1,
    JavaScript = 2,
    Both = 3
}
