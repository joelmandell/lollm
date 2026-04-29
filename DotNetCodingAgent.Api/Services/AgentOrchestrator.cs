using System.Text;
using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Api.Services;

public sealed class AgentOrchestrator(
    ILlmClient llmClient,
    IKnowledgeRepository knowledgeRepository,
    PromptIntelligenceService promptIntelligence)
{
    public async Task<ChatResponse> ChatAsync(ChatRequest request, CancellationToken cancellationToken)
    {
        var analysis = promptIntelligence.Analyze(request.Prompt);
        var normalizedProjectTag = NormalizeProjectTag(request.ProjectTag);
        var snippets = await GetSnippetsAsync(
            request.Prompt,
            request.UseKnowledge,
            request.MaxKnowledgeSnippets,
            normalizedProjectTag,
            cancellationToken);
        var knowledgeBlock = BuildKnowledgeBlock(snippets);
        var rubberDuckInstructions = request.RubberDuckMode
            ? """
              RubberDuck reasoning mode:
              1) State assumptions explicitly.
              2) Walk through the reasoning step-by-step.
              3) Highlight uncertainty and tradeoffs.
              4) Recommend the least disruptive architectural change.
              """
            : "RubberDuck reasoning mode: disabled.";

        var systemPrompt = $"""
            You are a senior .NET 10 and C# assistant.
            Use provided documentation snippets when available and cite URLs inline.
            Prefer exact, practical C# and .NET guidance.
            Intent: {analysis.Intent}
            Preferred language: {analysis.Language}
            Project tag: {(normalizedProjectTag ?? "none")}
            {rubberDuckInstructions}
            """;

        var userPrompt = $"""
            Question:
            {request.Prompt}

            Documentation context:
            {knowledgeBlock}
            """;

        var answer = await llmClient.GenerateAsync(systemPrompt, userPrompt, cancellationToken);
        return new ChatResponse(
            answer,
            request.ConversationId ?? Guid.NewGuid().ToString("N"),
            snippets.Select(s => s.SourceUrl).Distinct().ToList(),
            $"Knowledge snippets used: {snippets.Count}. ProjectTag: {(normalizedProjectTag ?? "none")}. RubberDuck: {request.RubberDuckMode}.");
    }

    public async Task<GenerateCodeResponse> GenerateCodeAsync(GenerateCodeRequest request, CancellationToken cancellationToken)
    {
        var analysis = promptIntelligence.Analyze(request.Task);
        var normalizedProjectTag = NormalizeProjectTag(request.ProjectTag);
        var snippets = await GetSnippetsAsync(
            request.Task,
            request.UseKnowledge,
            request.MaxKnowledgeSnippets,
            normalizedProjectTag,
            cancellationToken);
        var knowledgeBlock = BuildKnowledgeBlock(snippets);
        var rubberDuckInstructions = request.RubberDuckMode
            ? """
              Use rubberduck reasoning to avoid architectural regressions:
              - State assumptions.
              - Compare alternatives.
              - Choose minimally disruptive changes.
              - Explain why this is safe for existing architecture.
              """
            : "Rubberduck reasoning: optional.";

        var planningSystemPrompt = """
            You are an agentic planner for .NET engineering work.
            Produce a concise implementation plan.
            """;

        var planningPrompt = $"""
            Create a step-by-step plan to implement this task in {request.Language}.
            Inferred intent: {analysis.Intent}
            Preferred language: {analysis.Language}
            Project tag: {(normalizedProjectTag ?? "none")}
            {rubberDuckInstructions}

            Task:
            {request.Task}

            Relevant docs:
            {knowledgeBlock}
            """;

        var plan = await llmClient.GenerateAsync(planningSystemPrompt, planningPrompt, cancellationToken);

        var codingSystemPrompt = """
            You are a principal coding engineer.
            Produce robust, production-ready code and explain key choices.
            """;

        var codingPrompt = $"""
            Task:
            {request.Task}

            Plan:
            {plan}

            Documentation context:
            {knowledgeBlock}

            Output format:
            1) Start with a single code block containing the full solution.
            2) After the code block, add a short explanation.
            """;

        var codeAndExplanation = await llmClient.GenerateAsync(codingSystemPrompt, codingPrompt, cancellationToken);
        return new GenerateCodeResponse(
            plan,
            codeAndExplanation,
            "Generated from planner + coder stages.",
            snippets.Select(s => s.SourceUrl).Distinct().ToList());
    }

    private async Task<IReadOnlyList<KnowledgeSnippet>> GetSnippetsAsync(
        string query,
        bool useKnowledge,
        int limit,
        string? normalizedProjectTag,
        CancellationToken cancellationToken)
    {
        if (!useKnowledge)
        {
            return [];
        }

        var effectiveLimit = Math.Max(1, limit);
        if (string.IsNullOrWhiteSpace(normalizedProjectTag))
        {
            return await knowledgeRepository.SearchAsync(query, effectiveLimit, cancellationToken);
        }

        var prefix = ProjectTagHelper.ToSourcePrefix(normalizedProjectTag);
        var projectLimit = Math.Max(1, (effectiveLimit + 1) / 2);
        var projectSnippets = await knowledgeRepository.SearchBySourceUrlPrefixAsync(query, prefix, projectLimit, cancellationToken);
        var globalSnippets = await knowledgeRepository.SearchAsync(query, effectiveLimit, cancellationToken);

        return projectSnippets
            .Concat(globalSnippets)
            .GroupBy(s => $"{s.SourceUrl}|{s.Content}")
            .Select(g => g.First())
            .Take(effectiveLimit)
            .ToList();
    }

    private static string BuildKnowledgeBlock(IReadOnlyList<KnowledgeSnippet> snippets)
    {
        if (snippets.Count == 0)
        {
            return "No indexed documentation available.";
        }

        var builder = new StringBuilder();
        for (var i = 0; i < snippets.Count; i++)
        {
            var snippet = snippets[i];
            builder.AppendLine($"[{i + 1}] {snippet.SourceTitle} ({snippet.SourceUrl})");
            builder.AppendLine(snippet.Content.Length > 1200 ? snippet.Content[..1200] : snippet.Content);
            builder.AppendLine();
        }

        return builder.ToString();
    }

    private static string? NormalizeProjectTag(string? projectTag)
    {
        if (string.IsNullOrWhiteSpace(projectTag))
        {
            return null;
        }

        return ProjectTagHelper.Normalize(projectTag);
    }
}
