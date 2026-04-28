using System.Text;
using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Api.Services;

public sealed class AgentOrchestrator(ILlmClient llmClient, IKnowledgeRepository knowledgeRepository)
{
    public async Task<ChatResponse> ChatAsync(ChatRequest request, CancellationToken cancellationToken)
    {
        var snippets = await GetSnippetsAsync(request.Prompt, request.UseKnowledge, request.MaxKnowledgeSnippets, cancellationToken);
        var knowledgeBlock = BuildKnowledgeBlock(snippets);

        var systemPrompt = """
            You are a senior .NET 10 and C# assistant.
            Use provided documentation snippets when available and cite URLs inline.
            Prefer exact, practical C# and .NET guidance.
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
            $"Knowledge snippets used: {snippets.Count}");
    }

    public async Task<GenerateCodeResponse> GenerateCodeAsync(GenerateCodeRequest request, CancellationToken cancellationToken)
    {
        var snippets = await GetSnippetsAsync(request.Task, request.UseKnowledge, request.MaxKnowledgeSnippets, cancellationToken);
        var knowledgeBlock = BuildKnowledgeBlock(snippets);

        var planningSystemPrompt = """
            You are an agentic planner for .NET engineering work.
            Produce a concise implementation plan.
            """;

        var planningPrompt = $"""
            Create a step-by-step plan to implement this task in {request.Language}:
            {request.Task}

            Relevant docs:
            {knowledgeBlock}
            """;

        var plan = await llmClient.GenerateAsync(planningSystemPrompt, planningPrompt, cancellationToken);

        var codingSystemPrompt = """
            You are a principal C# and .NET 10 engineer.
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
        CancellationToken cancellationToken)
    {
        if (!useKnowledge)
        {
            return [];
        }

        return await knowledgeRepository.SearchAsync(query, Math.Max(1, limit), cancellationToken);
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
}
