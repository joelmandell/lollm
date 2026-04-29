using System.Text;
using DotNetCodingAgent.Contracts;

namespace DotNetCodingAgent.Api.Services;

public sealed class AgentOrchestrator(
    ILlmClient llmClient,
    IKnowledgeRepository knowledgeRepository,
    PromptIntelligenceService promptIntelligence,
    CSharpCodeVerifier codeVerifier,
    EvalFeedbackService evalFeedbackService)
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
            analysis,
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

        var answer = await GenerateRefinedChatAnswerAsync(
            systemPrompt,
            userPrompt,
            analysis,
            normalizedProjectTag,
            request.RubberDuckMode,
            cancellationToken);
        return new ChatResponse(
            answer,
            request.ConversationId ?? Guid.NewGuid().ToString("N"),
            snippets.Select(s => s.SourceUrl).Distinct().ToList(),
            $"Knowledge snippets used: {snippets.Count}. ProjectTag: {(normalizedProjectTag ?? "none")}. RubberDuck: {request.RubberDuckMode}. Refinement: enabled.");
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
            analysis,
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

        var refined = await GenerateRefinedCodeAsync(
            codingSystemPrompt,
            codingPrompt,
            request,
            plan,
            cancellationToken);
        await evalFeedbackService.RecordGenerationAsync(
            request,
            refined.Metrics,
            refined.Output,
            cancellationToken);
        return new GenerateCodeResponse(
            plan,
            refined.Output,
            "Generated from planner + coder + self-critique refinement stages.",
            snippets.Select(s => s.SourceUrl).Distinct().ToList(),
            refined.Metrics);
    }

    private async Task<IReadOnlyList<KnowledgeSnippet>> GetSnippetsAsync(
        string query,
        bool useKnowledge,
        int limit,
        string? normalizedProjectTag,
        PromptAnalysis analysis,
        CancellationToken cancellationToken)
    {
        if (!useKnowledge)
        {
            return [];
        }

        using var retrievalCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        retrievalCts.CancelAfter(TimeSpan.FromSeconds(6));
        var retrievalToken = retrievalCts.Token;

        var effectiveLimit = Math.Max(1, limit);
        try
        {
            if (string.IsNullOrWhiteSpace(normalizedProjectTag))
            {
                var globalOnly = await knowledgeRepository.SearchAsync(query, effectiveLimit * 3, retrievalToken);
                return globalOnly
                    .DistinctBy(s => $"{s.SourceUrl}|{s.Content}")
                    .Where(s => ShouldKeepSnippetForPrompt(query, analysis, s))
                    .OrderByDescending(s => ScoreSnippet(query, analysis, s))
                    .Take(effectiveLimit)
                    .ToList();
            }

            var prefix = ProjectTagHelper.ToSourcePrefix(normalizedProjectTag);
            var projectLimit = Math.Max(1, (effectiveLimit + 1) / 2);
            var projectSnippets = await knowledgeRepository.SearchBySourceUrlPrefixAsync(query, prefix, projectLimit, retrievalToken);
            var globalSnippets = await knowledgeRepository.SearchAsync(query, effectiveLimit, retrievalToken);

            return projectSnippets
                .Concat(globalSnippets)
                .GroupBy(s => $"{s.SourceUrl}|{s.Content}")
                .Select(g => g.First())
                .Where(s => ShouldKeepSnippetForPrompt(query, analysis, s))
                .OrderByDescending(s => ScoreSnippet(query, analysis, s))
                .Take(effectiveLimit)
                .ToList();
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return [];
        }
    }

    private static bool ShouldKeepSnippetForPrompt(string query, PromptAnalysis analysis, KnowledgeSnippet snippet)
    {
        var queryLower = query.ToLowerInvariant();
        var url = snippet.SourceUrl.ToLowerInvariant();
        var title = snippet.SourceTitle.ToLowerInvariant();
        var content = snippet.Content.Length > 2000 ? snippet.Content[..2000].ToLowerInvariant() : snippet.Content.ToLowerInvariant();

        var dotnetHeavyPrompt = analysis.Language == PromptLanguage.CSharp
                                || queryLower.Contains(".net")
                                || queryLower.Contains("dotnet")
                                || queryLower.Contains("c#")
                                || queryLower.Contains("ef core")
                                || queryLower.Contains("dbcontext")
                                || queryLower.Contains("asp.net")
                                || queryLower.Contains("minimal api");

        if (!dotnetHeavyPrompt)
        {
            return true;
        }

        if (url.StartsWith("project://", StringComparison.Ordinal))
        {
            return true;
        }

        var allowsFrontendKnowledge = analysis.Language == PromptLanguage.JavaScript
                                      || analysis.Language == PromptLanguage.Both
                                      || queryLower.Contains("javascript")
                                      || queryLower.Contains("js interop")
                                      || queryLower.Contains("typescript")
                                      || queryLower.Contains("blazor")
                                      || queryLower.Contains("wasm")
                                      || queryLower.Contains("webassembly")
                                      || queryLower.Contains("browser")
                                      || queryLower.Contains("frontend");

        var explicitFrontendDoc = url.Contains("mdn")
                                  || content.Contains("javascript")
                                  || content.Contains("typescript")
                                  || content.Contains("react")
                                  || content.Contains("vue")
                                  || content.Contains("ember")
                                  || content.Contains("svelte");
        if (explicitFrontendDoc && !allowsFrontendKnowledge)
        {
            return false;
        }

        return url.Contains("learn.microsoft.com")
               || url.Contains("dotnet")
               || url.Contains("aspnet")
               || url.Contains("efcore")
               || url.Contains("npgsql")
               || (url.Contains("github.com/dotnet"))
               || title.Contains("dotnet")
               || title.Contains("c#")
               || title.Contains("ef core")
               || content.Contains("dbcontext")
               || content.Contains("entityframework")
               || content.Contains("minimal api")
               || content.Contains("builder.services.adddbcontext");
    }

    private static int ScoreSnippet(string query, PromptAnalysis analysis, KnowledgeSnippet snippet)
    {
        var score = 0;
        var queryLower = query.ToLowerInvariant();
        var url = snippet.SourceUrl.ToLowerInvariant();
        var title = snippet.SourceTitle.ToLowerInvariant();
        var content = snippet.Content.Length > 2000 ? snippet.Content[..2000].ToLowerInvariant() : snippet.Content.ToLowerInvariant();

        if (url.StartsWith("project://", StringComparison.Ordinal))
        {
            score += 80;
        }

        var dotnetHeavyPrompt = analysis.Language == PromptLanguage.CSharp
                                || queryLower.Contains(".net")
                                || queryLower.Contains("dotnet")
                                || queryLower.Contains("c#")
                                || queryLower.Contains("ef core")
                                || queryLower.Contains("dbcontext")
                                || queryLower.Contains("asp.net")
                                || queryLower.Contains("minimal api");

        if (dotnetHeavyPrompt)
        {
            var allowsFrontendKnowledge = analysis.Language == PromptLanguage.JavaScript
                                          || analysis.Language == PromptLanguage.Both
                                          || queryLower.Contains("javascript")
                                          || queryLower.Contains("js interop")
                                          || queryLower.Contains("typescript")
                                          || queryLower.Contains("blazor")
                                          || queryLower.Contains("wasm")
                                          || queryLower.Contains("webassembly")
                                          || queryLower.Contains("browser")
                                          || queryLower.Contains("frontend");

            if (url.Contains("learn.microsoft.com"))
            {
                score += 90;
            }

            if (url.Contains("dotnet") || url.Contains("aspnet") || url.Contains("efcore"))
            {
                score += 25;
            }

            if (title.Contains("dotnet") || title.Contains("aspnet") || title.Contains("ef core") || title.Contains("c#"))
            {
                score += 30;
            }

            if (content.Contains("dbcontext") || content.Contains("entityframework") || content.Contains("mapget(") || content.Contains("usesqlite") || content.Contains("usenpgsql"))
            {
                score += 25;
            }

            if (!allowsFrontendKnowledge &&
                (url.Contains("mdn") || content.Contains("ember") || content.Contains("svelte") || content.Contains("react") || content.Contains("vue") || content.Contains("javascript")))
            {
                score -= 80;
            }

            if (url.Contains("github.com") && !url.StartsWith("project://", StringComparison.Ordinal))
            {
                score -= 35;
            }
        }

        if (queryLower.Contains("sqlite") && (content.Contains("sqlite") || url.Contains("sqlite")))
        {
            score += 20;
        }

        if ((queryLower.Contains("postgres") || queryLower.Contains("postgresql") || queryLower.Contains("npgsql")) &&
            (content.Contains("npgsql") || content.Contains("postgres") || url.Contains("npgsql")))
        {
            score += 20;
        }

        return score;
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
            builder.AppendLine(snippet.Content.Length > 450 ? snippet.Content[..450] : snippet.Content);
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

    private async Task<string> GenerateRefinedChatAnswerAsync(
        string systemPrompt,
        string userPrompt,
        PromptAnalysis analysis,
        string? normalizedProjectTag,
        bool rubberDuckMode,
        CancellationToken cancellationToken)
    {
        var fast = await llmClient.GenerateAsync(systemPrompt, userPrompt, cancellationToken);
        if (!IsLowConfidenceSurface(fast))
        {
            return fast;
        }

        var retryPrompt = $"""
            {userPrompt}

            Hard requirements:
            - Do not emit fallback/disclaimer text.
            - Return concrete .NET guidance directly.
            - Be concise.
            """;
        return await llmClient.GenerateAsync(systemPrompt, retryPrompt, cancellationToken);
    }

    private async Task<RefinedCodeResult> GenerateRefinedCodeAsync(
        string codingSystemPrompt,
        string codingPrompt,
        GenerateCodeRequest request,
        string plan,
        CancellationToken cancellationToken)
    {
        var draftCandidates = await GenerateCandidatesAsync(codingSystemPrompt, codingPrompt, 3, cancellationToken);
        var draft = draftCandidates
            .OrderByDescending(c => ScoreCodeCandidate(request.Task, c))
            .First();

        var criticSystemPrompt = """
            You are a strict C# code reviewer.
            Find correctness, reliability, architecture-fit, and maintainability issues.
            Return a concise review with prioritized fixes.
            """;

        var criticPrompt = $"""
            Task:
            {request.Task}

            Language:
            {request.Language}

            Project tag:
            {(string.IsNullOrWhiteSpace(request.ProjectTag) ? "none" : request.ProjectTag)}

            RubberDuck mode:
            {request.RubberDuckMode}

            Plan:
            {plan}

            Candidate solution:
            {draft}

            Review checklist:
            - Correctness and compile plausibility
            - Missing edge-case handling
            - CancellationToken and async safety where relevant
            - Architecture-preserving design
            - Clear explanation after code
            """;

        var critique = await llmClient.GenerateAsync(criticSystemPrompt, criticPrompt, cancellationToken);

        var refineSystemPrompt = """
            You are a principal .NET coding assistant improving a draft.
            Produce production-oriented code with concise explanation.
            """;

        var refinePrompt = $"""
            Task:
            {request.Task}

            Initial plan:
            {plan}

            Draft solution:
            {draft}

            Review feedback:
            {critique}

            Output format:
            1) Start with a single code block containing the full improved solution.
            2) After the code block, add a short explanation.
            """;

        var refined = await llmClient.GenerateAsync(refineSystemPrompt, refinePrompt, cancellationToken);
        var refinedScore = ScoreCodeCandidate(request.Task, refined);

        var rescueSystemPrompt = """
            You are a senior .NET code generator.
            Return compile-ready, high-signal code output.
            """;

        var rescuePrompt = $"""
            Task:
            {request.Task}

            Candidate answer (too weak):
            {TruncateForPrompt(refined, 1400)}

            Regenerate with strict requirements:
            - First output a single csharp code block.
            - Match task entities/routes exactly (for example todo/todos when requested).
            - Include complete CRUD endpoints when CRUD is requested.
            - Avoid generic planning prose.
            - Add only a short explanation after code.
            - Enforce these task-specific constraints:
            {BuildTaskSpecificConstraints(request.Task)}
            """;

        var rescueCandidates = await GenerateCandidatesAsync(rescueSystemPrompt, rescuePrompt, 2, cancellationToken);
        var bestRescued = rescueCandidates
            .OrderByDescending(c => ScoreCodeCandidate(request.Task, c))
            .First();
        var rescuedScore = ScoreCodeCandidate(request.Task, bestRescued);
        var best = rescuedScore > refinedScore ? bestRescued : refined;
        var bestScore = Math.Max(rescuedScore, refinedScore);
        if (bestScore >= 65)
        {
            return await VerifyAndRepairCodeCandidateAsync(
                request,
                plan,
                best,
                cancellationToken);
        }

        var finalRescueSystemPrompt = """
            You are a strict .NET 10 code generator.
            Output must satisfy all requirements exactly and remain C# only.
            """;

        var finalRescuePrompt = $"""
            Task:
            {request.Task}

            Previous output was still low quality:
            {TruncateForPrompt(best, 1200)}

            Non-negotiable requirements:
            - Return one csharp code block first.
            - Use ASP.NET Core minimal API patterns for .NET tasks.
            - Do not include JavaScript frameworks or frontend stacks unless explicitly requested.
            - Enforce these task-specific constraints exactly:
            {BuildTaskSpecificConstraints(request.Task)}
            """;

        var finalRescueCandidates = await GenerateCandidatesAsync(finalRescueSystemPrompt, finalRescuePrompt, 2, cancellationToken);
        var finalRescued = finalRescueCandidates
            .OrderByDescending(c => ScoreCodeCandidate(request.Task, c))
            .First();
        var winner = ScoreCodeCandidate(request.Task, finalRescued) > bestScore ? finalRescued : best;
        return await VerifyAndRepairCodeCandidateAsync(
            request,
            plan,
            winner,
            cancellationToken);
    }

    private async Task<RefinedCodeResult> VerifyAndRepairCodeCandidateAsync(
        GenerateCodeRequest request,
        string plan,
        string initialCandidate,
        CancellationToken cancellationToken)
    {
        const int maxRepairIterations = 2;
        var candidate = initialCandidate;
        CodeVerificationResult? lastVerification = null;

        for (var i = 0; i <= maxRepairIterations; i++)
        {
            var verification = codeVerifier.Verify(request.Task, candidate);
            lastVerification = verification;
            var score = ScoreCodeCandidate(request.Task, candidate);
            if (verification.IsValid && score >= 65)
            {
                return new RefinedCodeResult(
                    NormalizeGeneratedCodeOutput(candidate),
                    new CodeGenerationMetrics(
                        VerificationAttempts: i + 1,
                        RepairIterationsUsed: i,
                        VerificationPassed: true,
                        LastVerificationErrors: []));
            }

            if (i == maxRepairIterations)
            {
                return new RefinedCodeResult(
                    NormalizeGeneratedCodeOutput(candidate),
                    new CodeGenerationMetrics(
                        VerificationAttempts: i + 1,
                        RepairIterationsUsed: i,
                        VerificationPassed: false,
                        LastVerificationErrors: verification.Errors));
            }

            var repairSystemPrompt = """
                You are a .NET 10 repair agent.
                Fix code using diagnostics and keep edits minimal.
                Return exactly one csharp code block, then a short explanation.
                """;

            var repairPrompt = $"""
                Task:
                {request.Task}

                Plan:
                {plan}

                Current candidate:
                {TruncateForPrompt(candidate, 2600)}

                Verification diagnostics:
                {verification.Summary}

                Repair requirements:
                - Fix all listed errors.
                - Preserve correct parts of the current implementation.
                - Keep task-specific constraints:
                {BuildTaskSpecificConstraints(request.Task)}
                """;

            var repairCandidates = await GenerateCandidatesAsync(repairSystemPrompt, repairPrompt, 2, cancellationToken);
            candidate = repairCandidates
                .OrderByDescending(c =>
                {
                    var result = codeVerifier.Verify(request.Task, c);
                    var verificationBonus = result.IsValid ? 30 : -Math.Min(30, result.Errors.Count * 8);
                    return ScoreCodeCandidate(request.Task, c) + verificationBonus;
                })
                .First();
        }

        return new RefinedCodeResult(
            NormalizeGeneratedCodeOutput(candidate),
            new CodeGenerationMetrics(
                VerificationAttempts: maxRepairIterations + 1,
                RepairIterationsUsed: maxRepairIterations,
                VerificationPassed: lastVerification?.IsValid == true,
                LastVerificationErrors: lastVerification?.Errors ?? []));
    }

    private sealed record RefinedCodeResult(string Output, CodeGenerationMetrics Metrics);

    private static bool LooksLikeCodeRequest(string text)
    {
        var lower = text.ToLowerInvariant();
        return lower.Contains("code")
            || lower.Contains("program.cs")
            || lower.Contains("c#")
            || lower.Contains("dotnet")
            || lower.Contains(".net")
            || lower.Contains("minimal api")
            || lower.Contains("endpoint")
            || lower.Contains("crud")
            || lower.Contains("todo api");
    }

    private static int ScoreCodeCandidate(string task, string candidate)
    {
        var score = 0;
        var taskLower = task.ToLowerInvariant();
        var outputLower = candidate.ToLowerInvariant();

        if (outputLower.Contains("```csharp", StringComparison.Ordinal))
        {
            score += 35;
        }

        if (outputLower.Contains("mapget(", StringComparison.Ordinal))
        {
            score += 10;
        }

        if (outputLower.Contains("mappost(", StringComparison.Ordinal))
        {
            score += 10;
        }

        if (taskLower.Contains("crud") || taskLower.Contains("create") || taskLower.Contains("update") || taskLower.Contains("delete"))
        {
            if (outputLower.Contains("mapput(", StringComparison.Ordinal))
            {
                score += 10;
            }

            if (outputLower.Contains("mapdelete(", StringComparison.Ordinal))
            {
                score += 10;
            }
        }

        if (taskLower.Contains("todo"))
        {
            if (outputLower.Contains("todo", StringComparison.Ordinal))
            {
                score += 15;
            }

            if (outputLower.Contains("/todos", StringComparison.Ordinal))
            {
                score += 10;
            }
        }

        if (taskLower.Contains("ef core") || taskLower.Contains("entity framework"))
        {
            if (outputLower.Contains("dbcontext", StringComparison.Ordinal))
            {
                score += 15;
            }
            else
            {
                score -= 20;
            }
        }

        if (taskLower.Contains("sqlite"))
        {
            if (outputLower.Contains("usesqlite(", StringComparison.Ordinal))
            {
                score += 20;
            }
            else
            {
                score -= 30;
            }
        }

        if (taskLower.Contains("postgres") || taskLower.Contains("postgresql") || taskLower.Contains("npgsql"))
        {
            if (outputLower.Contains("usenpgsql(", StringComparison.Ordinal))
            {
                score += 20;
            }
            else
            {
                score -= 30;
            }
        }

        if (taskLower.Contains("hello.db"))
        {
            if (outputLower.Contains("hello.db", StringComparison.Ordinal))
            {
                score += 15;
            }
            else
            {
                score -= 25;
            }
        }

        if (taskLower.Contains(".net") || taskLower.Contains("dotnet") || taskLower.Contains("c#") || taskLower.Contains("minimal api"))
        {
            if (ContainsFrontendFrameworkTokens(outputLower))
            {
                score -= 45;
            }
        }

        if (outputLower.Contains("plan for:", StringComparison.Ordinal))
        {
            score -= 35;
        }

        if (outputLower.Length < 300)
        {
            score -= 10;
        }

        return score;
    }

    private static bool ContainsFrontendFrameworkTokens(string outputLower)
    {
        return outputLower.Contains("javascript", StringComparison.Ordinal)
            || outputLower.Contains("typescript", StringComparison.Ordinal)
            || outputLower.Contains("react", StringComparison.Ordinal)
            || outputLower.Contains("svelte", StringComparison.Ordinal)
            || outputLower.Contains("ember", StringComparison.Ordinal)
            || outputLower.Contains("vue", StringComparison.Ordinal)
            || outputLower.Contains("node.js", StringComparison.Ordinal)
            || outputLower.Contains("npm ", StringComparison.Ordinal);
    }

    private static int ScoreChatCandidate(string userPrompt, string candidate)
    {
        var score = 0;
        var promptLower = userPrompt.ToLowerInvariant();
        var outputLower = candidate.ToLowerInvariant();

        if (IsLowConfidenceSurface(candidate))
        {
            return -120;
        }

        if (LooksLikeCodeRequest(userPrompt))
        {
            score += ScoreCodeCandidate(userPrompt, candidate);
            return score;
        }

        if (!outputLower.Contains("plan for:", StringComparison.Ordinal))
        {
            score += 10;
        }

        if (outputLower.Length >= 220)
        {
            score += 10;
        }

        if ((promptLower.Contains(".net") || promptLower.Contains("dotnet") || promptLower.Contains("c#")) &&
            !ContainsFrontendFrameworkTokens(outputLower))
        {
            score += 10;
        }

        return score;
    }

    private static bool IsLowConfidenceSurface(string text)
    {
        return text.Contains("Unable to produce a high-confidence answer", StringComparison.OrdinalIgnoreCase);
    }

    private async Task<IReadOnlyList<string>> GenerateCandidatesAsync(
        string systemPrompt,
        string userPrompt,
        int count,
        CancellationToken cancellationToken)
    {
        var boundedCount = Math.Max(1, count);
        var candidates = new List<string>(boundedCount);
        for (var i = 0; i < boundedCount; i++)
        {
            var variantPrompt = $"""
                {userPrompt}

                Variation hint: exploration path {Guid.NewGuid():N}
                """;
            var candidate = await llmClient.GenerateAsync(systemPrompt, variantPrompt, cancellationToken);
            if (IsLowConfidenceSurface(candidate))
            {
                var constrainedPrompt = $"""
                    {userPrompt}

                    Hard requirements:
                    - Do not return fallback/disclaimer text.
                    - Return concrete output immediately.
                    """;
                candidate = await llmClient.GenerateAsync(systemPrompt, constrainedPrompt, cancellationToken);
            }
            candidates.Add(candidate);
        }

        return candidates;
    }

    private static string BuildTaskSpecificConstraints(string task)
    {
        var constraints = new List<string>();
        var lower = task.ToLowerInvariant();

        if (lower.Contains("todo"))
        {
            constraints.Add("- Use Todo/Todos naming and /todos routes.");
        }

        if (lower.Contains("ef core") || lower.Contains("entity framework"))
        {
            constraints.Add("- Include a DbContext and entity model wiring.");
        }

        if (lower.Contains("sqlite"))
        {
            constraints.Add("- Configure EF Core with UseSqlite(...).");
        }

        if (lower.Contains("postgres") || lower.Contains("postgresql") || lower.Contains("npgsql"))
        {
            constraints.Add("- Configure EF Core with UseNpgsql(...).");
        }

        if (lower.Contains("hello.db"))
        {
            constraints.Add("- Use the exact SQLite datasource file name: hello.db.");
        }

        if (lower.Contains("minimal api"))
        {
            constraints.Add("- Use MapGet/MapPost/MapPut/MapDelete minimal API endpoints.");
        }

        if (constraints.Count == 0)
        {
            constraints.Add("- Follow the prompt requirements exactly with compile-ready C# code.");
        }

        return string.Join('\n', constraints);
    }

    private static string TruncateForPrompt(string text, int maxChars)
    {
        if (string.IsNullOrWhiteSpace(text) || text.Length <= maxChars)
        {
            return text;
        }

        return text[..maxChars];
    }

    private static string NormalizeGeneratedCodeOutput(string candidate)
    {
        if (string.IsNullOrWhiteSpace(candidate))
        {
            return string.Empty;
        }

        var marker = "```csharp";
        var start = candidate.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (start < 0)
        {
            return candidate.Length > 4000 ? candidate[..4000] : candidate;
        }

        var codeFenceEnd = candidate.IndexOf("```", start + marker.Length, StringComparison.OrdinalIgnoreCase);
        if (codeFenceEnd < 0)
        {
            return candidate.Length > 4000 ? candidate[..4000] : candidate;
        }

        var codeBlock = candidate[start..(codeFenceEnd + 3)];
        var remainder = candidate[(codeFenceEnd + 3)..].Trim();
        if (string.IsNullOrWhiteSpace(remainder))
        {
            return codeBlock;
        }

        var sanitizedRemainder = remainder
            .Replace("Candidate answer (too weak):", string.Empty, StringComparison.OrdinalIgnoreCase)
            .Replace("Previous output was still low quality:", string.Empty, StringComparison.OrdinalIgnoreCase)
            .Trim();
        if (sanitizedRemainder.Length > 300)
        {
            sanitizedRemainder = sanitizedRemainder[..300].TrimEnd();
        }

        return $"{codeBlock}\n\n{sanitizedRemainder}";
    }
}
