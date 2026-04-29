using DotNetCodingAgent.Api.Options;
using DotNetCodingAgent.Api.Services;
using DotNetCodingAgent.Contracts;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddOpenApi();
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowClients", policy =>
        policy.AllowAnyHeader().AllowAnyMethod().AllowAnyOrigin());
});

builder.Services.Configure<KnowledgeOptions>(builder.Configuration.GetSection("Knowledge"));
builder.Services.Configure<LocalModelOptions>(builder.Configuration.GetSection("LocalModel"));
builder.Services.Configure<ModelBackendOptions>(builder.Configuration.GetSection("ModelBackend"));

builder.Services.AddSingleton<PromptIntelligenceService>();
builder.Services.AddSingleton<CSharpCodeVerifier>();
builder.Services.AddSingleton<EvalFeedbackService>();
builder.Services.AddSingleton<LocalMarkovLlmClient>();
builder.Services.AddSingleton<ITrainableLlmClient>(provider => provider.GetRequiredService<LocalMarkovLlmClient>());
builder.Services.AddHttpClient<TransformerServiceLlmClient>((provider, client) =>
{
    var options = provider.GetRequiredService<IOptions<ModelBackendOptions>>().Value;
    client.BaseAddress = new Uri(options.TransformerBaseUrl);
    client.Timeout = TimeSpan.FromSeconds(Math.Max(10, options.TransformerTimeoutSeconds));
});
builder.Services.AddHttpClient<HostedOpenAiLlmClient>((provider, client) =>
{
    var options = provider.GetRequiredService<IOptions<ModelBackendOptions>>().Value;
    var openAiBase = options.OpenAiBaseUrl?.Trim();
    if (string.IsNullOrWhiteSpace(openAiBase))
    {
        openAiBase = "https://api.openai.com/v1";
    }

    if (!openAiBase.EndsWith('/'))
    {
        openAiBase += "/";
    }

    client.BaseAddress = new Uri(openAiBase);
    client.Timeout = TimeSpan.FromSeconds(Math.Max(10, options.TransformerTimeoutSeconds));
});
builder.Services.AddSingleton<ILlmClient, RoutedLlmClient>();
builder.Services.AddHttpClient<IWebContentFetcher, WebContentFetcher>();
builder.Services.AddSingleton<IKnowledgeRepository, SqliteKnowledgeRepository>();
builder.Services.AddSingleton<KnowledgeIngestionService>();
builder.Services.AddSingleton<ModelTrainingService>();
builder.Services.AddSingleton<ModelBenchmarkService>();
builder.Services.AddSingleton<CodingEvalService>();
builder.Services.AddSingleton<ModelStackService>();
builder.Services.AddSingleton<FeedbackCorpusService>();
builder.Services.AddSingleton<ImprovementCycleService>();
builder.Services.AddSingleton<ImprovementTrainingService>();
builder.Services.AddSingleton<ProjectZipTrainingService>();
builder.Services.AddSingleton<TrainingBootstrapService>();
builder.Services.AddSingleton<AgentOrchestrator>();
builder.Services.AddHostedService<KnowledgeRefreshWorker>();

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseCors("AllowClients");
app.UseHttpsRedirection();

await InitializeKnowledgeAsync(app);

var chatGroup = app.MapGroup("/api");

chatGroup.MapPost("/chat", async (
    ChatRequest request,
    AgentOrchestrator orchestrator,
    CancellationToken cancellationToken) =>
{
    var response = await orchestrator.ChatAsync(request, cancellationToken);
    return Results.Ok(response);
});

chatGroup.MapPost("/agent/generate-code", async (
    GenerateCodeRequest request,
    AgentOrchestrator orchestrator,
    CancellationToken cancellationToken) =>
{
    var response = await orchestrator.GenerateCodeAsync(request, cancellationToken);
    return Results.Ok(response);
});

var knowledgeGroup = app.MapGroup("/api/knowledge");

knowledgeGroup.MapGet("/sources", async (
    IKnowledgeRepository repository,
    CancellationToken cancellationToken) =>
{
    var sources = await repository.GetSourcesAsync(cancellationToken);
    var response = sources
        .Select(s => new KnowledgeSourceDto(s.Id, s.Url, s.Title, s.LastIndexedUtc, s.IsActive))
        .ToList();
    return Results.Ok(response);
});

knowledgeGroup.MapPost("/sources", async (
    AddKnowledgeSourceRequest request,
    IKnowledgeRepository repository,
    CancellationToken cancellationToken) =>
{
    var source = await repository.AddSourceAsync(request.Url, request.Title, cancellationToken);
    var response = new KnowledgeSourceDto(source.Id, source.Url, source.Title, source.LastIndexedUtc, source.IsActive);
    return Results.Ok(response);
});

knowledgeGroup.MapPost("/ingest", async (
    IngestKnowledgeRequest request,
    KnowledgeIngestionService ingestionService,
    CancellationToken cancellationToken) =>
{
    var response = await ingestionService.IngestAsync(request.Url, request.Force, cancellationToken);
    return Results.Ok(response);
});

knowledgeGroup.MapPost("/seed-defaults", async (
    IKnowledgeRepository repository,
    IOptions<KnowledgeOptions> options,
    CancellationToken cancellationToken) =>
{
    var added = 0;
    foreach (var seedUrl in options.Value.SeedUrls)
    {
        await repository.AddSourceAsync(seedUrl, null, cancellationToken);
        added++;
    }

    return Results.Ok(new OperationResponse(true, $"Seeded {added} default sources."));
});

var modelGroup = app.MapGroup("/api/model");

modelGroup.MapGet("/status", (ModelTrainingService trainingService) =>
{
    var status = trainingService.GetStatus();
    return Results.Ok(status);
});

modelGroup.MapPost("/train", async (
    TrainModelRequest request,
    ModelTrainingService trainingService,
    CancellationToken cancellationToken) =>
{
    var response = await trainingService.TrainAsync(request.Epochs, cancellationToken);
    return Results.Ok(response);
});

modelGroup.MapPost("/bootstrap", async (
    BootstrapTrainingRequest request,
    TrainingBootstrapService bootstrapService,
    CancellationToken cancellationToken) =>
{
    var response = await bootstrapService.BootstrapAsync(request, cancellationToken);
    return Results.Ok(response);
});

modelGroup.MapPost("/benchmark", async (
    ModelBenchmarkRequest request,
    ModelBenchmarkService benchmarkService,
    CancellationToken cancellationToken) =>
{
    var response = await benchmarkService.RunAsync(request.MaxCases, cancellationToken);
    return Results.Ok(response);
});

modelGroup.MapPost("/evaluate-coding", async (
    CodingEvalRequest request,
    CodingEvalService codingEvalService,
    CancellationToken cancellationToken) =>
{
    var response = await codingEvalService.RunAsync(request, cancellationToken);
    return Results.Ok(response);
});

modelGroup.MapPost("/export-corpus", async (
    ExportCorpusRequest request,
    ModelStackService modelStackService,
    CancellationToken cancellationToken) =>
{
    var response = await modelStackService.ExportCorpusAsync(request, cancellationToken);
    return Results.Ok(response);
});

modelGroup.MapGet("/backend-status", async (
    ModelStackService modelStackService,
    CancellationToken cancellationToken) =>
{
    var response = await modelStackService.GetBackendStatusAsync(cancellationToken);
    return Results.Ok(response);
});

modelGroup.MapGet("/feedback-status", async (
    EvalFeedbackService evalFeedbackService,
    CancellationToken cancellationToken) =>
{
    var response = await evalFeedbackService.GetStatusAsync(cancellationToken);
    return Results.Ok(response);
});

modelGroup.MapPost("/build-feedback-corpus", async (
    BuildFeedbackCorpusRequest request,
    FeedbackCorpusService feedbackCorpusService,
    CancellationToken cancellationToken) =>
{
    var response = await feedbackCorpusService.BuildAsync(request, cancellationToken);
    return Results.Ok(response);
});

modelGroup.MapPost("/run-improvement-cycle", async (
    RunImprovementCycleRequest request,
    ImprovementCycleService improvementCycleService,
    CancellationToken cancellationToken) =>
{
    var response = await improvementCycleService.RunAsync(request, cancellationToken);
    return Results.Ok(response);
});

modelGroup.MapPost("/run-improvement-training", async (
    RunImprovementTrainingRequest request,
    ImprovementTrainingService improvementTrainingService,
    CancellationToken cancellationToken) =>
{
    var response = await improvementTrainingService.RunAsync(request, cancellationToken);
    return Results.Ok(response);
});

modelGroup.MapGet("/improvement-training-status", (
    ImprovementTrainingService improvementTrainingService) =>
{
    var response = improvementTrainingService.GetStatus();
    return Results.Ok(response);
});

modelGroup.MapPost("/train-project-zip", async (
    HttpRequest httpRequest,
    ProjectZipTrainingService projectZipTrainingService,
    CancellationToken cancellationToken) =>
{
    if (!httpRequest.HasFormContentType)
    {
        return Results.BadRequest(new { error = "Expected multipart/form-data request." });
    }

    var form = await httpRequest.ReadFormAsync(cancellationToken);
    var projectTag = form["projectTag"].ToString();
    if (string.IsNullOrWhiteSpace(projectTag))
    {
        return Results.BadRequest(new { error = "projectTag is required." });
    }

    var epochs = 1;
    if (int.TryParse(form["epochs"].ToString(), out var parsedEpochs))
    {
        epochs = Math.Max(1, parsedEpochs);
    }

    var zipFile = form.Files.GetFile("zipFile");
    if (zipFile is null || zipFile.Length <= 0)
    {
        return Results.BadRequest(new { error = "zipFile is required." });
    }

    if (!zipFile.FileName.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
    {
        return Results.BadRequest(new { error = "Only .zip files are supported." });
    }

    await using var stream = zipFile.OpenReadStream();
    var response = await projectZipTrainingService.TrainFromZipAsync(stream, projectTag, epochs, cancellationToken);
    return Results.Ok(response);
});

app.MapGet("/v1/models", () =>
{
    var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
    return Results.Ok(new
    {
        @object = "list",
        data = new[]
        {
            new
            {
                id = "lollm-coder-001",
                @object = "model",
                created = now,
                owned_by = "lollm"
            },
            new
            {
                id = "lollm-rubberduck-001",
                @object = "model",
                created = now,
                owned_by = "lollm"
            }
        }
    });
});

app.MapPost("/v1/chat/completions", async (
    OpenAiChatCompletionsRequest request,
    AgentOrchestrator orchestrator,
    CancellationToken cancellationToken) =>
{
    var systemMessage = request.Messages?
        .LastOrDefault(m => string.Equals(m.Role, "system", StringComparison.OrdinalIgnoreCase))
        ?.Content;
    var userMessage = request.Messages?
        .LastOrDefault(m => string.Equals(m.Role, "user", StringComparison.OrdinalIgnoreCase))
        ?.Content;

    if (string.IsNullOrWhiteSpace(userMessage))
    {
        return Results.BadRequest(new { error = new { message = "No user message provided." } });
    }

    var composedPrompt = string.IsNullOrWhiteSpace(systemMessage)
        ? userMessage
        : $"{systemMessage}\n\n{userMessage}";

    var normalizedModel = (request.Model ?? "lollm-coder-001").Trim().ToLowerInvariant();
    var rubberDuckMode = normalizedModel.Contains("rubberduck", StringComparison.OrdinalIgnoreCase);

    var assistantOutput = IsCodePrompt(composedPrompt)
        ? (await orchestrator.GenerateCodeAsync(
            new GenerateCodeRequest(
                composedPrompt,
                UseKnowledge: true,
                MaxKnowledgeSnippets: 3,
                RubberDuckMode: rubberDuckMode),
            cancellationToken)).Code
        : (await orchestrator.ChatAsync(
            new ChatRequest(
                composedPrompt,
                UseKnowledge: true,
                MaxKnowledgeSnippets: 8,
                RubberDuckMode: rubberDuckMode),
            cancellationToken)).Answer;
    var completionId = $"chatcmpl-{Guid.NewGuid():N}";
    return Results.Ok(new
    {
        id = completionId,
        @object = "chat.completion",
        created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
        model = request.Model ?? "lollm-coder-001",
        choices = new[]
        {
            new
            {
                index = 0,
                message = new { role = "assistant", content = NormalizeAssistantText(assistantOutput) },
                finish_reason = "stop"
            }
        },
        usage = new
        {
            prompt_tokens = 0,
            completion_tokens = 0,
            total_tokens = 0
        }
    });
});

app.MapPost("/v1/completions", async (
    OpenAiCompletionsRequest request,
    AgentOrchestrator orchestrator,
    CancellationToken cancellationToken) =>
{
    var promptText = request.Prompt;
    if (string.IsNullOrWhiteSpace(promptText))
    {
        return Results.BadRequest(new { error = new { message = "No prompt provided." } });
    }

    var normalizedModel = (request.Model ?? "lollm-coder-001").Trim().ToLowerInvariant();
    var rubberDuckMode = normalizedModel.Contains("rubberduck", StringComparison.OrdinalIgnoreCase);

    var assistantOutput = IsCodePrompt(promptText)
        ? (await orchestrator.GenerateCodeAsync(
            new GenerateCodeRequest(
                promptText,
                UseKnowledge: true,
                MaxKnowledgeSnippets: 3,
                RubberDuckMode: rubberDuckMode),
            cancellationToken)).Code
        : (await orchestrator.ChatAsync(
            new ChatRequest(
                promptText,
                UseKnowledge: true,
                MaxKnowledgeSnippets: 8,
                RubberDuckMode: rubberDuckMode),
            cancellationToken)).Answer;
    var completionId = $"cmpl-{Guid.NewGuid():N}";
    return Results.Ok(new
    {
        id = completionId,
        @object = "text_completion",
        created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
        model = request.Model ?? "lollm-coder-001",
        choices = new[]
        {
            new
            {
                text = NormalizeAssistantText(assistantOutput),
                index = 0,
                finish_reason = "stop"
            }
        },
        usage = new
        {
            prompt_tokens = 0,
            completion_tokens = 0,
            total_tokens = 0
        }
    });
});

app.MapPost("/v1/responses", async (
    OpenAiResponsesRequest request,
    AgentOrchestrator orchestrator,
    CancellationToken cancellationToken) =>
{
    var input = request.Input?.Trim();
    if (string.IsNullOrWhiteSpace(input))
    {
        return Results.BadRequest(new { error = new { message = "No input provided." } });
    }

    var normalizedModel = (request.Model ?? "lollm-coder-001").Trim().ToLowerInvariant();
    var rubberDuckMode = normalizedModel.Contains("rubberduck", StringComparison.OrdinalIgnoreCase);
    var assistantOutput = IsCodePrompt(input)
        ? (await orchestrator.GenerateCodeAsync(
            new GenerateCodeRequest(
                input,
                UseKnowledge: true,
                MaxKnowledgeSnippets: 3,
                RubberDuckMode: rubberDuckMode),
            cancellationToken)).Code
        : (await orchestrator.ChatAsync(
            new ChatRequest(input, UseKnowledge: true, MaxKnowledgeSnippets: 8, RubberDuckMode: rubberDuckMode),
            cancellationToken)).Answer;

    return Results.Ok(new
    {
        id = $"resp-{Guid.NewGuid():N}",
        @object = "response",
        created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
        model = request.Model ?? "lollm-coder-001",
        output = new[]
        {
            new
            {
                @type = "message",
                role = "assistant",
                content = new[]
                {
                    new { @type = "output_text", text = NormalizeAssistantText(assistantOutput) }
                }
            }
        }
    });
});

app.Run();

static async Task InitializeKnowledgeAsync(WebApplication app)
{
    using var scope = app.Services.CreateScope();
    var repository = scope.ServiceProvider.GetRequiredService<IKnowledgeRepository>();
    var options = scope.ServiceProvider.GetRequiredService<IOptions<KnowledgeOptions>>();

    await repository.InitializeAsync(CancellationToken.None);

    foreach (var seedUrl in options.Value.SeedUrls)
    {
        await repository.AddSourceAsync(seedUrl, null, CancellationToken.None);
    }
}

static bool IsCodePrompt(string prompt)
{
    var lower = prompt.ToLowerInvariant();
    return lower.Contains("code")
        || lower.Contains("program.cs")
        || lower.Contains("c#")
        || lower.Contains("dotnet")
        || lower.Contains(".net")
        || lower.Contains("minimal api")
        || lower.Contains("endpoint")
        || lower.Contains("crud")
        || lower.Contains("class ")
        || lower.Contains("method ");
}

static string NormalizeAssistantText(string text)
{
    if (string.IsNullOrWhiteSpace(text))
    {
        return string.Empty;
    }

    var normalized = text.Trim();
    if (normalized.StartsWith("[LocalModel]", StringComparison.Ordinal))
    {
        normalized = normalized["[LocalModel]".Length..].TrimStart();
    }

    return normalized;
}

public sealed record OpenAiChatMessage(string Role, string Content);
public sealed record OpenAiChatCompletionsRequest(
    string? Model,
    IReadOnlyList<OpenAiChatMessage> Messages);

public sealed record OpenAiCompletionsRequest(
    string? Model,
    string Prompt);

public sealed record OpenAiResponsesRequest(
    string? Model,
    string? Input);
