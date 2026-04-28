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

builder.Services.AddSingleton<ITrainableLlmClient, LocalMarkovLlmClient>();
builder.Services.AddSingleton<ILlmClient>(provider => provider.GetRequiredService<ITrainableLlmClient>());
builder.Services.AddHttpClient<IWebContentFetcher, WebContentFetcher>();
builder.Services.AddSingleton<IKnowledgeRepository, SqliteKnowledgeRepository>();
builder.Services.AddSingleton<KnowledgeIngestionService>();
builder.Services.AddSingleton<ModelTrainingService>();
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
