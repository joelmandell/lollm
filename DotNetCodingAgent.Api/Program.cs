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

builder.Services.Configure<LlmOptions>(builder.Configuration.GetSection("Llm"));
builder.Services.Configure<KnowledgeOptions>(builder.Configuration.GetSection("Knowledge"));

builder.Services.AddHttpClient<ILlmClient, OpenAiLlmClient>();
builder.Services.AddHttpClient<IWebContentFetcher, WebContentFetcher>();
builder.Services.AddSingleton<IKnowledgeRepository, SqliteKnowledgeRepository>();
builder.Services.AddSingleton<KnowledgeIngestionService>();
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
