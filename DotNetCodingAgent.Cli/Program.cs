using System.Net.Http.Json;
using System.Net.Http.Headers;
using DotNetCodingAgent.Contracts;

var apiBaseUrl = Environment.GetEnvironmentVariable("DOTNET_AGENT_API_BASE_URL") ?? "http://localhost:5101";
using var httpClient = new HttpClient { BaseAddress = new Uri(apiBaseUrl) };
httpClient.Timeout = TimeSpan.FromMinutes(30);

if (args.Length == 0)
{
    PrintHelp();
    return;
}

var command = args[0].ToLowerInvariant();

switch (command)
{
    case "chat":
        await RunChatAsync(args);
        break;
    case "generate":
        await RunGenerateAsync(args);
        break;
    case "add-source":
        await RunAddSourceAsync(args);
        break;
    case "list-sources":
        await RunListSourcesAsync();
        break;
    case "ingest":
        await RunIngestAsync(args);
        break;
    case "train":
        await RunTrainAsync(args);
        break;
    case "model-status":
        await RunModelStatusAsync();
        break;
    case "seed-defaults":
        await RunSeedDefaultsAsync();
        break;
    case "bootstrap":
        await RunBootstrapAsync(args);
        break;
    case "benchmark":
        await RunBenchmarkAsync(args);
        break;
    case "backend-status":
        await RunBackendStatusAsync();
        break;
    case "evaluate-coding":
        await RunEvaluateCodingAsync(args);
        break;
    case "feedback-status":
        await RunFeedbackStatusAsync();
        break;
    case "build-feedback-corpus":
        await RunBuildFeedbackCorpusAsync(args);
        break;
    case "run-improvement-cycle":
        await RunImprovementCycleAsync(args);
        break;
    case "export-corpus":
        await RunExportCorpusAsync(args);
        break;
    case "train-project":
        await RunTrainProjectAsync(args);
        break;
    default:
        Console.WriteLine($"Unknown command: {command}");
        PrintHelp();
        break;
}

async Task RunChatAsync(string[] arguments)
{
    if (arguments.Length < 2)
    {
        Console.WriteLine("Usage: chat \"your question\"");
        return;
    }

    var prompt = BuildCommandText(arguments);
    var projectTag = GetOptionValue(arguments, "--project");
    var rubberDuckMode = arguments.Any(a => string.Equals(a, "--rubberduck", StringComparison.OrdinalIgnoreCase));
    var response = await PostAsync<ChatRequest, ChatResponse>(
        "/api/chat",
        new ChatRequest(prompt, ProjectTag: projectTag, RubberDuckMode: rubberDuckMode));
    Console.WriteLine(response?.Answer ?? "No answer.");
}

async Task RunGenerateAsync(string[] arguments)
{
    if (arguments.Length < 2)
    {
        Console.WriteLine("Usage: generate \"task description\"");
        return;
    }

    var task = BuildCommandText(arguments);
    var projectTag = GetOptionValue(arguments, "--project");
    var rubberDuckMode = arguments.Any(a => string.Equals(a, "--rubberduck", StringComparison.OrdinalIgnoreCase));
    var response = await PostAsync<GenerateCodeRequest, GenerateCodeResponse>(
        "/api/agent/generate-code",
        new GenerateCodeRequest(task, ProjectTag: projectTag, RubberDuckMode: rubberDuckMode));

    if (response is null)
    {
        Console.WriteLine("No response.");
        return;
    }

    Console.WriteLine("PLAN:");
    Console.WriteLine(response.Plan);
    Console.WriteLine();
    Console.WriteLine("CODE:");
    Console.WriteLine(response.Code);
    if (response.Metrics is not null)
    {
        Console.WriteLine();
        Console.WriteLine("METRICS:");
        Console.WriteLine($"  Verification passed: {response.Metrics.VerificationPassed}");
        Console.WriteLine($"  Attempts: {response.Metrics.VerificationAttempts}");
        Console.WriteLine($"  Repair iterations: {response.Metrics.RepairIterationsUsed}");
        if (response.Metrics.LastVerificationErrors.Count > 0)
        {
            Console.WriteLine("  Last verification errors:");
            foreach (var error in response.Metrics.LastVerificationErrors.Take(3))
            {
                Console.WriteLine($"    - {error}");
            }
        }
    }
}

async Task RunAddSourceAsync(string[] arguments)
{
    if (arguments.Length < 2)
    {
        Console.WriteLine("Usage: add-source \"https://example.com/docs\"");
        return;
    }

    var url = arguments[1];
    var response = await PostAsync<AddKnowledgeSourceRequest, KnowledgeSourceDto>(
        "/api/knowledge/sources",
        new AddKnowledgeSourceRequest(url));

    Console.WriteLine(response is null
        ? "No response."
        : $"Added source: {response.Title} ({response.Url})");
}

async Task RunListSourcesAsync()
{
    var response = await httpClient.GetFromJsonAsync<IReadOnlyList<KnowledgeSourceDto>>("/api/knowledge/sources");
    if (response is null || response.Count == 0)
    {
        Console.WriteLine("No sources.");
        return;
    }

    foreach (var source in response)
    {
        Console.WriteLine($"{source.Id}: {source.Title}");
        Console.WriteLine($"  URL: {source.Url}");
        Console.WriteLine($"  Last Indexed: {(source.LastIndexedUtc?.ToString("u") ?? "Never")}");
    }
}

async Task RunIngestAsync(string[] arguments)
{
    var url = arguments.Length >= 2 ? arguments[1] : null;
    var response = await PostAsync<IngestKnowledgeRequest, OperationResponse>(
        "/api/knowledge/ingest",
        new IngestKnowledgeRequest(url, true));
    Console.WriteLine(response?.Message ?? "No response.");
}

async Task RunTrainAsync(string[] arguments)
{
    var epochs = 1;
    if (arguments.Length >= 2 && int.TryParse(arguments[1], out var parsedEpochs))
    {
        epochs = parsedEpochs;
    }

    var response = await PostAsync<TrainModelRequest, TrainModelResponse>(
        "/api/model/train",
        new TrainModelRequest(epochs));

    if (response is null)
    {
        Console.WriteLine("No response.");
        return;
    }

    Console.WriteLine(response.Message);
    Console.WriteLine($"Vocabulary: {response.VocabularySize}");
    Console.WriteLine($"Total tokens: {response.TotalTokens}");
}

async Task RunModelStatusAsync()
{
    var response = await httpClient.GetFromJsonAsync<ModelStatusResponse>("/api/model/status");
    if (response is null)
    {
        Console.WriteLine("No model status available.");
        return;
    }

    Console.WriteLine($"Model path: {response.ModelPath}");
    Console.WriteLine($"Vocabulary: {response.VocabularySize}");
    Console.WriteLine($"Total tokens: {response.TotalTokens}");
    Console.WriteLine($"Last trained: {(response.LastTrainedUtc?.ToString("u") ?? "Never")}");
}

async Task RunSeedDefaultsAsync()
{
    var response = await httpClient.PostAsync("/api/knowledge/seed-defaults", null);
    var payload = await response.Content.ReadAsStringAsync();
    if (!response.IsSuccessStatusCode)
    {
        Console.WriteLine($"Request failed ({(int)response.StatusCode}): {payload}");
        return;
    }

    var data = System.Text.Json.JsonSerializer.Deserialize<OperationResponse>(payload, new System.Text.Json.JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true
    });
    Console.WriteLine(data?.Message ?? "Default sources seeded.");
}

async Task RunBootstrapAsync(string[] arguments)
{
    var epochs = 1;
    if (arguments.Length >= 2 && int.TryParse(arguments[1], out var parsedEpochs))
    {
        epochs = parsedEpochs;
    }

    var response = await PostAsync<BootstrapTrainingRequest, BootstrapTrainingResponse>(
        "/api/model/bootstrap",
        new BootstrapTrainingRequest(true, true, epochs));

    if (response is null)
    {
        Console.WriteLine("No response.");
        return;
    }

    Console.WriteLine(response.Message);
    Console.WriteLine($"Sources: {response.SourceCount}");
    Console.WriteLine($"Training vocab: {response.Training.VocabularySize}");
    Console.WriteLine($"Training tokens: {response.Training.TotalTokens}");
}

async Task RunBenchmarkAsync(string[] arguments)
{
    var maxCases = 10;
    if (arguments.Length >= 2 && int.TryParse(arguments[1], out var parsedCases))
    {
        maxCases = parsedCases;
    }

    var response = await PostAsync<ModelBenchmarkRequest, ModelBenchmarkResponse>(
        "/api/model/benchmark",
        new ModelBenchmarkRequest(maxCases));

    if (response is null)
    {
        Console.WriteLine("No response.");
        return;
    }

    Console.WriteLine($"Average score: {response.AverageScore}");
    Console.WriteLine($"Cases: {response.CaseCount}");
    foreach (var result in response.Results)
    {
        Console.WriteLine($"- {result.Prompt}");
        Console.WriteLine($"  Score: {result.Score}");
        Console.WriteLine($"  Notes: {result.Notes}");
    }
}

async Task RunBackendStatusAsync()
{
    var response = await httpClient.GetFromJsonAsync<BackendStatusResponse>("/api/model/backend-status");
    if (response is null)
    {
        Console.WriteLine("No backend status available.");
        return;
    }

    Console.WriteLine($"Provider: {response.Provider}");
    Console.WriteLine($"Transformer URL: {response.TransformerBaseUrl}");
    Console.WriteLine($"Transformer healthy: {response.TransformerHealthy}");
}

async Task RunEvaluateCodingAsync(string[] arguments)
{
    var response = await PostAsync<CodingEvalRequest, CodingEvalResponse>(
        "/api/model/evaluate-coding",
        new CodingEvalRequest());

    if (response is null)
    {
        Console.WriteLine("No response.");
        return;
    }

    Console.WriteLine($"Average score: {response.AverageScore}");
    Console.WriteLine($"Cases: {response.CaseCount}");
    foreach (var result in response.Results)
    {
        Console.WriteLine($"- {result.Prompt}");
        Console.WriteLine($"  Score: {result.Score}");
        Console.WriteLine($"  Verification passed: {result.VerificationPassed}");
        Console.WriteLine($"  Attempts: {result.VerificationAttempts}, Repairs: {result.RepairIterationsUsed}");
        Console.WriteLine($"  Notes: {result.Notes}");
    }
}

async Task RunFeedbackStatusAsync()
{
    var response = await httpClient.GetFromJsonAsync<FeedbackStatusResponse>("/api/model/feedback-status");
    if (response is null)
    {
        Console.WriteLine("No feedback status available.");
        return;
    }

    Console.WriteLine($"Feedback directory: {response.FeedbackDirectory}");
    Console.WriteLine($"Generation feedback file: {response.GenerationFeedbackPath}");
    Console.WriteLine($"Eval runs file: {response.EvalRunsPath}");
    Console.WriteLine($"Generation entries: {response.GenerationFeedbackEntries}");
    Console.WriteLine($"Eval run entries: {response.EvalRunEntries}");
    Console.WriteLine($"Last updated: {(response.LastUpdatedUtc?.ToString("u") ?? "Never")}");
}

async Task RunBuildFeedbackCorpusAsync(string[] arguments)
{
    var maxItems = 500;
    if (arguments.Length >= 2 && int.TryParse(arguments[1], out var parsedMax))
    {
        maxItems = Math.Max(1, parsedMax);
    }

    var threshold = 65;
    if (arguments.Length >= 3 && int.TryParse(arguments[2], out var parsedThreshold))
    {
        threshold = Math.Clamp(parsedThreshold, 0, 100);
    }

    var response = await PostAsync<BuildFeedbackCorpusRequest, BuildFeedbackCorpusResponse>(
        "/api/model/build-feedback-corpus",
        new BuildFeedbackCorpusRequest(
            MaxItems: maxItems,
            LowScoreThreshold: threshold,
            IncludePassingSamples: false,
            IncludeJsonl: true,
            IncludeText: true));

    if (response is null)
    {
        Console.WriteLine("No response.");
        return;
    }

    Console.WriteLine(response.Message);
    Console.WriteLine($"Source generation entries: {response.SourceGenerationEntries}");
    Console.WriteLine($"Source eval entries: {response.SourceEvalEntries}");
    Console.WriteLine($"Selected items: {response.SelectedItems}");
    if (!string.IsNullOrWhiteSpace(response.JsonlPath))
    {
        Console.WriteLine($"JSONL: {response.JsonlPath}");
    }

    if (!string.IsNullOrWhiteSpace(response.TextPath))
    {
        Console.WriteLine($"Text: {response.TextPath}");
    }
}

async Task RunImprovementCycleAsync(string[] arguments)
{
    var maxItems = 500;
    if (arguments.Length >= 2 && int.TryParse(arguments[1], out var parsedMax))
    {
        maxItems = Math.Max(1, parsedMax);
    }

    var threshold = 65;
    if (arguments.Length >= 3 && int.TryParse(arguments[2], out var parsedThreshold))
    {
        threshold = Math.Clamp(parsedThreshold, 0, 100);
    }

    var response = await PostAsync<RunImprovementCycleRequest, RunImprovementCycleResponse>(
        "/api/model/run-improvement-cycle",
        new RunImprovementCycleRequest(
            FeedbackCorpusMaxItems: maxItems,
            FeedbackLowScoreThreshold: threshold,
            IncludePassingSamplesInCorpus: false));

    if (response is null)
    {
        Console.WriteLine("No response.");
        return;
    }

    Console.WriteLine(response.Message);
    Console.WriteLine($"Success: {response.Success}");
    Console.WriteLine($"Eval average: {response.Evaluation.AverageScore} ({response.Evaluation.CaseCount} cases)");
    Console.WriteLine($"Corpus items: {response.FeedbackCorpus.SelectedItems}");
    if (!string.IsNullOrWhiteSpace(response.FeedbackCorpus.JsonlPath))
    {
        Console.WriteLine($"Corpus JSONL: {response.FeedbackCorpus.JsonlPath}");
    }

    Console.WriteLine($"Feedback generation entries: {response.FeedbackStatus.GenerationFeedbackEntries}");
    Console.WriteLine($"Feedback eval entries: {response.FeedbackStatus.EvalRunEntries}");
}

async Task RunExportCorpusAsync(string[] arguments)
{
    var includeJsonl = true;
    var includeText = true;

    if (arguments.Length >= 2)
    {
        includeJsonl = arguments[1].Contains("jsonl", StringComparison.OrdinalIgnoreCase)
                       || arguments[1].Contains("both", StringComparison.OrdinalIgnoreCase);
        includeText = arguments[1].Contains("text", StringComparison.OrdinalIgnoreCase)
                      || arguments[1].Contains("both", StringComparison.OrdinalIgnoreCase);
    }

    var response = await PostAsync<ExportCorpusRequest, ExportCorpusResponse>(
        "/api/model/export-corpus",
        new ExportCorpusRequest(includeJsonl, includeText));

    if (response is null)
    {
        Console.WriteLine("No response.");
        return;
    }

    Console.WriteLine(response.Message);
    Console.WriteLine($"Chunks: {response.ChunkCount}");
    if (!string.IsNullOrWhiteSpace(response.JsonlPath))
    {
        Console.WriteLine($"JSONL: {response.JsonlPath}");
    }
    if (!string.IsNullOrWhiteSpace(response.TextPath))
    {
        Console.WriteLine($"Text: {response.TextPath}");
    }
}

async Task RunTrainProjectAsync(string[] arguments)
{
    if (arguments.Length < 3)
    {
        Console.WriteLine("Usage: train-project <projectTag> <zipPath> [epochs]");
        return;
    }

    var projectTag = arguments[1];
    var zipPath = arguments[2];
    var epochs = 1;
    if (arguments.Length >= 4 && int.TryParse(arguments[3], out var parsedEpochs))
    {
        epochs = Math.Max(1, parsedEpochs);
    }

    if (!File.Exists(zipPath))
    {
        Console.WriteLine($"Zip file not found: {zipPath}");
        return;
    }

    await using var fileStream = File.OpenRead(zipPath);
    using var fileContent = new StreamContent(fileStream);
    fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/zip");

    using var formData = new MultipartFormDataContent
    {
        { new StringContent(projectTag), "projectTag" },
        { new StringContent(epochs.ToString()), "epochs" },
        { fileContent, "zipFile", Path.GetFileName(zipPath) }
    };

    var response = await httpClient.PostAsync("/api/model/train-project-zip", formData);
    var payload = await response.Content.ReadAsStringAsync();
    if (!response.IsSuccessStatusCode)
    {
        Console.WriteLine($"Request failed ({(int)response.StatusCode}): {payload}");
        return;
    }

    var data = System.Text.Json.JsonSerializer.Deserialize<ProjectZipTrainingResponse>(payload, new System.Text.Json.JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true
    });

    if (data is null)
    {
        Console.WriteLine("No response.");
        return;
    }

    Console.WriteLine(data.Message);
    Console.WriteLine($"Project: {data.ProjectTag}");
    Console.WriteLine($"Files indexed: {data.FilesIndexed}");
    Console.WriteLine($"Chunks indexed: {data.ChunkCount}");
    Console.WriteLine($"Training chunks: {data.TrainedChunks}");
}

async Task<TResponse?> PostAsync<TRequest, TResponse>(string route, TRequest request)
{
    var response = await httpClient.PostAsJsonAsync(route, request);
    var payload = await response.Content.ReadAsStringAsync();
    if (!response.IsSuccessStatusCode)
    {
        Console.WriteLine($"Request failed ({(int)response.StatusCode}): {payload}");
        return default;
    }

    return System.Text.Json.JsonSerializer.Deserialize<TResponse>(payload, new System.Text.Json.JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true
    });
}

void PrintHelp()
{
    Console.WriteLine("DotNet Coding Agent CLI");
    Console.WriteLine($"API Base URL: {apiBaseUrl}");
    Console.WriteLine();
    Console.WriteLine("Commands:");
    Console.WriteLine("  chat \"question\"");
    Console.WriteLine("    optional: --project <tag> --rubberduck");
    Console.WriteLine("  generate \"coding task\"");
    Console.WriteLine("    optional: --project <tag> --rubberduck");
    Console.WriteLine("  add-source \"https://docs-or-github-repo-url\"");
    Console.WriteLine("  list-sources");
    Console.WriteLine("  ingest [optional-url]");
    Console.WriteLine("  train [epochs]");
    Console.WriteLine("  model-status");
    Console.WriteLine("  seed-defaults");
    Console.WriteLine("  bootstrap [epochs]");
    Console.WriteLine("  benchmark [maxCases]");
    Console.WriteLine("  backend-status");
    Console.WriteLine("  evaluate-coding");
    Console.WriteLine("  feedback-status");
    Console.WriteLine("  build-feedback-corpus [maxItems] [lowScoreThreshold]");
    Console.WriteLine("  run-improvement-cycle [maxItems] [lowScoreThreshold]");
    Console.WriteLine("  export-corpus [both|jsonl|text]");
    Console.WriteLine("  train-project <projectTag> <zipPath> [epochs]");
}

string? GetOptionValue(string[] arguments, string optionName)
{
    for (var i = 0; i < arguments.Length - 1; i++)
    {
        if (string.Equals(arguments[i], optionName, StringComparison.OrdinalIgnoreCase))
        {
            return arguments[i + 1];
        }
    }

    return null;
}

string BuildCommandText(string[] arguments)
{
    var parts = new List<string>();
    for (var i = 1; i < arguments.Length; i++)
    {
        var arg = arguments[i];
        if (string.Equals(arg, "--rubberduck", StringComparison.OrdinalIgnoreCase))
        {
            continue;
        }

        if (string.Equals(arg, "--project", StringComparison.OrdinalIgnoreCase))
        {
            i++;
            continue;
        }

        parts.Add(arg);
    }

    return string.Join(' ', parts);
}
