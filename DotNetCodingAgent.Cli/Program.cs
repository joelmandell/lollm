using System.Net.Http.Json;
using DotNetCodingAgent.Contracts;

var apiBaseUrl = Environment.GetEnvironmentVariable("DOTNET_AGENT_API_BASE_URL") ?? "http://localhost:5050";
using var httpClient = new HttpClient { BaseAddress = new Uri(apiBaseUrl) };

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

    var prompt = string.Join(' ', arguments.Skip(1));
    var response = await PostAsync<ChatRequest, ChatResponse>("/api/chat", new ChatRequest(prompt));
    Console.WriteLine(response?.Answer ?? "No answer.");
}

async Task RunGenerateAsync(string[] arguments)
{
    if (arguments.Length < 2)
    {
        Console.WriteLine("Usage: generate \"task description\"");
        return;
    }

    var task = string.Join(' ', arguments.Skip(1));
    var response = await PostAsync<GenerateCodeRequest, GenerateCodeResponse>(
        "/api/agent/generate-code",
        new GenerateCodeRequest(task));

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
    Console.WriteLine("  generate \"coding task\"");
    Console.WriteLine("  add-source \"https://docs-url\"");
    Console.WriteLine("  list-sources");
    Console.WriteLine("  ingest [optional-url]");
}
