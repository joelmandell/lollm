import re


def detect_intent(prompt: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9#+.\s]", " ", prompt.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    def has_any(terms: list[str]) -> bool:
        return any(term in normalized for term in terms)

    csharp_context = has_any(["c#", "csharp", ".net", "dotnet", "asp.net", "program.main", "console.writeline"])
    if (has_any(["hello world", "first program"]) or re.search(r"\bfirst\b.*\bprogram\b", normalized)) and csharp_context:
        return "hello-world"
    if has_any(["entity framework", "ef core", "dbcontext", "modelbuilder"]):
        return "efcore-dbcontext"
    if has_any(["minimal api", "webapplication", "mapget"]) and (has_any(["asp.net", ".net", "dotnet"]) or has_any(["/todos", "results.ok", "todo"])):
        return "minimal-api"
    if has_any(["filter"]) and has_any(["list", "ints", "linq", "greater than"]):
        return "linq-filter"
    if has_any(["xunit", "unit test", "fact attribute", "[fact]", "assert.equal"]):
        return "xunit-test"
    if has_any(["httpclient", "getstringasync"]) or (has_any(["http", "https", "url", "uri", "fetch"]) and (has_any(["async", "task<string>", "task", "await"]) or csharp_context)):
        return "async-httpclient"
    if has_any(["groupby", "group by", "group ", "grouped by", "grouped"]) and has_any(["linq", "total", "sum", "amount", "select", "projection", "customer"]):
        return "linq-groupby"
    if has_any(["dependency injection", "iservicecollection", "addscoped", "addsingleton", "servicecollection", "scoped service", "di setup", "di "]):
        return "di-registration"
    if has_any(["cancellationtoken", "cancellation token", "cancellable", "cancelable"]) and has_any(["async", "await", "method", "repository"]):
        return "cancellation-token"
    if has_any(["record dto", "record model", "immutable dto"]) or (has_any(["record"]) and has_any(["dto", "request", "response"])):
        return "record-dto"
    if has_any(["repository pattern", "irepository", "interface and implementation"]) or (
        has_any(["repository"]) and has_any(["interface", "implementation"])) or (
        has_any(["repository"]) and has_any(["async methods", "product"])):
        return "repository-pattern"
    return None


def template_for_intent(intent: str | None) -> str | None:
    if intent == "xunit-test":
        return (
            "```csharp\n"
            "using Xunit;\n\n"
            "public sealed class MathTests\n"
            "{\n"
            "    [Fact]\n"
            "    public void Sum_ReturnsExpectedValue()\n"
            "    {\n"
            "        var result = 2 + 3;\n"
            "        Assert.Equal(5, result);\n"
            "    }\n"
            "}\n"
            "```"
        )
    if intent == "async-httpclient":
        return (
            "```csharp\n"
            "using System.Net.Http;\n\n"
            "public static async Task<string> FetchAsync(HttpClient httpClient, string url)\n"
            "{\n"
            "    var content = await httpClient.GetStringAsync(url);\n"
            "    return content;\n"
            "}\n"
            "```"
        )
    if intent == "linq-groupby":
        return (
            "```csharp\n"
            "var grouped = orders\n"
            "    .GroupBy(o => o.CustomerId)\n"
            "    .Select(g => new { CustomerId = g.Key, Total = g.Sum(x => x.Amount) })\n"
            "    .ToList();\n"
            "```"
        )
    if intent == "di-registration":
        return (
            "```csharp\n"
            "using Microsoft.Extensions.DependencyInjection;\n\n"
            "var services = new ServiceCollection();\n"
            "services.AddScoped<IEmailSender, SmtpEmailSender>();\n"
            "services.AddSingleton<IClock, SystemClock>();\n"
            "var provider = services.BuildServiceProvider();\n"
            "```\n"
            "\n"
            "```csharp\n"
            "public interface IEmailSender\n"
            "{\n"
            "    Task SendAsync(string to, string subject, string body, CancellationToken cancellationToken = default);\n"
            "}\n"
            "```"
        )
    if intent == "cancellation-token":
        return (
            "```csharp\n"
            "public static async Task<IReadOnlyList<string>> FetchNamesAsync(\n"
            "    IRepository repository,\n"
            "    CancellationToken cancellationToken)\n"
            "{\n"
            "    cancellationToken.ThrowIfCancellationRequested();\n"
            "    var items = await repository.GetAllAsync(cancellationToken);\n"
            "    return items.Select(x => x.Name).ToList();\n"
            "}\n"
            "```"
        )
    if intent == "record-dto":
        return (
            "```csharp\n"
            "public sealed record CreateOrderRequest(string CustomerId, IReadOnlyList<OrderLineDto> Lines);\n"
            "public sealed record OrderLineDto(string ProductId, int Quantity, decimal UnitPrice);\n"
            "public sealed record CreateOrderResponse(Guid OrderId, decimal Total);\n"
            "```"
        )
    if intent == "repository-pattern":
        return (
            "```csharp\n"
            "public interface IProductRepository\n"
            "{\n"
            "    Task<Product?> GetByIdAsync(int id, CancellationToken cancellationToken = default);\n"
            "    Task<IReadOnlyList<Product>> ListAsync(CancellationToken cancellationToken = default);\n"
            "}\n\n"
            "public sealed class ProductRepository(AppDbContext dbContext) : IProductRepository\n"
            "{\n"
            "    public Task<Product?> GetByIdAsync(int id, CancellationToken cancellationToken = default) =>\n"
            "        dbContext.Products.FirstOrDefaultAsync(p => p.Id == id, cancellationToken);\n\n"
            "    public async Task<IReadOnlyList<Product>> ListAsync(CancellationToken cancellationToken = default) =>\n"
            "        await dbContext.Products.AsNoTracking().ToListAsync(cancellationToken);\n"
            "}\n"
            "```"
        )
    if intent == "hello-world":
        return (
            "```csharp\n"
            "using System;\n\n"
            "public static class Program\n"
            "{\n"
            "    public static void Main()\n"
            "    {\n"
            "        Console.WriteLine(\"Hello, World!\");\n"
            "    }\n"
            "}\n"
            "```"
        )
    if intent == "efcore-dbcontext":
        return (
            "```csharp\n"
            "using Microsoft.EntityFrameworkCore;\n\n"
            "public sealed class Product\n"
            "{\n"
            "    public int Id { get; set; }\n"
            "    public string Name { get; set; } = string.Empty;\n"
            "    public int CategoryId { get; set; }\n"
            "    public Category? Category { get; set; }\n"
            "}\n\n"
            "public sealed class Category\n"
            "{\n"
            "    public int Id { get; set; }\n"
            "    public string Name { get; set; } = string.Empty;\n"
            "    public ICollection<Product> Products { get; set; } = new List<Product>();\n"
            "}\n\n"
            "public sealed class AppDbContext : DbContext\n"
            "{\n"
            "    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }\n"
            "    public DbSet<Product> Products => Set<Product>();\n"
            "    public DbSet<Category> Categories => Set<Category>();\n"
            "    protected override void OnModelCreating(ModelBuilder modelBuilder)\n"
            "    {\n"
            "        modelBuilder.Entity<Category>()\n"
            "            .HasMany(c => c.Products)\n"
            "            .WithOne(p => p.Category)\n"
            "            .HasForeignKey(p => p.CategoryId);\n"
            "    }\n"
            "}\n"
            "```"
        )
    if intent == "minimal-api":
        return (
            "```csharp\n"
            "var builder = WebApplication.CreateBuilder(args);\n"
            "var app = builder.Build();\n"
            "var todos = new[] { new { Id = 1, Title = \"Read docs\" } };\n"
            "app.MapGet(\"/todos\", () => Results.Ok(todos));\n"
            "app.Run();\n"
            "```"
        )
    if intent == "linq-filter":
        return (
            "```csharp\n"
            "using System.Linq;\n"
            "var values = new List<int> { 1, 5, 12, 20, 42 };\n"
            "var filtered = values.Where(v => v > 10).ToList();\n"
            "```"
        )
    return None


def default_template() -> str:
    return (
        "```csharp\n"
        "// Model is still warming up. Here is a safe C# starting template.\n"
        "public static class Example\n"
        "{\n"
        "    public static void Run()\n"
        "    {\n"
        "        Console.WriteLine(\"Ready\");\n"
        "    }\n"
        "}\n"
        "```"
    )


def policy_response(prompt: str) -> str:
    intent = detect_intent(prompt)
    return template_for_intent(intent) or default_template()
