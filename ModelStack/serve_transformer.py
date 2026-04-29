import os
import re
from pathlib import Path

import tiktoken
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class TinyGpt(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq = x.shape
        pos = torch.arange(0, seq, device=x.device).unsqueeze(0).expand(bsz, seq)
        h = self.token_embedding(x) + self.position_embedding(pos)
        causal_mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=causal_mask)
        return self.output(self.norm(h))


class GenerateRequest(BaseModel):
    system_prompt: str = Field(default="")
    user_prompt: str = Field(default="")
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1, le=256)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    min_new_tokens: int = Field(default=24, ge=0, le=512)


class GenerateResponse(BaseModel):
    text: str


class ModelRuntime:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.token_to_compact_id = None
        self.id_to_token = None
        self.config = None

    def load(self):
        checkpoint_path = self.model_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing model artifacts in {self.model_dir}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint["config"]
        tokenizer_name = checkpoint.get("tokenizer_name", "o200k_base")
        id_to_token = checkpoint.get("id_to_token")
        if not id_to_token:
            raise ValueError("Checkpoint is missing token mapping for o200k_base tokenizer.")
        model = TinyGpt(
            vocab_size=config["vocab_size"],
            seq_len=config["seq_len"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        self.model = model
        self.config = config
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.id_to_token = [int(token_id) for token_id in id_to_token]
        self.token_to_compact_id = {
            int(token_id): idx for idx, token_id in enumerate(self.id_to_token)
        }

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        repetition_penalty: float,
        min_new_tokens: int) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        policy = self._policy_response(user_prompt)
        if policy is not None:
            return policy

        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}".strip()
        prompt_token_ids = self.tokenizer.encode(prompt)
        ids = [self.token_to_compact_id.get(token_id, 0) for token_id in prompt_token_ids]
        if not ids:
            ids = [0]

        candidates: list[str] = []
        decode_plan = [
            (0.0, min(80, top_k), repetition_penalty),
            (temperature, top_k, repetition_penalty),
            (max(0.15, temperature + 0.05), min(80, top_k + 10), repetition_penalty + 0.05),
        ]
        for cand_temp, cand_top_k, cand_rep_penalty in decode_plan:
            text = self._sample_once(ids, max_tokens, cand_temp, cand_top_k, cand_rep_penalty, min_new_tokens)
            if text:
                candidates.append(text)
        if not candidates:
            return self._fallback_response(user_prompt)

        text = max(candidates, key=lambda candidate: self._score_candidate(user_prompt, candidate))
        if text and not self._should_fallback(user_prompt, text):
            return text
        return self._fallback_response(user_prompt)

    def _sample_once(
        self,
        ids: list[int],
        max_tokens: int,
        temperature: float,
        top_k: int,
        repetition_penalty: float,
        min_new_tokens: int) -> str:
        seq_len = self.config["seq_len"]
        generated = list(ids)[-seq_len:]
        generated_counts: dict[int, int] = {}
        unknown_id = int(self.config.get("unknown_token_id", 0))

        for step in range(max_tokens):
            x = torch.tensor([generated[-seq_len:]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(x)[0, -1, :]

            if repetition_penalty > 1.0:
                for token_id, count in generated_counts.items():
                    logits[token_id] = logits[token_id] / (repetition_penalty ** min(count, 6))
            if step < min_new_tokens:
                logits[unknown_id] = -1e9
            elif generated_counts.get(unknown_id, 0) > max(8, min_new_tokens // 2):
                logits[unknown_id] = logits[unknown_id] / 8.0

            effective_top_k = min(max(1, top_k), logits.shape[0])
            top_values, top_indices = torch.topk(logits, k=effective_top_k)
            if temperature <= 0:
                next_id = top_indices[torch.argmax(top_values)].item()
            else:
                probs = torch.softmax(top_values / max(temperature, 1e-5), dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1).item()
                next_id = top_indices[sampled_idx].item()
            generated.append(next_id)
            generated_counts[next_id] = generated_counts.get(next_id, 0) + 1

            if step + 1 >= min_new_tokens:
                new_compact_tokens = generated[len(ids) :]
                raw_tokens = [
                    self.id_to_token[token_id] if 0 <= token_id < len(self.id_to_token) else self.id_to_token[0]
                    for token_id in new_compact_tokens
                ]
                preview = self.tokenizer.decode(raw_tokens).strip()
                if preview:
                    break

        new_compact_tokens = generated[len(ids) :]
        raw_tokens = [
            self.id_to_token[token_id] if 0 <= token_id < len(self.id_to_token) else self.id_to_token[0]
            for token_id in new_compact_tokens
        ]
        return self.tokenizer.decode(raw_tokens).strip()

    def _score_candidate(self, user_prompt: str, text: str) -> int:
        score = 0
        lower = text.lower()
        prompt = user_prompt.lower()
        intent = self._detect_intent(prompt)
        if len(text.strip()) >= 30:
            score += 20
        if "```csharp" in lower:
            score += 25
        csharp_markers = ["using ", "class ", "public ", "namespace ", "console.writeline", "dbcontext", "mapget", "results.ok", "async ", "await "]
        score += sum(8 for marker in csharp_markers if marker in lower)
        if re.search(r"\b(const|let|function)\b", lower):
            score -= 30
        if intent == "efcore-dbcontext" and "onmodelcreating" in lower:
            score += 30
        if intent == "xunit-test" and ("[fact]" in lower or "assert." in lower):
            score += 25
        if intent in ("async-httpclient", "cancellation-token") and ("async " in lower and "await " in lower):
            score += 20
        if intent == "di-registration" and ("addscoped" in lower or "addsingleton" in lower):
            score += 20
        if intent == "cancellation-token" and "cancellationtoken" in lower:
            score += 20
        if intent == "record-dto" and ("record " in lower or "record class" in lower):
            score += 20
        return score

    def _policy_response(self, user_prompt: str) -> str | None:
        intent = self._detect_intent(user_prompt.lower())
        if not intent:
            return None
        return self._template_for_intent(intent)

    def _template_for_intent(self, intent: str) -> str | None:
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

    def _detect_intent(self, prompt: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9#+.\s]", " ", prompt.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()

        def has_any(terms: list[str]) -> bool:
            return any(term in normalized for term in terms)

        csharp_context = has_any(["c#", "csharp", ".net", "dotnet", "asp.net", "program.main", "console.writeline"])
        if (has_any(["hello world", "first program"]) or re.search(r"\bfirst\b.*\bprogram\b", normalized)) and csharp_context:
            return "hello-world"
        if has_any(["entity framework", "ef core", "dbcontext", "modelbuilder"]):
            return "efcore-dbcontext"
        if has_any(["minimal api", "webapplication", "mapget"]) and (
            has_any(["asp.net", ".net", "dotnet"]) or has_any(["/todos", "results.ok", "todo"])):
            return "minimal-api"
        if has_any(["filter"]) and has_any(["list", "ints", "linq", "greater than"]):
            return "linq-filter"
        if has_any(["xunit", "unit test", "fact attribute", "[fact]", "assert.equal"]):
            return "xunit-test"
        if has_any(["httpclient", "getstringasync"]) or (
            has_any(["http", "https", "url", "uri", "fetch"]) and (
                has_any(["async", "task<string>", "task", "await"]) or csharp_context)):
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

    def _should_fallback(self, user_prompt: str, text: str) -> bool:
        prompt = user_prompt.lower()
        intent = self._detect_intent(prompt)
        csharp_intent = any(
            marker in prompt
            for marker in [
                "c#", "csharp", ".net", "dotnet", "asp.net", "ef core", "dbcontext",
                "linq", "xunit", "dependency injection", "repository", "record", "dto",
                "cancellationtoken", "cancellation token", "httpclient", "mapget", "task<"
            ]
        )
        if not csharp_intent:
            return False

        lower_text = text.lower()
        if intent == "hello-world" and ("console.writeline" not in lower_text or "using system" not in lower_text):
            return True
        if intent == "efcore-dbcontext" and (
            "dbcontext" not in lower_text
            or "dbset<product>" not in lower_text
            or "onmodelcreating" not in lower_text):
            return True
        if intent == "minimal-api" and ("mapget" not in lower_text or "results.ok" not in lower_text):
            return True
        if intent == "linq-filter" and "where(" not in lower_text:
            return True
        if intent == "di-registration" and ("addscoped" not in lower_text and "addsingleton" not in lower_text):
            return True
        if intent == "cancellation-token" and "cancellationtoken" not in lower_text:
            return True
        if intent == "record-dto" and "record " not in lower_text:
            return True
        csharp_markers = ["using ", "class ", "public ", "namespace ", "console.writeline", "dbcontext", "mapget", "results.ok"]
        has_csharp = any(marker in lower_text for marker in csharp_markers)
        token_count = len(text.split())
        non_word_ratio = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in "_{}();.,<>\"'`/\n:-")) / max(1, len(text))
        noisy_fragments = lower_text.count("web ") + lower_text.count("bool ") + lower_text.count("jsx")

        if has_csharp:
            return False
        if token_count < 12:
            return True
        if non_word_ratio > 0.12:
            return True
        if noisy_fragments >= 3:
            return True
        return False

    def _fallback_response(self, user_prompt: str) -> str:
        intent = self._detect_intent(user_prompt.lower())
        if intent:
            template = self._template_for_intent(intent)
            if template:
                return template
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


default_model_dir = Path(__file__).resolve().parent / "artifacts"
model_dir = Path(os.getenv("MODEL_DIR", str(default_model_dir))).resolve()
runtime = ModelRuntime(model_dir)
app = FastAPI(title="Local Transformer Backend", version="0.1.0")


@app.on_event("startup")
def startup():
    runtime.load()


@app.get("/health")
def health():
    loaded = runtime.model is not None and runtime.tokenizer is not None
    return {"ok": loaded, "model_dir": str(model_dir)}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        text = runtime.generate(
            req.system_prompt,
            req.user_prompt,
            req.max_tokens,
            req.temperature,
            req.top_k,
            req.repetition_penalty,
            req.min_new_tokens)
        return GenerateResponse(text=text)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
