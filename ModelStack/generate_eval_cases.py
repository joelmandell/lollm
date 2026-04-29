import argparse
import itertools
import json
from pathlib import Path


def make_cases() -> list[dict]:
    cases: list[dict] = []

    def add_intent_cases(
        intent_id: str,
        openings: list[str],
        subjects: list[str],
        constraints: list[str],
        expects: list[str],
        forbids: list[str]):
        combinations = itertools.product(openings, subjects, constraints)
        for index, (opening, subject, constraint) in enumerate(combinations, start=1):
            prompt = f"{opening} {subject} {constraint}".strip()
            prompt = " ".join(prompt.split())
            cases.append(
                {
                    "id": f"{intent_id}-{index:04d}",
                    "prompt": prompt,
                    "expects": expects,
                    "forbids": forbids,
                    "language": "csharp",
                }
            )

    add_intent_cases(
        "hello-world",
        openings=["Write", "Create", "Show", "Generate", "Provide"],
        subjects=[
            "a C# hello world example",
            "the first C# program",
            "a C# entry point that prints Hello World",
            "a dotnet hello world program",
        ],
        constraints=[
            "with Program.Main.",
            "using Console.WriteLine.",
            "and keep it minimal.",
            "targeting .NET.",
        ],
        expects=["Console.WriteLine", "Main"],
        forbids=["console.log", "function "],
    )

    add_intent_cases(
        "efcore-dbcontext",
        openings=["Create", "Implement", "Show", "Generate", "Write"],
        subjects=[
            "an EF Core DbContext for Product and Category",
            "an Entity Framework Core context with Product/Category",
            "a DbContext with Product and Category entities",
            "an EF Core model for Product + Category",
        ],
        constraints=[
            "including OnModelCreating relationship config.",
            "with DbSet<Product> and DbSet<Category>.",
            "with one-to-many relationship mapping.",
            "ready for .NET app usage.",
        ],
        expects=["DbContext", "DbSet<Product>", "OnModelCreating"],
        forbids=["sequelize", "TypeORM", "mongoose"],
    )

    add_intent_cases(
        "minimal-api",
        openings=["Create", "Write", "Show", "Implement", "Provide"],
        subjects=[
            "an ASP.NET minimal API for GET /todos",
            "a .NET minimal API endpoint that returns todos",
            "a WebApplication with MapGet /todos",
            "a minimal API in ASP.NET returning todo data",
        ],
        constraints=[
            "return Results.Ok.",
            "using WebApplication builder.",
            "with a small in-memory list.",
            "as concise C# sample.",
        ],
        expects=["MapGet", "Results.Ok", "WebApplication"],
        forbids=["express()", "app.get("],
    )

    add_intent_cases(
        "linq-filter",
        openings=["Write", "Show", "Implement", "Give", "Create"],
        subjects=[
            "a C# LINQ filter over a list of ints",
            "C# code to filter integer list values",
            "a list filtering example in C#",
            "LINQ filtering sample for integers",
        ],
        constraints=[
            "keeping values greater than 10.",
            "using Where and ToList.",
            "with concise C# syntax.",
            "for a basic demo.",
        ],
        expects=["Where(", "ToList", "List<int>"],
        forbids=["Array.prototype.filter", "lodash", "reduce("],
    )

    add_intent_cases(
        "xunit-test",
        openings=["Write", "Create", "Show", "Generate", "Provide"],
        subjects=[
            "an xUnit test in C# for 2 + 3 = 5",
            "a C# unit test using xUnit",
            "a fact-based xUnit sample test",
            "xUnit assertion example in C#",
        ],
        constraints=[
            "using [Fact].",
            "with Assert.Equal.",
            "as clean minimal test.",
            "for .NET testing.",
        ],
        expects=["[Fact]", "Assert.Equal", "using Xunit"],
        forbids=["jest", "describe(", "it("],
    )

    add_intent_cases(
        "async-httpclient",
        openings=["Write", "Show", "Implement", "Create", "Generate"],
        subjects=[
            "an async C# HttpClient method fetching URL content",
            "a C# async method using HttpClient.GetStringAsync",
            "a method that requests HTTP content in C#",
            "an async .NET helper for URL fetch",
        ],
        constraints=[
            "returning Task<string>.",
            "using await.",
            "with HttpClient parameter injection.",
            "as production-style sample.",
        ],
        expects=["async Task<string>", "HttpClient", "await"],
        forbids=["fetch(", "axios", "Promise"],
    )

    add_intent_cases(
        "linq-groupby",
        openings=["Write", "Show", "Implement", "Provide", "Create"],
        subjects=[
            "a C# LINQ GroupBy for order totals per customer",
            "a group by query in LINQ for amount totals",
            "an orders grouped-by-customer projection",
            "LINQ aggregation grouped by customer",
        ],
        constraints=[
            "with Select projection.",
            "using Sum and ToList.",
            "for a concise report result.",
            "as clear C# code.",
        ],
        expects=["GroupBy", "Select", "ToList"],
        forbids=["reduce(", "lodash", "map("],
    )

    add_intent_cases(
        "di-registration",
        openings=["Write", "Show", "Implement", "Provide", "Create"],
        subjects=[
            "dependency injection registration using IServiceCollection",
            ".NET DI setup with scoped and singleton services",
            "a ServiceCollection registration example",
            "DI container configuration in C#",
        ],
        constraints=[
            "include AddScoped.",
            "and call BuildServiceProvider.",
            "with concise setup code.",
            "for backend service wiring.",
        ],
        expects=["ServiceCollection", "AddScoped", "BuildServiceProvider"],
        forbids=["container.bind", "inversify", "angular.module"],
    )

    add_intent_cases(
        "cancellation-token",
        openings=["Write", "Show", "Implement", "Create", "Generate"],
        subjects=[
            "an async C# method with CancellationToken",
            "a method forwarding cancellation token to repository calls",
            "a cancellable async service method in .NET",
            "an awaitable method handling CancellationToken",
        ],
        constraints=[
            "using await.",
            "with CancellationToken parameter.",
            "passing token to downstream async call.",
            "as clean C# sample.",
        ],
        expects=["async Task", "CancellationToken", "await"],
        forbids=["Promise", "then(", "abortcontroller"],
    )

    add_intent_cases(
        "record-dto",
        openings=["Write", "Create", "Show", "Generate", "Provide"],
        subjects=[
            "C# record DTOs for order create request/response",
            "immutable request and response records in C#",
            "record models for API request/response",
            "C# record-based DTO definitions",
        ],
        constraints=[
            "include CreateOrderRequest and CreateOrderResponse.",
            "with concise immutable types.",
            "as clean API contract sample.",
            "for .NET app models.",
        ],
        expects=["record", "CreateOrderRequest", "CreateOrderResponse"],
        forbids=["type CreateOrderRequest =", "interface CreateOrderRequest", "export type"],
    )

    add_intent_cases(
        "repository-pattern",
        openings=["Write", "Implement", "Show", "Create", "Provide"],
        subjects=[
            "a C# repository interface and implementation for Product",
            "repository pattern with async methods in .NET",
            "IProductRepository and ProductRepository sample",
            "repository abstraction plus EF implementation",
        ],
        constraints=[
            "use Task-based methods.",
            "with interface and concrete class.",
            "including async list retrieval.",
            "as production-style pattern.",
        ],
        expects=["interface IProductRepository", "class ProductRepository", "Task<"],
        forbids=["export class", "function ProductRepository", "prototype"],
    )

    return cases


def main():
    parser = argparse.ArgumentParser(description="Generate large C# eval corpus for backend quality testing.")
    parser.add_argument("--out-path", default="ModelStack/evals/coding_quality_cases_1000.jsonl")
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args()

    all_cases = make_cases()
    if not all_cases:
        raise RuntimeError("No base cases generated.")

    style_suffixes = [
        " Keep it concise and idiomatic.",
        " Prefer modern C# style.",
        " Return code first, then brief explanation.",
        " Avoid JavaScript syntax.",
        " Keep naming clear and production-ready.",
        " Use .NET conventions.",
        " Provide a minimal but complete snippet.",
        " Favor readability over cleverness.",
    ]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        total_base = len(all_cases)
        for idx in range(args.target_count):
            base = all_cases[idx % total_base]
            cycle = idx // total_base
            suffix = style_suffixes[cycle % len(style_suffixes)]
            case = {
                "id": f"{base['id']}-r{idx + 1:08d}",
                "prompt": f"{base['prompt']}{suffix}",
                "expects": base["expects"],
                "forbids": base["forbids"],
                "language": base["language"],
            }
            handle.write(json.dumps(case, ensure_ascii=True))
            handle.write("\n")
            if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
                print(f"generated={idx + 1}", flush=True)

    print(f"Generated {args.target_count} cases at {out_path}")


if __name__ == "__main__":
    main()
