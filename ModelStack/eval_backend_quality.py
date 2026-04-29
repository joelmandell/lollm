import argparse
import json
import re
from pathlib import Path

import requests
from policy_templates import policy_response


def iter_cases(path: Path, max_cases: int):
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            seen += 1
            if max_cases > 0 and seen >= max_cases:
                break


def score_case(output: str, case: dict) -> tuple[int, list[str]]:
    score = 100
    notes: list[str] = []
    lower = output.lower()

    for expected in case.get("expects", []):
        if expected.lower() not in lower:
            score -= 20
            notes.append(f"missing:{expected}")

    for forbidden in case.get("forbids", []):
        if forbidden.lower() in lower:
            score -= 25
            notes.append(f"forbidden:{forbidden}")

    if not output.strip():
        score -= 40
        notes.append("empty-output")

    if case.get("language") == "csharp":
        csharp_markers = ["public ", "class ", "using ", "namespace ", "Console.WriteLine", "var "]
        if not any(marker.lower() in lower for marker in csharp_markers):
            score -= 20
            notes.append("weak-csharp-signal")
        if re.search(r"\b(const|let|function|=>\s*\{)", output):
            score -= 15
            notes.append("js-syntax-detected")

    return max(0, score), notes


def main():
    parser = argparse.ArgumentParser(description="Evaluate coding quality from backend /generate endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--cases-path", default="ModelStack/evals/coding_quality_cases.jsonl")
    parser.add_argument("--out-path", default="ModelStack/artifacts/eval_latest.json")
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--mode", choices=["backend", "policy"], default="backend")
    args = parser.parse_args()

    results = []

    for case in iter_cases(Path(args.cases_path), args.max_cases):
        if args.mode == "policy":
            output = policy_response(case["prompt"])
        else:
            response = requests.post(
                f"{args.base_url}/generate",
                json={
                    "system_prompt": "You are a senior C# coding assistant. Return concise, correct C# code first.",
                    "user_prompt": case["prompt"],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                    "min_new_tokens": 32,
                },
                timeout=120,
            )
            response.raise_for_status()
            output = response.json().get("text", "")
        score, notes = score_case(output, case)
        results.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "score": score,
                "notes": notes,
                "output_preview": output[:400],
            }
        )

    average = round(sum(item["score"] for item in results) / max(1, len(results)))
    payload = {
        "average_score": average,
        "count": len(results),
        "results": [] if args.summary_only else results,
    }
    if results:
        payload["failed_cases"] = sum(1 for result in results if result["score"] < 100)
        payload["perfect_cases"] = sum(1 for result in results if result["score"] == 100)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
