import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import requests


DEFAULT_REPOS = [
    "dotnet/runtime",
    "dotnet/aspnetcore",
    "dotnet/roslyn",
    "dotnet/efcore",
    "dotnet/orleans",
    "dapr/dotnet-sdk",
    "MassTransit/MassTransit",
    "FluentValidation/FluentValidation",
    "AutoMapper/AutoMapper",
    "xunit/xunit",
]

ALLOWED_EXTENSIONS = {".cs", ".csproj", ".sln", ".props", ".targets", ".editorconfig"}


def api_get(url: str, token: str | None):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "lollm-corpus-fetcher"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()


def get_default_branch(owner: str, repo: str, token: str | None) -> str:
    data = api_get(f"https://api.github.com/repos/{owner}/{repo}", token)
    return data.get("default_branch", "main")


def download_repo_zip(owner: str, repo: str, branch: str, token: str | None) -> bytes:
    headers = {"User-Agent": "lollm-corpus-fetcher"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    response = requests.get(url, headers=headers, timeout=300)
    response.raise_for_status()
    return response.content


def extract_repo_corpus(
    zip_bytes: bytes,
    max_files: int,
    max_chars_per_file: int,
    include_readme: bool):
    texts = []
    files = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.suffix.lower() in ALLOWED_EXTENSIONS or (include_readme and path.name.lower().startswith("readme")):
                try:
                    with archive.open(name) as handle:
                        content = handle.read().decode("utf-8", errors="ignore")
                except Exception:
                    continue
                if not content.strip():
                    continue
                content = content[:max_chars_per_file]
                texts.append((name, content))
                files.append(name)
                if len(texts) >= max_files:
                    break
    return texts, files


def main():
    parser = argparse.ArgumentParser(description="Fetch high-quality C# GitHub repositories into training corpus.")
    parser.add_argument("--out-path", default="ModelStack/data/github_csharp_quality_corpus.txt")
    parser.add_argument("--manifest-path", default="ModelStack/data/github_csharp_quality_manifest.json")
    parser.add_argument("--repos", default=",".join(DEFAULT_REPOS))
    parser.add_argument("--max-repos", type=int, default=6)
    parser.add_argument("--max-files-per-repo", type=int, default=1200)
    parser.add_argument("--max-chars-per-file", type=int, default=18000)
    parser.add_argument("--include-readme", action="store_true")
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN")
    repo_specs = [item.strip() for item in args.repos.split(",") if item.strip()][: max(1, args.max_repos)]

    out_path = Path(args.out_path)
    manifest_path = Path(args.manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "repos": [],
        "total_files": 0,
        "total_chars": 0,
    }

    with out_path.open("w", encoding="utf-8") as output:
        for repo_spec in repo_specs:
            if "/" not in repo_spec:
                continue
            owner, repo = repo_spec.split("/", 1)
            branch = get_default_branch(owner, repo, token)
            zip_bytes = download_repo_zip(owner, repo, branch, token)
            texts, files = extract_repo_corpus(
                zip_bytes,
                max_files=args.max_files_per_repo,
                max_chars_per_file=args.max_chars_per_file,
                include_readme=args.include_readme)

            total_chars = 0
            for file_name, content in texts:
                output.write(f"## Repo: {owner}/{repo}\n")
                output.write(f"### File: {file_name}\n")
                output.write(content)
                output.write("\n\n")
                total_chars += len(content)

            manifest["repos"].append(
                {
                    "repo": f"{owner}/{repo}",
                    "branch": branch,
                    "file_count": len(files),
                    "char_count": total_chars,
                }
            )
            manifest["total_files"] += len(files)
            manifest["total_chars"] += total_chars
            print(f"Fetched {owner}/{repo}: files={len(files)} chars={total_chars}", flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote corpus to {out_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
