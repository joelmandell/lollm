import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests


def run_command(command: list[str], cwd: Path):
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd), check=True)


def export_corpus(api_base: str):
    response = requests.post(
        f"{api_base}/api/model/export-corpus",
        json={"includeJsonl": True, "includeText": True},
        timeout=600,
    )
    response.raise_for_status()
    print("Export corpus:", response.json())


def main():
    parser = argparse.ArgumentParser(description="Run iterative train+eval cycle for model quality.")
    parser.add_argument("--api-base", default="http://localhost:5101")
    parser.add_argument("--profile", default="ModelStack/configs/scaled_train_profile.json")
    parser.add_argument("--modelstack-dir", default="ModelStack")
    parser.add_argument("--torchrun", action="store_true")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--fetch-github-quality", action="store_true")
    parser.add_argument("--feedback-corpus", default=None, help="Optional extra corpus path relative to modelstack-dir.")
    args = parser.parse_args()

    root = Path.cwd()
    modelstack_dir = root / args.modelstack_dir
    profile = json.loads((root / args.profile).read_text(encoding="utf-8"))

    if not args.skip_export:
        export_corpus(args.api_base)

    if args.fetch_github_quality:
        fetch_cmd = [sys.executable, "fetch_csharp_quality_corpus.py", "--max-repos", "6", "--max-files-per-repo", "1200"]
        run_command(fetch_cmd, modelstack_dir)

    extra_corpora = [profile.get("extra_corpus", "data/csharp_instruction_seed.txt")]
    github_corpus = modelstack_dir / "data/github_csharp_quality_corpus.txt"
    if github_corpus.exists():
        extra_corpora.append("data/github_csharp_quality_corpus.txt")
    if args.feedback_corpus:
        feedback_corpus = modelstack_dir / args.feedback_corpus
        if feedback_corpus.exists():
            extra_corpora.append(args.feedback_corpus)

    train_args = [
        "train_transformer.py",
        "--data-dir", "data",
        "--tokenizer", str(profile["tokenizer"]),
        "--extra-corpus", ",".join(extra_corpora),
        "--extra-weight", str(profile.get("extra_weight", 4)),
        "--vocab-size", str(profile["vocab_size"]),
        "--seq-len", str(profile["seq_len"]),
        "--micro-batch-size", str(profile["micro_batch_size"]),
        "--grad-accum-steps", str(profile["grad_accum_steps"]),
        "--steps", str(profile["steps"]),
        "--d-model", str(profile["d_model"]),
        "--n-heads", str(profile["n_heads"]),
        "--n-layers", str(profile["n_layers"]),
        "--dropout", str(profile["dropout"]),
        "--lr", str(profile["lr"]),
        "--log-every", str(profile["log_every"]),
        "--eval-every", str(profile["eval_every"]),
        "--eval-steps", str(profile["eval_steps"]),
        "--save-every", str(profile["save_every"]),
    ]
    if profile.get("amp", False):
        train_args.append("--amp")
    if profile.get("resume", False):
        train_args.append("--resume")

    if args.torchrun:
        cmd = ["torchrun", "--nproc_per_node", str(args.nproc_per_node)] + train_args
    else:
        cmd = [sys.executable] + train_args
    run_command(cmd, modelstack_dir)

    if not args.skip_eval:
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "serve_transformer:app", "--host", "127.0.0.1", "--port", "8010"],
            cwd=str(modelstack_dir),
        )
        try:
            for _ in range(40):
                try:
                    health = requests.get("http://127.0.0.1:8010/health", timeout=2)
                    if health.ok:
                        break
                except Exception:
                    pass
                time.sleep(1)

            eval_cmd = [sys.executable, "ModelStack/eval_backend_quality.py", "--base-url", "http://127.0.0.1:8010"]
            run_command(eval_cmd, root)
        finally:
            server.terminate()
            server.wait(timeout=15)


if __name__ == "__main__":
    main()
