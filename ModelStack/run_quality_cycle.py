import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests


def validate_training_profile(profile: dict) -> list[str]:
    errors: list[str] = []
    profile_name = profile.get("profile", "unknown")
    required_ints = [
        "vocab_size",
        "seq_len",
        "micro_batch_size",
        "grad_accum_steps",
        "steps",
        "d_model",
        "n_heads",
        "n_layers",
        "log_every",
        "eval_every",
        "eval_steps",
        "save_every",
    ]
    for key in required_ints:
        value = profile.get(key)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"[{profile_name}] '{key}' must be a positive integer.")

    d_model = profile.get("d_model")
    n_heads = profile.get("n_heads")
    if isinstance(d_model, int) and isinstance(n_heads, int) and d_model > 0 and n_heads > 0:
        if d_model % n_heads != 0:
            errors.append(
                f"[{profile_name}] d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

    dropout = profile.get("dropout")
    if not isinstance(dropout, (int, float)) or float(dropout) < 0 or float(dropout) >= 1:
        errors.append(f"[{profile_name}] 'dropout' must be in range [0, 1).")

    lr = profile.get("lr")
    if not isinstance(lr, (int, float)) or float(lr) <= 0:
        errors.append(f"[{profile_name}] 'lr' must be a positive number.")

    ratio = profile.get("priority_token_ratio")
    if ratio is not None and (not isinstance(ratio, (int, float)) or float(ratio) < 0 or float(ratio) > 1):
        errors.append(f"[{profile_name}] 'priority_token_ratio' must be in range [0, 1].")

    tokenizer = profile.get("tokenizer")
    if not isinstance(tokenizer, str) or not tokenizer.strip():
        errors.append(f"[{profile_name}] 'tokenizer' must be a non-empty string.")

    return errors


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


def build_weighted_corpus_spec(modelstack_dir: Path, feedback_corpus_arg: str | None, profile: dict) -> str:
    defaults = {
        "data/csharp_instruction_seed.txt": 8,
        "data/corpus_dotnet_focus.txt": 8,
        "data/prompt_interpretation_focus.txt": 8,
        "data/dotnet_csharp_sft_pairs.txt": 14,
        "data/github_csharp_quality_corpus.txt": 3,
        "data/feedback_corpus.txt": 12,
    }
    configured = profile.get("corpus_weights", {})
    weights = {**defaults, **configured}

    entries: list[str] = []
    seen: set[str] = set()

    def add_if_exists(relative_path: str):
        if relative_path in seen:
            return
        absolute = modelstack_dir / relative_path
        if absolute.exists():
            seen.add(relative_path)
            weight = max(1, int(weights.get(relative_path, profile.get("extra_weight", 4))))
            entries.append(f"{relative_path}:{weight}")

    add_if_exists(profile.get("extra_corpus", "data/csharp_instruction_seed.txt"))
    add_if_exists("data/corpus_dotnet_focus.txt")
    add_if_exists("data/prompt_interpretation_focus.txt")
    add_if_exists("data/dotnet_csharp_sft_pairs.txt")
    add_if_exists("data/github_csharp_quality_corpus.txt")
    if feedback_corpus_arg:
        add_if_exists(feedback_corpus_arg)

    return ",".join(entries)


def build_train_args(profile: dict, weighted_spec: str) -> list[str]:
    train_args = [
        "train_transformer.py",
        "--data-dir", "data",
        "--tokenizer", str(profile["tokenizer"]),
        "--extra-corpus-spec", weighted_spec,
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
        "--clean-corpus",
    ]
    if "high_signal_weight_threshold" in profile:
        train_args.extend(["--high-signal-weight-threshold", str(profile["high_signal_weight_threshold"])])
    if "priority_token_ratio" in profile:
        train_args.extend(["--priority-token-ratio", str(profile["priority_token_ratio"])])
    if profile.get("amp", False):
        train_args.append("--amp")
    if profile.get("resume", False):
        train_args.append("--resume")
    return train_args


def read_available_memory_gb() -> float | None:
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    kib = float(parts[1])
                    return kib / (1024 * 1024)
    except Exception:
        return None
    return None


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
    parser.add_argument("--fallback-profile", default="ModelStack/configs/stable_train_profile.json")
    parser.add_argument("--emergency-profile", default="ModelStack/configs/ultra_stable_train_profile.json")
    args = parser.parse_args()

    root = Path.cwd()
    modelstack_dir = root / args.modelstack_dir
    profile = json.loads((root / args.profile).read_text(encoding="utf-8"))
    fallback_profile_path = root / args.fallback_profile
    fallback_profile = json.loads(fallback_profile_path.read_text(encoding="utf-8")) if fallback_profile_path.exists() else None
    emergency_profile_path = root / args.emergency_profile
    emergency_profile = json.loads(emergency_profile_path.read_text(encoding="utf-8")) if emergency_profile_path.exists() else None

    available_memory_gb = read_available_memory_gb()
    min_required_memory_gb = profile.get("min_available_memory_gb")
    if (
        fallback_profile is not None
        and available_memory_gb is not None
        and isinstance(min_required_memory_gb, (int, float))
        and available_memory_gb < float(min_required_memory_gb)
    ):
        print(
            f"Available memory {available_memory_gb:.2f}GB is below profile minimum "
            f"{float(min_required_memory_gb):.2f}GB. Using fallback profile.")
        profile = fallback_profile

    if not args.skip_export:
        export_corpus(args.api_base)

    if args.fetch_github_quality:
        fetch_cmd = [sys.executable, "fetch_csharp_quality_corpus.py", "--max-repos", "6", "--max-files-per-repo", "1200"]
        run_command(fetch_cmd, modelstack_dir)

    weighted_spec = build_weighted_corpus_spec(modelstack_dir, args.feedback_corpus, profile)
    profiles_to_try: list[dict] = [profile]
    for candidate in [fallback_profile, emergency_profile]:
        if candidate is None:
            continue
        if all(existing.get("profile") != candidate.get("profile") for existing in profiles_to_try):
            profiles_to_try.append(candidate)

    completed = False
    skipped_profiles: list[str] = []
    for index, active_profile in enumerate(profiles_to_try):
        validation_errors = validate_training_profile(active_profile)
        if validation_errors:
            profile_name = active_profile.get("profile", f"profile-{index}")
            skipped_profiles.append(profile_name)
            for message in validation_errors:
                print(f"Skipping invalid profile: {message}")
            continue

        train_args = build_train_args(active_profile, weighted_spec)
        if args.torchrun and index == 0:
            cmd = ["torchrun", "--nproc_per_node", str(args.nproc_per_node)] + train_args
        else:
            cmd = [sys.executable] + train_args
        try:
            run_command(cmd, modelstack_dir)
            completed = True
            break
        except subprocess.CalledProcessError as exc:
            if exc.returncode not in (-9, 137) or index == len(profiles_to_try) - 1:
                raise
            next_profile = profiles_to_try[index + 1].get("profile", "fallback")
            print(
                f"Training profile '{active_profile.get('profile', 'unknown')}' terminated "
                f"(likely OOM). Retrying with '{next_profile}'.")

    if not completed:
        if len(skipped_profiles) == len(profiles_to_try):
            raise RuntimeError(
                "No valid training profile available after validation. "
                f"Skipped profiles: {', '.join(skipped_profiles)}.")
        raise RuntimeError("Training did not complete with any configured profile.")

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
