import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel


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


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    max_start = tokens.shape[0] - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[s : s + seq_len] for s in starts]).to(device)
    y = torch.stack([tokens[s + 1 : s + seq_len + 1] for s in starts]).to(device)
    return x, y


def init_distributed():
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size == 1:
        return False, 0, 1, None

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device("cuda", local_rank)
    else:
        backend = "gloo"
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)
    return True, rank, world_size, device


def log(rank: int, message: str):
    if rank == 0:
        print(message, flush=True)


def evaluate(model: nn.Module, tokens: torch.Tensor, seq_len: int, batch_size: int, device: torch.device, steps: int = 10) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(steps):
            x, y = sample_batch(tokens, batch_size, seq_len, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def run(args):
    distributed, rank, world_size, distributed_device = init_distributed()

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = data_dir / "corpus.txt"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    tokenizer_name = args.tokenizer
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    text = corpus_path.read_text(encoding="utf-8", errors="ignore")
    extra_specs = [item.strip() for item in (args.extra_corpus or "").replace(";", ",").split(",") if item.strip()]
    for spec in extra_specs:
        extra_path = Path(spec)
        if not extra_path.is_absolute():
            extra_path = Path.cwd() / extra_path
        if not extra_path.exists():
            candidate = data_dir / Path(spec).name
            if candidate.exists():
                extra_path = candidate
        if not extra_path.exists():
            continue
        extra_text = extra_path.read_text(encoding="utf-8", errors="ignore")
        if extra_text.strip():
            text = text + ("\n\n" + extra_text) * max(1, args.extra_weight)
            log(rank, f"Loaded extra corpus from {extra_path} with weight {args.extra_weight}")
    raw_token_ids = tokenizer.encode(text)
    if len(raw_token_ids) < args.seq_len + 2:
        raise ValueError("Corpus is too small for current sequence length.")

    fallback_token_id = tokenizer.encode(" ")[0]
    max_vocab = max(2, args.vocab_size)
    most_common = Counter(raw_token_ids).most_common(max_vocab - 1)
    id_to_token = [fallback_token_id]
    for token_id, _ in most_common:
        if token_id != fallback_token_id:
            id_to_token.append(token_id)

    token_to_id = {token_id: idx for idx, token_id in enumerate(id_to_token)}
    compact_token_ids = [token_to_id.get(token_id, 0) for token_id in raw_token_ids]

    tokens = torch.tensor(compact_token_ids, dtype=torch.long)
    vocab_size = len(id_to_token)
    if args.cpu:
        device = torch.device("cpu")
    elif distributed_device is not None:
        device = distributed_device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyGpt(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)

    start_step = 0
    best_eval = float("inf")
    latest_checkpoint = artifacts_dir / "checkpoint_latest.pt"
    if args.resume and latest_checkpoint.exists():
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model_state = checkpoint["model_state"]
        if distributed:
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint.get("step", 0))
        best_eval = float(checkpoint.get("best_eval", best_eval))
        log(rank, f"Resumed from step {start_step}")

    model.train()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = sample_batch(tokens, args.micro_batch_size, args.seq_len, device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and args.amp):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_every == 0:
            reduced_loss = accumulated_loss
            if distributed:
                t = torch.tensor([accumulated_loss], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                reduced_loss = float(t.item())
            ppl = math.exp(min(reduced_loss * args.grad_accum_steps, 20))
            log(rank, f"step={step} loss={reduced_loss * args.grad_accum_steps:.4f} ppl={ppl:.2f}")

        if step % args.eval_every == 0:
            eval_loss = evaluate(model, tokens, args.seq_len, args.micro_batch_size, device, steps=args.eval_steps)
            if distributed:
                t = torch.tensor([eval_loss], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                eval_loss = float(t.item())
            if rank == 0:
                metrics = {
                    "step": step,
                    "eval_loss": eval_loss,
                    "eval_ppl": math.exp(min(eval_loss, 20)),
                    "world_size": world_size,
                    "tokenizer": tokenizer_name,
                }
                (artifacts_dir / "metrics_latest.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                log(rank, f"eval step={step} loss={eval_loss:.4f} ppl={metrics['eval_ppl']:.2f}")
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    best_path = artifacts_dir / "checkpoint_best.pt"
                    state_dict = model.module.state_dict() if distributed else model.state_dict()
                    torch.save(
                        {
                            "model_state": state_dict,
                            "tokenizer_name": tokenizer_name,
                            "id_to_token": id_to_token,
                            "config": {
                                "seq_len": args.seq_len,
                                "d_model": args.d_model,
                                "n_heads": args.n_heads,
                                "n_layers": args.n_layers,
                                "dropout": args.dropout,
                                "vocab_size": vocab_size,
                                "unknown_token_id": 0,
                            },
                        },
                        best_path)
                    log(rank, f"Saved best checkpoint to {best_path}")

        if step % args.save_every == 0 and rank == 0:
            state_dict = model.module.state_dict() if distributed else model.state_dict()
            torch.save(
                {
                    "model_state": state_dict,
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "best_eval": best_eval,
                    "tokenizer_name": tokenizer_name,
                    "id_to_token": id_to_token,
                    "config": {
                        "seq_len": args.seq_len,
                        "d_model": args.d_model,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "dropout": args.dropout,
                        "vocab_size": vocab_size,
                        "unknown_token_id": 0,
                    },
                },
                latest_checkpoint)
            log(rank, f"Saved latest checkpoint at step {step}")

    if rank == 0:
        state_dict = model.module.state_dict() if distributed else model.state_dict()
        final_checkpoint = {
            "model_state": state_dict,
            "tokenizer_name": tokenizer_name,
            "id_to_token": id_to_token,
            "config": {
                "seq_len": args.seq_len,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
                "vocab_size": vocab_size,
                "unknown_token_id": 0,
            },
        }
        torch.save(final_checkpoint, artifacts_dir / "checkpoint.pt")
        log(rank, f"Saved checkpoint to {artifacts_dir / 'checkpoint.pt'}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Train a local transformer for coding assistant generation.")
    parser.add_argument("--data-dir", default="ModelStack/data")
    parser.add_argument("--artifacts-dir", default="ModelStack/artifacts")
    parser.add_argument("--tokenizer", default="o200k_base")
    parser.add_argument("--extra-corpus", default="data/csharp_instruction_seed.txt")
    parser.add_argument("--extra-weight", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=12)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())
