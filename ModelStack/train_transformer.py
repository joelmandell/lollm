import argparse
import math
import random
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def train_tokenizer(corpus_path: Path, tokenizer_prefix: Path, vocab_size: int) -> Path:
    model_path = tokenizer_prefix.with_suffix(".model")
    if model_path.exists():
        return model_path
    tokenizer_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(tokenizer_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
    )
    return model_path


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_len: int, device: torch.device):
    max_start = tokens.shape[0] - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[s : s + seq_len] for s in starts]).to(device)
    y = torch.stack([tokens[s + 1 : s + seq_len + 1] for s in starts]).to(device)
    return x, y


def run(args):
    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = data_dir / "corpus.txt"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    tokenizer_prefix = artifacts_dir / "tokenizer"
    tokenizer_model = train_tokenizer(corpus_path, tokenizer_prefix, args.vocab_size)

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    text = corpus_path.read_text(encoding="utf-8", errors="ignore")
    token_ids = tokenizer.encode(text)
    if len(token_ids) < args.seq_len + 2:
        raise ValueError("Corpus is too small for current sequence length.")

    tokens = torch.tensor(token_ids, dtype=torch.long)
    vocab_size = tokenizer.vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = TinyGpt(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = sample_batch(tokens, args.batch_size, args.seq_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            ppl = math.exp(min(loss.item(), 20))
            print(f"step={step} loss={loss.item():.4f} ppl={ppl:.2f}")

    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "vocab_size": vocab_size,
        },
    }
    torch.save(checkpoint, artifacts_dir / "checkpoint.pt")
    print(f"Saved checkpoint to {artifacts_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a local tiny transformer for coding assistant generation.")
    parser.add_argument("--data-dir", default="ModelStack/data")
    parser.add_argument("--artifacts-dir", default="ModelStack/artifacts")
    parser.add_argument("--vocab-size", type=int, default=24000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    run(parser.parse_args())
