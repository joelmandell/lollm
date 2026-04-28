import os
from pathlib import Path

import sentencepiece as spm
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


class GenerateResponse(BaseModel):
    text: str


class ModelRuntime:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = None

    def load(self):
        checkpoint_path = self.model_dir / "checkpoint.pt"
        tokenizer_path = self.model_dir / "tokenizer.model"
        if not checkpoint_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(f"Missing model artifacts in {self.model_dir}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint["config"]
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
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}".strip()
        ids = self.tokenizer.encode(prompt)
        if not ids:
            ids = [self.tokenizer.bos_id()]

        seq_len = self.config["seq_len"]
        generated = list(ids)[-seq_len:]

        for _ in range(max_tokens):
            x = torch.tensor([generated[-seq_len:]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(x)[0, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits).item()
            else:
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_id)
            if next_id == self.tokenizer.eos_id():
                break

        new_tokens = generated[len(ids) :]
        return self.tokenizer.decode(new_tokens).strip()


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
        text = runtime.generate(req.system_prompt, req.user_prompt, req.max_tokens, req.temperature)
        return GenerateResponse(text=text)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
