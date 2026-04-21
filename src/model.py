"""
GPT-style causal transformer for music generation.

Text prompt tokens are prepended as prefix; model is trained with
next-token prediction over the combined text+music sequence.
"""

from __future__ import annotations
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from src.tokenizer import VOCAB_SIZE

CHECKPOINT_DIR = Path("checkpoints")
CONFIG_FILE    = CHECKPOINT_DIR / "config.json"
WEIGHTS_FILE   = CHECKPOINT_DIR / "model.pt"

DEFAULT_CONFIG = dict(
    vocab_size   = VOCAB_SIZE,
    d_model      = 256,
    n_heads      = 8,
    n_layers     = 4,
    max_seq_len  = 2048,
    dropout      = 0.1,
)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        scale = math.sqrt(self.head_dim)
        attn  = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True,
        )
        out = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MusicTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg     = cfg
        d            = cfg["d_model"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        self.pos_emb = nn.Embedding(cfg["max_seq_len"], d)
        self.drop    = nn.Dropout(cfg["dropout"])
        self.blocks  = nn.ModuleList(
            [Block(d, cfg["n_heads"], cfg["dropout"]) for _ in range(cfg["n_layers"])]
        )
        self.ln_f    = nn.LayerNorm(d)
        self.head    = nn.Linear(d, cfg["vocab_size"], bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: list[int],
        max_new: int = 400,
        temperature: float = 1.0,
        top_k: int = 50,
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
    ) -> list[int]:
        from src.tokenizer import EOS
        self.eval()
        idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        max_ctx = self.cfg["max_seq_len"]
        for _ in range(max_new):
            idx_cond = idx[:, -max_ctx:]
            logits   = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs  = torch.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)
            idx    = torch.cat([idx, next_t], dim=1)
            if next_t.item() == EOS:
                break
        return idx[0].tolist()


def build_model(cfg: dict | None = None) -> MusicTransformer:
    cfg = cfg or DEFAULT_CONFIG
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    return MusicTransformer(cfg)


def load_model(device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu") -> tuple[MusicTransformer, dict]:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError("No model config found. Run 'Build Model' first.")
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    model = MusicTransformer(cfg).to(device)
    if WEIGHTS_FILE.exists():
        state = torch.load(WEIGHTS_FILE, map_location=device, weights_only=False)
        model.load_state_dict(state)
    return model, cfg


def model_status() -> str:
    if not CONFIG_FILE.exists():
        return "Not built"
    info = f"Config: {CONFIG_FILE}"
    if WEIGHTS_FILE.exists():
        size = WEIGHTS_FILE.stat().st_size // 1024
        info += f" | Weights: {size} KB"
    else:
        info += " | Weights: none (untrained)"
    return info
