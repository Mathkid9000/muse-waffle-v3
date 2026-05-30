"""
GPT-style causal transformer for music generation.

Text prompt tokens are prepended as prefix; model is trained with
next-token prediction over the combined text+music sequence.
"""

from __future__ import annotations
import math
import json
from pathlib import Path

from dotenv import load_dotenv
import os
import torch
import torch.nn as nn
from torch import Tensor

from src.tokenizer import VOCAB_SIZE

load_dotenv()
CHECKPOINT_DIR = Path(os.getenv('STORAGE_DIR')) / "checkpoints"
CONFIG_FILE = CHECKPOINT_DIR / "config.json"
WEIGHTS_FILE = CHECKPOINT_DIR / "model.pt"

DEFAULT_CONFIG = dict(
    vocab_size = VOCAB_SIZE,
    d_model = 512,
    n_heads = 8,
    n_layers = 6,
    max_seq_len = 2048,
    dropout = 0.1,
    epoches = []
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

    @torch.no_grad()
    def generate_infill(
        self,
        prefix_tokens: list[int],
        suffix_tokens: list[int],
        header_tokens: list[int] | None = None,
        target_bars: int = 4,
        max_new: int = 400,
        temperature: float = 1.0,
        top_k: int = 50,
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
    ) -> list[int]:
        """Generate tokens to fill a gap using FIM (fill-in-the-middle) approach.

        Constructs: [BOS] + header + prefix + [FILL_START] + suffix + [FILL_END]
        Then generates autoregressively until EOS or target bar count is reached.

        Args:
            prefix_tokens: Tokens before the gap (without BOS)
            suffix_tokens: Tokens after the gap (without EOS)
            header_tokens: Instrument header tokens for the score (optional)
            target_bars: Number of BAR tokens to generate (matches gap size)
            max_new: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            device: Device to run on

        Returns:
            Generated fill tokens (without special markers)
        """
        from src.tokenizer import BOS, EOS, BAR, FILL_START, FILL_END

        self.eval()

        # Build FIM prompt: [BOS] + header + prefix + [FILL_START] + suffix + [FILL_END]
        hdr = header_tokens or []
        prompt = [BOS] + hdr + prefix_tokens + [FILL_START] + suffix_tokens + [FILL_END]
        idx = torch.tensor([prompt], dtype=torch.long, device=device)
        max_ctx = self.cfg["max_seq_len"]

        generated: list[int] = []
        bar_count = 0

        for _ in range(max_new):
            idx_cond = idx[:, -max_ctx:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Prevent EOS until the target number of bars has been generated
            if bar_count < target_bars:
                logits[:, EOS] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)
            next_token = next_t.item()

            if next_token == EOS:
                break

            if next_token == BAR:
                bar_count += 1
                print(f'Written {bar_count} bars of {target_bars}')
                if bar_count > target_bars:
                    break  # next measure starting — all target measures are complete

            generated.append(next_token)
            idx = torch.cat([idx, next_t], dim=1)

        return generated


def build_model(cfg: dict | None = None) -> MusicTransformer:
    cfg = cfg or DEFAULT_CONFIG
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    return MusicTransformer(cfg)

def append_epoch_to_config(cfg: dict, training_loss, val_loss):
    if cfg is None:
        print("Will not overwrite config with null config")
        return
    
    if 'epoches' not in cfg:
        cfg['epoches'] = [(training_loss, val_loss)]
    else: 
        cfg['epoches'].append((training_loss, val_loss))
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

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


def expand_embeddings_for_infill(
    model: MusicTransformer,
    old_vocab_size: int = 412
) -> MusicTransformer:
    """Expand model embeddings and head for infill tokens.

    Adds space for 3 new tokens: FILL_START, FILL_END, EMPTY_BAR.
    Initializes new tokens from semantically similar existing tokens:
    - FILL_START: initialized from BOS (marks a boundary)
    - FILL_END: initialized from EOS (marks end of context)
    - EMPTY_BAR: initialized from BAR (represents a measure)

    Args:
        model: Existing MusicTransformer model
        old_vocab_size: Original vocabulary size (default 412)

    Returns:
        Model with expanded embeddings
    """
    from src.tokenizer import BOS, EOS, BAR, VOCAB_SIZE

    new_vocab_size = VOCAB_SIZE  # 415
    d_model = model.cfg["d_model"]

    if model.cfg["vocab_size"] >= new_vocab_size:
        return model  # Already expanded

    # Create new embedding layer
    old_tok_emb = model.tok_emb.weight.data
    new_tok_emb = nn.Embedding(new_vocab_size, d_model)
    nn.init.normal_(new_tok_emb.weight, std=0.02)

    # Copy old embeddings
    new_tok_emb.weight.data[:old_vocab_size] = old_tok_emb

    # Initialize new tokens from similar existing tokens
    # FILL_START (412) ← BOS (1)
    new_tok_emb.weight.data[old_vocab_size] = old_tok_emb[BOS].clone()
    # FILL_END (413) ← EOS (2)
    new_tok_emb.weight.data[old_vocab_size + 1] = old_tok_emb[EOS].clone()
    # EMPTY_BAR (414) ← BAR (3)
    new_tok_emb.weight.data[old_vocab_size + 2] = old_tok_emb[BAR].clone()

    model.tok_emb = new_tok_emb

    # Create new output head
    old_head = model.head.weight.data
    new_head = nn.Linear(d_model, new_vocab_size, bias=False)
    nn.init.normal_(new_head.weight, std=0.02)

    # Copy old weights
    new_head.weight.data[:old_vocab_size] = old_head

    # Initialize new output weights similarly
    new_head.weight.data[old_vocab_size] = old_head[BOS].clone()
    new_head.weight.data[old_vocab_size + 1] = old_head[EOS].clone()
    new_head.weight.data[old_vocab_size + 2] = old_head[BAR].clone()

    model.head = new_head

    # Update config
    model.cfg["vocab_size"] = new_vocab_size

    return model


def expand_embeddings_for_metadata(
    model: MusicTransformer,
    old_vocab_size: int = 415,
) -> MusicTransformer:
    """Expand model embeddings from 415 to 463 to add TEMPO and INSTR tokens.

    New token initialisation:
    - TEMPO tokens (415-430): mean of all existing embeddings (semantically novel)
    - INSTR tokens (431-462): copy of BAR embedding (structural separator role)
    """
    from src.tokenizer import BAR, VOCAB_SIZE

    new_vocab_size = VOCAB_SIZE  # 463
    d_model = model.cfg["d_model"]

    if model.cfg["vocab_size"] >= new_vocab_size:
        return model  # Already expanded

    old_tok_emb = model.tok_emb.weight.data
    mean_emb = old_tok_emb.mean(dim=0)
    bar_emb  = old_tok_emb[BAR].clone()

    new_tok_emb = nn.Embedding(new_vocab_size, d_model)
    nn.init.normal_(new_tok_emb.weight, std=0.02)
    new_tok_emb.weight.data[:old_vocab_size] = old_tok_emb

    # TEMPO tokens <- mean embedding
    for i in range(16):
        new_tok_emb.weight.data[old_vocab_size + i] = mean_emb.clone()
    # INSTR tokens <- BAR embedding
    for i in range(32):
        new_tok_emb.weight.data[old_vocab_size + 16 + i] = bar_emb.clone()

    model.tok_emb = new_tok_emb

    old_head = model.head.weight.data
    new_head = nn.Linear(d_model, new_vocab_size, bias=False)
    nn.init.normal_(new_head.weight, std=0.02)
    new_head.weight.data[:old_vocab_size] = old_head
    for i in range(16):
        new_head.weight.data[old_vocab_size + i] = old_head.mean(dim=0).clone()
    for i in range(32):
        new_head.weight.data[old_vocab_size + 16 + i] = old_head[BAR].clone()

    model.head = new_head
    model.cfg["vocab_size"] = new_vocab_size
    return model


def build_infill_model(cfg: dict | None = None, override_cfg = False):
    """Build a model with expanded vocabulary for infill training."""
    from src.tokenizer import VOCAB_SIZE

    if not cfg:
        with open(CONFIG_FILE, "r") as f:
            cfg = json.load(f)

    cfg = (cfg and not override_cfg) or DEFAULT_CONFIG.copy()
    cfg["vocab_size"] = VOCAB_SIZE

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

    return MusicTransformer(cfg), cfg
