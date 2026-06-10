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

from ml.tokenizer import VOCAB_SIZE

load_dotenv('env.env')
CHECKPOINT_DIR = Path(os.getenv('STORAGE_DIR')) / "checkpoints"


def model_paths(name: str) -> dict[str, Path]:
    model_dir = CHECKPOINT_DIR / name
    return {
        "dir":       model_dir,
        "config":    model_dir / "config.json",
        "weights":   model_dir / "model.pt",
        "optimizer": model_dir / "optimizer.pt",
        "val_split": model_dir / "val_paths.json",
    }


# Legacy aliases — app.py still imports these directly
CONFIG_FILE = model_paths("infill")["config"]
WEIGHTS_FILE = model_paths("infill")["weights"]
OPTIMIZER_FILE = model_paths("infill")["optimizer"]
VAL_SPLIT_FILE = model_paths("infill")["val_split"]

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
        from ml.tokenizer import EOS
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
        from ml.tokenizer import BOS, EOS, BAR, FILL_START, FILL_END

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

    @torch.no_grad()
    def generate_standard(
        self,
        header_tokens: list[int],
        first_bar_prefix: list[int] | None = None,
        target_bars: int = 4,
        max_new: int = 1000,
        temperature: float = 0.9,
        top_k: int = 50,
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
    ) -> list[int]:
        """Generate tokens from scratch using the standard autoregressive format.

        Uses [BOS] + header + first_bar_prefix as the prompt, then samples
        autoregressively. This is in-distribution for the 50% standard training
        examples, unlike generate_infill() with empty prefix/suffix which the
        model has never seen during training.

        Args:
            header_tokens: Instrument header tokens (INSTR_OFF + idx)
            first_bar_prefix: Tokens to force after the header, e.g.
                [BAR, KEY_SIG_token, TIME_SIG_token, TEMPO_token]. Forces the
                model into a known key/meter/tempo before free sampling begins.
                Defaults to [BAR] if None.
            target_bars: Total number of BAR tokens to produce (including any
                BAR tokens in first_bar_prefix)
            max_new: Maximum tokens to generate autoregressively
            temperature: Sampling temperature (0.8–1.0 recommended)
            top_k: Top-k sampling parameter
            device: Device to run on

        Returns:
            Generated measure tokens (without BOS/EOS; prepend header and
            wrap with BOS/EOS when constructing the full sequence)
        """
        from ml.tokenizer import BOS, EOS, BAR

        self.eval()

        forced = first_bar_prefix if first_bar_prefix is not None else [BAR]
        # Count any BAR tokens already committed via first_bar_prefix
        bar_count = forced.count(BAR)

        prompt = [BOS] + header_tokens + forced
        idx = torch.tensor([prompt], dtype=torch.long, device=device)
        max_ctx = self.cfg["max_seq_len"]

        generated: list[int] = []

        for _ in range(max_new):
            idx_cond = idx[:, -max_ctx:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

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
                    break

            generated.append(next_token)
            idx = torch.cat([idx, next_t], dim=1)

        return generated

    @torch.no_grad()
    def generate_standard_stream(
        self,
        header_tokens: list[int],
        first_bar_prefix: list[int] | None = None,
        target_bars: int = 4,
        max_new: int = 1000,
        temperature: float = 0.9,
        top_k: int = 50,
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
    ):
        """Like generate_standard but yields accumulated token list after each completed BAR."""
        from ml.tokenizer import BOS, EOS, BAR

        self.eval()

        forced = first_bar_prefix if first_bar_prefix is not None else [BAR]
        bar_count = forced.count(BAR)

        prompt = [BOS] + header_tokens + forced
        idx = torch.tensor([prompt], dtype=torch.long, device=device)
        max_ctx = self.cfg["max_seq_len"]

        generated: list[int] = []

        for i in range(max_new):
            idx_cond = idx[:, -max_ctx:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if bar_count < target_bars:
                logits[:, EOS] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)
            next_token = next_t.item()

            if next_token == EOS:
                break

            if next_token == BAR:
                bar_count += 1
                if bar_count > target_bars:
                    break

            generated.append(next_token)
            idx = torch.cat([idx, next_t], dim=1)

            if i % 2 == 0:
                yield list(generated)

        yield list(generated)

    @torch.no_grad()
    def generate_infill_stream(
        self,
        prefix_tokens: list[int],
        suffix_tokens: list[int],
        header_tokens: list[int] | None = None,
        target_bars: int = 4,
        max_new: int = 400,
        temperature: float = 1.0,
        top_k: int = 50,
        device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu",
    ):
        """Like generate_infill but yields accumulated token list after each completed BAR."""
        from ml.tokenizer import BOS, EOS, BAR, FILL_START, FILL_END

        self.eval()

        hdr = header_tokens or []
        prompt = [BOS] + hdr + prefix_tokens + [FILL_START] + suffix_tokens + [FILL_END]
        idx = torch.tensor([prompt], dtype=torch.long, device=device)
        max_ctx = self.cfg["max_seq_len"]

        generated: list[int] = []
        bar_count = 0

        for i in range(max_new):
            idx_cond = idx[:, -max_ctx:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if bar_count < target_bars:
                logits[:, EOS] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)
            next_token = next_t.item()

            if next_token == EOS:
                break

            if next_token == BAR:
                bar_count += 1
                if bar_count > target_bars:
                    break

            generated.append(next_token)
            idx = torch.cat([idx, next_t], dim=1)

            # Yield snapshot after each completed measure (BAR token now included)
            if i % 2 == 0:
                yield list(generated)

        yield list(generated)


def append_epoch_to_config(name: str, cfg: dict, training_loss, val_loss):
    if cfg is None:
        print("Will not overwrite config with null config")
        return

    if 'epoches' not in cfg:
        cfg['epoches'] = [(training_loss, val_loss)]
    else:
        cfg['epoches'].append((training_loss, val_loss))

    with open(model_paths(name)["config"], "w") as f:
        json.dump(cfg, f, indent=2)

def load_model(
    name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[MusicTransformer, dict]:
    paths = model_paths(name)
    print(paths)
    config_path = paths["config"]
    if not config_path.exists() and name == "infill":
        legacy = CHECKPOINT_DIR / "config.json"
        if legacy.exists():
            config_path = legacy
    if not config_path.exists():
        raise FileNotFoundError(f"No config for model '{name}'. Run 'Build Model' first.")
    with open(config_path) as f:
        cfg = json.load(f)
    model = MusicTransformer(cfg).to(device)
    weights = paths["weights"]
    if not weights.exists() and name == "infill":
        legacy_w = CHECKPOINT_DIR / "infill_model.pt"
        if legacy_w.exists():
            weights = legacy_w
    if weights.exists():
        state = torch.load(weights, map_location=device, weights_only=False)
        model.load_state_dict(state)
    return model, cfg

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
    from ml.tokenizer import BOS, EOS, BAR, VOCAB_SIZE

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
    from ml.tokenizer import BAR, VOCAB_SIZE

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


def build_model(name: str, cfg: dict | None = None, override_cfg=False):
    """Build a model with expanded vocabulary."""
    from ml.tokenizer import VOCAB_SIZE

    paths = model_paths(name)
    if not cfg and paths["config"].exists():
        with open(paths["config"], "r") as f:
            cfg = json.load(f)

    cfg = DEFAULT_CONFIG.copy() if override_cfg else (cfg or DEFAULT_CONFIG.copy())
    cfg["vocab_size"] = VOCAB_SIZE

    paths["dir"].mkdir(parents=True, exist_ok=True)
    with open(paths["config"], "w") as f:
        json.dump(cfg, f, indent=2)

    return MusicTransformer(cfg), cfg
