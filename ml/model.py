"""MMT-BERT generator: Multitrack-Music-Transformer-style decoder over
compound events (arXiv 2409.00919).

Each input position is a 7-field event row (see :mod:`ml.events`).  Fields are
embedded separately, concatenated, and projected to d_model (the paper's
"individually embedded first, and then concatenated"); the decoder output is
mapped through one linear head per field ("separate softmax layers ... to the
vocabulary sizes of the different element types").

Sampling is type-first: the event type is sampled from the type head, then the
remaining fields are sampled from their heads under hard per-type validity
masks (a field the type does not use is forced to its structural zero, a field
it does use is masked to its legal range).  The bar field is never sampled —
it is forced deterministically from BAR-event counting, which is what makes
measure alignment during infill exact.

Checkpoint layout is unchanged from the previous system:
``STORAGE_DIR/checkpoints/<name>/{config.json, model.pt, optimizer.pt,
val_paths.json}``.  ``config.json`` now carries ``model_type`` ("mmt_gen" or
"musicbert_disc") and :func:`load_model` dispatches on it; legacy flat-token
configs (those with a ``vocab_size`` key) raise a clear error.
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

from ml.events import (
    F_TYPE, F_BAR, F_POS, F_PITCH, F_DUR, F_TRACK, F_VALUE, N_FIELDS,
    PAD_T, SOS_T, EOS_T, INSTR_T, BAR_T, KEY_T, TIME_T, TEMPO_T, CLEF_T,
    NOTE_T, GRACE_T, REST_T, CHORD_T, FILL_START_T, FILL_END_T, FILL_GEN_T,
    N_TYPES, FIELD_VOCAB_SIZES, FIELD_VALID_RANGES, CONTENT_TYPES,
    MAX_BARS, MAX_POS, MAX_PARTS, MAX_VOICES_PER_PART, MAX_REAL_PARTS, N_CLEFS,
    instr_event,
)

load_dotenv('env.env')
CHECKPOINT_DIR = Path(os.getenv('STORAGE_DIR')) / "checkpoints"

DEFAULT_FIELD_EMB_DIMS = [32, 128, 96, 128, 32, 32, 64]  # sums to 512

DEFAULT_GEN_CONFIG = dict(
    model_type = "mmt_gen",
    schema_version = 2,
    field_vocab_sizes = list(FIELD_VOCAB_SIZES),
    field_emb_dims = list(DEFAULT_FIELD_EMB_DIMS),
    d_model = 512,
    n_heads = 8,
    n_layers = 6,
    max_seq_len = 1024,
    dropout = 0.1,
    epoches = [],
)

DEFAULT_DISC_CONFIG = dict(
    model_type = "musicbert_disc",
    schema_version = 2,
    field_vocab_sizes = list(FIELD_VOCAB_SIZES),
    field_emb_dims = list(DEFAULT_FIELD_EMB_DIMS),
    d_model = 512,
    n_heads = 8,
    n_layers = 6,
    max_seq_len = 1024,
    dropout = 0.1,
    epoches = [],
)


def model_paths(name: str) -> dict[str, Path]:
    model_dir = CHECKPOINT_DIR / name
    return {
        "dir":       model_dir,
        "config":    model_dir / "config.json",
        "weights":   model_dir / "model.pt",
        "optimizer": model_dir / "optimizer.pt",
        "val_split": model_dir / "val_paths.json",
    }


def _disallowed_track_values(target_parts: list[int] | None) -> list[int]:
    """Track-field values to mask so generation only writes target staves.

    Track value v encodes lane v-1 = part*2 + voice; a value is disallowed when
    its real part index is not in target_parts.  Empty when target_parts is None.
    """
    if target_parts is None:
        return []
    allowed = set(target_parts)
    return [
        lane + 1
        for lane in range(MAX_PARTS)
        if (lane // MAX_VOICES_PER_PART) not in allowed
    ]


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """key_padding_mask: [B, T] bool, True = attend (only used when not causal)."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if not self.causal and key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=self.causal,
        )
        out = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, causal: bool = True):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, causal=causal)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CompoundEmbedding(nn.Module):
    """Per-field embeddings, concatenated and projected to d_model.

    ``forward`` takes hard event rows [B, T, 7]; ``forward_soft`` takes a list
    of 7 per-field probability tensors [B, T, V_k] and computes the expected
    embedding (probs @ weight) — the differentiable path the GAN generator
    uses to reach the discriminator through discrete tokens.  Hard rows fed to
    the discriminator during GAN training should be routed through
    ``forward_soft`` with one-hots so real and fake share one code path.
    """

    def __init__(self, field_vocab_sizes: list[int], field_emb_dims: list[int], d_model: int):
        super().__init__()
        assert len(field_vocab_sizes) == len(field_emb_dims) == N_FIELDS
        self.embs = nn.ModuleList([
            nn.Embedding(v, d) for v, d in zip(field_vocab_sizes, field_emb_dims)
        ])
        self.proj = nn.Linear(sum(field_emb_dims), d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        parts = [emb(x[..., k]) for k, emb in enumerate(self.embs)]
        return self.proj(torch.cat(parts, dim=-1))

    def forward_soft(self, field_probs: list[Tensor]) -> Tensor:
        parts = [probs @ emb.weight for probs, emb in zip(field_probs, self.embs)]
        return self.proj(torch.cat(parts, dim=-1))


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_head(
    logits: Tensor,
    temperature: float,
    top_k: int,
    allowed: Tensor | None = None,
) -> int:
    """Sample one value from a 1-D head under an optional boolean allow-mask."""
    logits = logits / max(temperature, 1e-6)
    if allowed is not None:
        logits = logits.masked_fill(~allowed, float("-inf"))
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < v[-1], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all() or probs.sum() <= 0:
        # Everything masked out (shouldn't happen): fall back to argmax of raw.
        return int(logits.argmax().item())
    return int(torch.multinomial(probs, 1).item())


def _range_mask(size: int, lo: int, hi: int, device) -> Tensor:
    mask = torch.zeros(size, dtype=torch.bool, device=device)
    mask[lo:hi + 1] = True
    return mask


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MMTGenerator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        vocab = cfg.get("field_vocab_sizes", FIELD_VOCAB_SIZES)
        dims  = cfg.get("field_emb_dims", DEFAULT_FIELD_EMB_DIMS)
        self.embed   = CompoundEmbedding(vocab, dims, d)
        self.pos_emb = nn.Embedding(cfg["max_seq_len"], d)
        self.drop    = nn.Dropout(cfg["dropout"])
        self.blocks  = nn.ModuleList(
            [Block(d, cfg["n_heads"], cfg["dropout"], causal=True)
             for _ in range(cfg["n_layers"])]
        )
        self.ln_f  = nn.LayerNorm(d)
        self.heads = nn.ModuleList([nn.Linear(d, v, bias=False) for v in vocab])
        self.apply(self._init_weights)

        residual_scale = 1.0 / math.sqrt(2 * cfg["n_layers"])
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=0.02 * residual_scale)
            nn.init.normal_(block.mlp[-2].weight, std=0.02 * residual_scale)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def hidden(self, x: Tensor) -> Tensor:
        """[B, T, 7] event rows -> [B, T, d_model] final hidden states."""
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.embed(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        return self.ln_f(h)

    def forward(self, x: Tensor) -> list[Tensor]:
        """[B, T, 7] -> list of 7 per-field logits tensors [B, T, V_k]."""
        h = self.hidden(x)
        return [head(h) for head in self.heads]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- sampling ------------------------------------------------------------

    def _step_logits(self, idx: Tensor) -> list[Tensor]:
        """Per-field logits at the last position. idx is [1, T, 7]."""
        max_ctx = self.cfg["max_seq_len"]
        h = self.hidden(idx[:, -max_ctx:])
        return [head(h[:, -1, :])[0] for head in self.heads]

    def _sample_event(
        self,
        logits: list[Tensor],
        temperature: float,
        top_k: int,
        cur_bar: int | None,
        next_bar: int,
        last_pos: int,
        allow_eos: bool,
        allow_bar: bool,
        track_mask: list[int],
        max_parts: int | None,
        device,
    ) -> list[int]:
        """Sample one event row under structural constraints (type first)."""
        # -- type ---------------------------------------------------------
        type_allowed = torch.ones(N_TYPES, dtype=torch.bool, device=device)
        type_allowed[[PAD_T, SOS_T, INSTR_T, FILL_START_T, FILL_END_T, FILL_GEN_T]] = False
        if not allow_eos:
            type_allowed[EOS_T] = False
        if not allow_bar:
            type_allowed[BAR_T] = False
        if cur_bar is None:
            # No measure open yet: the only legal continuations are BAR / EOS.
            content = torch.zeros_like(type_allowed)
            content[BAR_T] = allow_bar
            content[EOS_T] = allow_eos
            type_allowed &= content
        t = _sample_head(logits[F_TYPE], temperature, top_k, type_allowed)

        row = [t, 0, 0, 0, 0, 0, 0]
        if t == EOS_T:
            return row

        # -- bar (forced, never sampled) ------------------------------------
        row[F_BAR] = min(next_bar if t == BAR_T else (cur_bar or 0), MAX_BARS - 1)

        # -- pos -------------------------------------------------------------
        if t in CONTENT_TYPES:
            allowed = _range_mask(MAX_POS, min(last_pos, MAX_POS - 1), MAX_POS - 1, device)
            row[F_POS] = _sample_head(logits[F_POS], temperature, top_k, allowed)

        # -- remaining fields under per-type validity ranges -----------------
        ranges = FIELD_VALID_RANGES.get(t, {})
        for f in (F_PITCH, F_DUR, F_TRACK, F_VALUE):
            lo, hi = ranges.get(f, (0, 0))
            if (lo, hi) == (0, 0):
                continue
            allowed = _range_mask(FIELD_VOCAB_SIZES[f], lo, hi, device)
            if f == F_TRACK and t in CONTENT_TYPES:
                for v in track_mask:
                    allowed[v] = False
                if max_parts is not None:
                    allowed[max_parts * MAX_VOICES_PER_PART + 1:] = False
                if not allowed.any():
                    allowed = _range_mask(FIELD_VOCAB_SIZES[f], lo, hi, device)
            row[f] = _sample_head(logits[f], temperature, top_k, allowed)
        return row

    @torch.no_grad()
    def _generate_loop(
        self,
        prompt_events: list[list[int]],
        target_bars: int,
        bar_count: int,
        cur_bar: int | None,
        next_bar: int,
        last_pos: int,
        max_new: int,
        temperature: float,
        top_k: int,
        device: str,
        target_parts: list[int] | None = None,
        max_parts: int | None = None,
    ):
        """Shared autoregressive loop.  Yields the accumulated generated rows
        after each completed BAR and once at the end."""
        self.eval()
        idx = torch.tensor([prompt_events], dtype=torch.long, device=device)
        track_mask = _disallowed_track_values(target_parts)

        generated: list[list[int]] = []
        for _ in range(max_new):
            logits = self._step_logits(idx)
            allow_eos = bar_count >= target_bars
            allow_bar = True
            row = self._sample_event(
                logits, temperature, top_k, cur_bar, next_bar, last_pos,
                allow_eos, allow_bar, track_mask, max_parts, device,
            )

            if row[F_TYPE] == EOS_T:
                break
            if row[F_TYPE] == BAR_T:
                bar_count += 1
                if bar_count > target_bars:
                    break  # next measure starting — all target measures done
                cur_bar = row[F_BAR]
                next_bar = cur_bar + 1
                last_pos = 0
                if generated:
                    yield list(generated)
            elif row[F_TYPE] in CONTENT_TYPES:
                last_pos = row[F_POS]

            generated.append(row)
            idx = torch.cat(
                [idx, torch.tensor([[row]], dtype=torch.long, device=device)], dim=1)

        yield list(generated)

    @staticmethod
    def _prompt_bar_state(events: list[list[int]]) -> tuple[int, int | None, int, int]:
        """(bar_count, cur_bar, next_bar, last_pos) implied by committed events."""
        bar_count, cur_bar, last_pos = 0, None, 0
        for row in events:
            if row[F_TYPE] == BAR_T:
                bar_count += 1
                cur_bar = row[F_BAR]
                last_pos = 0
            elif row[F_TYPE] in CONTENT_TYPES and cur_bar is not None:
                last_pos = max(last_pos, row[F_POS])
        next_bar = (cur_bar + 1) if cur_bar is not None else 0
        return bar_count, cur_bar, next_bar, last_pos

    @torch.no_grad()
    def generate_standard(
        self,
        header_events: list[list[int]],
        forced_events: list[list[int]] | None = None,
        target_bars: int = 4,
        max_new: int = 1000,
        temperature: float = 0.9,
        top_k: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_parts: int | None = None,
    ) -> list[list[int]]:
        """Generate event rows from scratch: [SOS] + header + forced_events prompt.

        ``forced_events`` are committed rows (e.g. an opening
        [BAR, KEY, TIME, TEMPO] block, or a whole context window for append);
        their BAR rows count toward ``target_bars``.
        """
        for out in self.generate_standard_stream(
            header_events, forced_events, target_bars, max_new,
            temperature, top_k, device, max_parts,
        ):
            result = out
        return result

    @torch.no_grad()
    def generate_standard_stream(
        self,
        header_events: list[list[int]],
        forced_events: list[list[int]] | None = None,
        target_bars: int = 4,
        max_new: int = 1000,
        temperature: float = 0.9,
        top_k: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_parts: int | None = None,
    ):
        """Like generate_standard but yields the accumulated rows after each bar."""
        from ml.events import sos_event, bar_event
        forced = forced_events if forced_events is not None else [bar_event(0)]
        bar_count, cur_bar, next_bar, last_pos = self._prompt_bar_state(forced)
        prompt = [sos_event()] + list(header_events) + list(forced)
        yield from self._generate_loop(
            prompt, target_bars, bar_count, cur_bar, next_bar, last_pos,
            max_new, temperature, top_k, device, max_parts=max_parts,
        )

    @torch.no_grad()
    def generate_infill(
        self,
        prefix_events: list[list[int]],
        suffix_events: list[list[int]],
        header_events: list[list[int]] | None = None,
        target_bars: int = 4,
        gap_bar_start: int = 0,
        max_new: int = 400,
        temperature: float = 1.0,
        top_k: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        kept_gap_events: list[list[int]] | None = None,
        target_parts: list[int] | None = None,
    ) -> list[list[int]]:
        """FIM infill: fill ``target_bars`` measures at window bars
        gap_bar_start..gap_bar_start+target_bars-1, given prefix/suffix context
        and (for partial-staff infill) the kept staves' gap content."""
        for out in self.generate_infill_stream(
            prefix_events, suffix_events, header_events, target_bars,
            gap_bar_start, max_new, temperature, top_k, device,
            kept_gap_events, target_parts,
        ):
            result = out
        return result

    @torch.no_grad()
    def generate_infill_stream(
        self,
        prefix_events: list[list[int]],
        suffix_events: list[list[int]],
        header_events: list[list[int]] | None = None,
        target_bars: int = 4,
        gap_bar_start: int = 0,
        max_new: int = 400,
        temperature: float = 1.0,
        top_k: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        kept_gap_events: list[list[int]] | None = None,
        target_parts: list[int] | None = None,
    ):
        """Like generate_infill but yields the accumulated rows after each bar."""
        from ml.events import (
            sos_event, fill_start_event, fill_end_event, fill_gen_event,
        )
        hdr  = header_events or []
        kept = kept_gap_events or []
        prompt = ([sos_event()] + list(hdr) + list(prefix_events)
                  + [fill_start_event()] + list(suffix_events)
                  + [fill_end_event()] + list(kept) + [fill_gen_event()])
        yield from self._generate_loop(
            prompt, target_bars, bar_count=0, cur_bar=None,
            next_bar=gap_bar_start, last_pos=0,
            max_new=max_new, temperature=temperature, top_k=top_k,
            device=device, target_parts=target_parts,
        )

    @torch.no_grad()
    def sample_header(
        self,
        instr_indices: list[int],
        temperature: float = 0.9,
        top_k: int = 50,
        device: str = "cpu",
    ) -> list[list[int]]:
        """Build header INSTR events, letting the model pick each part's clef.

        Instrument category and part index are forced; the clef sub-field
        (pitch field of the INSTR event) is sampled from the model conditioned
        on the header so far — the compound-event analogue of the old
        "sample a CLEF token after each INSTR token" trick.
        """
        from ml.events import sos_event
        self.eval()
        header: list[list[int]] = []
        for part_idx, instr_idx in enumerate(instr_indices[:MAX_REAL_PARTS]):
            prompt = [sos_event()] + header
            idx = torch.tensor([prompt], dtype=torch.long, device=device)
            logits = self._step_logits(idx)
            allowed = _range_mask(FIELD_VOCAB_SIZES[F_PITCH], 1, N_CLEFS, device)
            clef_val = _sample_head(logits[F_PITCH], temperature, top_k, allowed)
            header.append(instr_event(part_idx, instr_idx, clef_val - 1))
        return header


# ---------------------------------------------------------------------------
# Checkpoint plumbing (layout unchanged from the previous system)
# ---------------------------------------------------------------------------

def append_epoch_to_config(name: str, cfg: dict, training_loss, val_loss):
    if cfg is None:
        return

    if 'epoches' not in cfg:
        cfg['epoches'] = [(training_loss, val_loss)]
    else:
        cfg['epoches'].append((training_loss, val_loss))

    with open(model_paths(name)["config"], "w") as f:
        json.dump(cfg, f, indent=2)


def _model_from_config(cfg: dict, name: str) -> nn.Module:
    model_type = cfg.get("model_type")
    if model_type is None:
        if "vocab_size" in cfg:
            raise ValueError(
                f"'{name}' is a legacy flat-token checkpoint (pre-MMT-BERT, "
                f"vocab_size={cfg['vocab_size']}). It cannot be loaded by the "
                "compound-event models — train a new MMT-BERT checkpoint."
            )
        raise ValueError(f"Config for '{name}' has no model_type.")
    if model_type == "mmt_gen":
        return MMTGenerator(cfg)
    if model_type == "musicbert_disc":
        from ml.musicbert import MusicBERTDiscriminator
        return MusicBERTDiscriminator(cfg)
    raise ValueError(f"Unknown model_type '{model_type}' for '{name}'.")


def load_model(
    name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Load a checkpoint by name; returns (model, cfg).

    Dispatches on config.json's ``model_type``: "mmt_gen" -> MMTGenerator,
    "musicbert_disc" -> MusicBERTDiscriminator.
    """
    paths = model_paths(name)
    config_path = paths["config"]
    if not config_path.exists():
        raise FileNotFoundError(f"No config for model '{name}'. Run 'Build Model' first.")
    with open(config_path) as f:
        cfg = json.load(f)
    model = _model_from_config(cfg, name).to(device)
    weights = paths["weights"]
    if weights.exists():
        state = torch.load(weights, map_location=device, weights_only=True)
        model.load_state_dict(state)
    return model, cfg


def build_model(
    name: str,
    cfg: dict | None = None,
    override_cfg: bool = False,
    model_type: str = "mmt_gen",
):
    """Create (or re-open) a model checkpoint dir and return (model, cfg).

    The schema-derived config keys (field vocab sizes, model_type) are always
    forced to the current code's values, mirroring how the old build_model
    pinned vocab_size.
    """
    paths = model_paths(name)
    if not cfg and paths["config"].exists() and not override_cfg:
        with open(paths["config"], "r") as f:
            cfg = json.load(f)

    default = DEFAULT_DISC_CONFIG if model_type == "musicbert_disc" else DEFAULT_GEN_CONFIG
    cfg = default.copy() if (override_cfg or not cfg) else cfg
    cfg["model_type"] = model_type
    cfg["schema_version"] = 2
    cfg["field_vocab_sizes"] = list(FIELD_VOCAB_SIZES)
    cfg.setdefault("field_emb_dims", list(DEFAULT_FIELD_EMB_DIMS))

    paths["dir"].mkdir(parents=True, exist_ok=True)
    with open(paths["config"], "w") as f:
        json.dump(cfg, f, indent=2)

    return _model_from_config(cfg, name), cfg
