"""MusicBERT-style bidirectional encoder used as the MMT-BERT discriminator.

Implemented in-repo in pure PyTorch over our compound-event vocabulary
(rather than adapting Microsoft's fairseq/OctupleMIDI checkpoint): the paper's
"quintuple tokens are converted into a single vector through the concatenation
of embeddings and a linear layer" is exactly :class:`ml.model.CompoundEmbedding`.

Two heads on top of the shared encoder:

  * MLM: one linear per field, used to pretrain the encoder on our token cache
    with masked-event prediction (MusicBERT's objective) — see
    :mod:`ml.mlm_trainer`.
  * critic: a single non-transformed linear on the [CLS] state — the C(.)
    scored by the relativistic GAN loss during fine-tuning — see
    :mod:`ml.gan_trainer`.

Masked positions have their entire event embedding replaced by a learned
[MASK] vector (masking the whole compound event, so no field of a masked
event leaks to the predictor).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ml.events import (
    F_TYPE, N_FIELDS, PAD_T, SOS_T, EOS_T,
    FILL_START_T, FILL_END_T, FILL_GEN_T,
    FIELD_VOCAB_SIZES,
)
from ml.model import Block, CompoundEmbedding, DEFAULT_FIELD_EMB_DIMS

# Event types that are never selected for masking (and never scored by MLM).
SPECIAL_TYPES = (PAD_T, SOS_T, EOS_T, FILL_START_T, FILL_END_T, FILL_GEN_T)


class MusicBERTDiscriminator(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        vocab = cfg.get("field_vocab_sizes", FIELD_VOCAB_SIZES)
        dims  = cfg.get("field_emb_dims", DEFAULT_FIELD_EMB_DIMS)

        self.embed    = CompoundEmbedding(vocab, dims, d)
        # +1 position for the prepended [CLS] vector.
        self.pos_emb  = nn.Embedding(cfg["max_seq_len"] + 1, d)
        self.cls_emb  = nn.Parameter(torch.randn(d) * 0.02)
        self.mask_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.drop     = nn.Dropout(cfg["dropout"])
        self.blocks   = nn.ModuleList(
            [Block(d, cfg["n_heads"], cfg["dropout"], causal=False)
             for _ in range(cfg["n_layers"])]
        )
        self.ln_f      = nn.LayerNorm(d)
        self.mlm_heads = nn.ModuleList([nn.Linear(d, v, bias=False) for v in vocab])
        self.critic    = nn.Linear(d, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- shared encoder -------------------------------------------------------

    def _encode_embedded(self, emb: Tensor, pad_mask: Tensor | None) -> Tensor:
        """[B, T, d] event embeddings -> [B, T+1, d] hidden states (CLS first).

        pad_mask: [B, T] bool, True at real (non-PAD) positions.
        """
        B, T, d = emb.shape
        cls = self.cls_emb.expand(B, 1, d)
        h = torch.cat([cls, emb], dim=1)
        pos = torch.arange(T + 1, device=emb.device)
        h = self.drop(h + self.pos_emb(pos))
        attend = None
        if pad_mask is not None:
            attend = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=emb.device), pad_mask], dim=1)
        for block in self.blocks:
            h = block(h, key_padding_mask=attend)
        return self.ln_f(h)

    @staticmethod
    def _pad_mask(x: Tensor) -> Tensor:
        return x[..., F_TYPE] != PAD_T

    # -- MLM head (pretraining) ------------------------------------------------

    def forward_mlm(self, x: Tensor, mask_positions: Tensor) -> list[Tensor]:
        """Masked-event prediction.

        Args:
            x: [B, T, 7] event rows (the TRUE rows; masking happens internally).
            mask_positions: [B, T] bool — positions whose event embedding is
                replaced by the learned [MASK] vector.

        Returns per-field logits [B, T, V_k] over the event positions.
        """
        emb = self.embed(x)
        emb = torch.where(mask_positions.unsqueeze(-1), self.mask_emb.to(emb.dtype), emb)
        h = self._encode_embedded(emb, self._pad_mask(x))[:, 1:]
        return [head(h) for head in self.mlm_heads]

    # -- critic head (GAN fine-tuning) ------------------------------------------

    def critic_score(self, x: Tensor) -> Tensor:
        """[B, T, 7] hard event rows -> [B] critic scores C(x)."""
        h = self._encode_embedded(self.embed(x), self._pad_mask(x))
        return self.critic(h[:, 0]).squeeze(-1)

    def critic_score_soft(
        self, field_probs: list[Tensor], pad_mask: Tensor | None = None,
    ) -> Tensor:
        """Differentiable critic over per-field distributions.

        Args:
            field_probs: 7 tensors [B, T, V_k] (softmax/Gumbel outputs from the
                generator, or one-hots for real rows — route reals through here
                too so both share one code path).
            pad_mask: [B, T] bool, True at real positions.
        """
        emb = self.embed.forward_soft(field_probs)
        h = self._encode_embedded(emb, pad_mask)
        return self.critic(h[:, 0]).squeeze(-1)


def maskable_positions(x: Tensor) -> Tensor:
    """[B, T, 7] -> [B, T] bool: positions eligible for MLM masking."""
    types = x[..., F_TYPE]
    eligible = torch.ones_like(types, dtype=torch.bool)
    for t in SPECIAL_TYPES:
        eligible &= types != t
    return eligible
