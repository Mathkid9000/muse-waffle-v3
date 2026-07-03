"""MMT-BERT phase (c): adversarial fine-tuning of the MLE-pretrained generator
against the MLM-pretrained MusicBERT discriminator (arXiv 2409.00919 §3.3).

Loss — relativistic standard GAN with a cross-entropy anchor.  The paper
prints

    L_D = E[log sigmoid(C(f) - C(r))]
    L_m = E[log sigmoid(C(r) - C(f))] - sum_i r_i log f_i

which drives the same fixed points as standard RSGAN but is unbounded below
(log sigmoid -> -inf) and numerically hazardous; we implement the equivalent
softplus form, which has the same optimum:

    L_D = softplus(C(f) - C(r)).mean()          # = -E[log sigmoid(C(r)-C(f))]
    L_G = softplus(C(r) - C(f)).mean() + lambda_ce * L_MLE

Gradient flow through discrete tokens (the paper is silent): the generator is
run teacher-forced on the real batch; its per-field distributions (optionally
hardened with straight-through Gumbel-softmax, tau annealed) are fed to the
discriminator as *soft embeddings* via CompoundEmbedding.forward_soft.  Real
rows go through the same soft path as one-hots so real and fake share one code
path.  Only the loss-masked region is replaced by generator output (for FIM
examples that is exactly the fill region — matching what inference produces);
the rest of the fake sequence is the real context.

Optimizer: Adagrad by default (the paper's choice, "to mitigate gradient
explosion/vanishing"); pass optimizer="adamw" (lr ~1e-5) if Adagrad's
monotonically shrinking step stalls long runs.

Checkpoints: generator weights are COPIED into a fresh ``out_name`` dir (so
the MLE checkpoint survives GAN instability) and the fine-tuned discriminator
into ``out_name + "-DISC"``.  Per-epoch (train_G, val_CE) pairs go through the
usual on_epoch callback (wire it to append_epoch_to_config); discriminator
losses are appended to ``cfg["gan"]["d_epoches"]`` in out_name's config.json.
Stop early if val CE rises more than ~10% above the phase-(a) baseline.
"""

from __future__ import annotations
import json
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.events import F_TYPE, N_FIELDS, PAD_T
from ml.model import load_model, model_paths, append_epoch_to_config
from ml.trainer import MusicDataset, _make_split, compound_ce_loss
from ml.infill_trainer import create_fim_example
from ml.preprocess import build_or_open_cache, _path_key, DEFAULT_CACHE_DIR


def _init_out_checkpoint(src_name: str, out_name: str, device: str, extra_cfg: dict | None = None):
    """Copy a checkpoint (config + weights) into a fresh dir; returns (model, cfg)."""
    model, cfg = load_model(src_name, device)
    cfg = dict(cfg)
    cfg["epoches"] = []
    cfg["gan_source"] = src_name
    if extra_cfg:
        cfg.update(extra_cfg)
    paths = model_paths(out_name)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    with open(paths["config"], "w") as f:
        json.dump(cfg, f, indent=2)
    torch.save(model.state_dict(), paths["weights"])
    return model, cfg


def _field_probs(
    logits: list[torch.Tensor], gumbel: bool, tau: float,
) -> list[torch.Tensor]:
    """Per-field distributions from generator logits (optionally ST-Gumbel)."""
    if gumbel:
        return [F.gumbel_softmax(lg, tau=tau, hard=True, dim=-1) for lg in logits]
    return [F.softmax(lg, dim=-1) for lg in logits]


def _one_hot_rows(y: torch.Tensor, vocab_sizes: list[int]) -> list[torch.Tensor]:
    return [F.one_hot(y[..., k], vocab_sizes[k]).float() for k in range(N_FIELDS)]


def _mix_fake(
    gen_probs: list[torch.Tensor],
    real_onehot: list[torch.Tensor],
    loss_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Generator output at loss-masked positions, real one-hots elsewhere."""
    m = loss_mask.unsqueeze(-1)
    return [m * g + (1.0 - m) * r for g, r in zip(gen_probs, real_onehot)]


def train_gan(
    gen_name: str,
    disc_name: str,
    out_name: str,
    data_paths: list[str],
    epochs: int,
    batch_size: int,
    device: str,
    log: Callable[[str], None],
    on_epoch: Callable[[int, float, float], None],
    stop_flag: list[bool],
    lambda_ce: float = 1.0,
    optimizer: str = "adagrad",
    lr_g: float = 1e-2,
    lr_d: float = 1e-2,
    gumbel: bool = True,
    tau_start: float = 2.0,
    tau_end: float = 0.5,
    fim_ratio: float = 0.5,
    resume: bool = False,
    cache_dir=None,
    cache_workers: int = 1,
):
    """Adversarially fine-tune generator `gen_name` against discriminator
    `disc_name`; writes results to `out_name` (+ `out_name`-DISC).

    Args:
        lambda_ce: weight of the teacher-forced CE anchor in L_G.
        optimizer: "adagrad" (paper) or "adamw" (use lr ~1e-5).
        gumbel: straight-through Gumbel-softmax on the fake distributions
            (tau annealed tau_start -> tau_end linearly over training);
            False feeds plain softmax probabilities.
        fim_ratio: fraction of FIM examples in the real batches (keep equal to
            the phase-(a) value so the generator's infill skill is preserved).
        resume: continue a previous GAN run (loads out_name / out_name-DISC
            instead of re-copying from gen_name / disc_name).
    """
    disc_out_name = out_name + "-DISC"

    if resume and model_paths(out_name)["config"].exists():
        gen, gen_cfg = load_model(out_name, device)
        disc, _ = load_model(disc_out_name, device)
        log(f"Resumed GAN checkpoints {out_name} / {disc_out_name}.")
    else:
        gen, gen_cfg = _init_out_checkpoint(
            gen_name, out_name, device,
            extra_cfg={"gan": {
                "disc": disc_name, "lambda_ce": lambda_ce,
                "optimizer": optimizer, "lr_g": lr_g, "lr_d": lr_d,
                "gumbel": gumbel, "d_epoches": [],
            }},
        )
        disc, _ = _init_out_checkpoint(disc_name, disc_out_name, device)
        log(f"Copied {gen_name} -> {out_name} and {disc_name} -> {disc_out_name}.")

    gen_paths, disc_paths = model_paths(out_name), model_paths(disc_out_name)
    vocab_sizes = gen.cfg.get("field_vocab_sizes")

    # -- data (same cache + persisted val split conventions as _train_loop) ----
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    store = build_or_open_cache(data_paths, cache_dir=cache_dir, n_workers=cache_workers, log=log)
    all_keys = [k for k in (_path_key(p) for p in data_paths) if store.has(k)]
    if not all_keys:
        raise RuntimeError("No sequences in token cache — run 'Download Dataset' first.")

    val_split_file = gen_paths["val_split"]
    if resume and val_split_file.exists():
        saved_val  = set(json.loads(val_split_file.read_text()))
        val_keys   = [k for k in all_keys if k in saved_val]
        train_keys = [k for k in all_keys if k not in saved_val]
        if not val_keys:
            val_keys, train_keys = _make_split(all_keys)
            val_split_file.write_text(json.dumps(val_keys))
    else:
        val_keys, train_keys = _make_split(all_keys)
        val_split_file.write_text(json.dumps(val_keys))

    max_len = min(gen.cfg["max_seq_len"], disc.cfg["max_seq_len"])
    train_set = MusicDataset(store.subset_view(train_keys), max_len, fim_ratio, fim_example_fn=create_fim_example)
    val_set   = MusicDataset(store.subset_view(val_keys),   max_len, fim_ratio, fim_example_fn=create_fim_example)
    log(f"GAN dataset: {len(train_set)} train / {len(val_set)} val (max_len={max_len}, fim_ratio={fim_ratio})")

    loader     = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader = DataLoader(val_set,   batch_size=batch_size, shuffle=False, drop_last=False)

    if optimizer == "adagrad":
        g_opt = torch.optim.Adagrad(gen.parameters(),  lr=lr_g)
        d_opt = torch.optim.Adagrad(disc.parameters(), lr=lr_d)
    elif optimizer == "adamw":
        g_opt = torch.optim.AdamW(gen.parameters(),  lr=lr_g, weight_decay=0.01)
        d_opt = torch.optim.AdamW(disc.parameters(), lr=lr_d, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}' (use 'adagrad' or 'adamw').")

    gen.to(device)
    disc.to(device)

    total_steps = max(epochs * len(loader), 1)
    global_step = 0

    for epoch in range(1, epochs + 1):
        if stop_flag[0]:
            log("GAN training stopped by user.")
            break

        gen.train()
        disc.train()
        g_loss_sum = d_loss_sum = ce_sum = 0.0
        batches = 0

        for x, y, loss_mask in tqdm(loader, desc=f"{out_name} GAN Epoch {epoch}"):
            if stop_flag[0]:
                break
            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
            pad_mask = y[..., F_TYPE] != PAD_T
            tau = tau_start + (tau_end - tau_start) * (global_step / total_steps)

            # Teacher-forced generator pass (shared by CE anchor and fake build).
            logits = gen(x)
            gen_probs = _field_probs(logits, gumbel, tau)
            real_onehot = _one_hot_rows(y, vocab_sizes)
            fake_probs = _mix_fake(gen_probs, real_onehot, loss_mask)

            # ---- discriminator step (fake detached from the generator) ----
            c_real = disc.critic_score_soft(real_onehot, pad_mask)
            c_fake_d = disc.critic_score_soft([p.detach() for p in fake_probs], pad_mask)
            loss_d = F.softplus(c_fake_d - c_real).mean()

            d_opt.zero_grad()
            loss_d.backward()
            nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            d_opt.step()

            # ---- generator step (gradients flow through D's soft embeddings) ----
            c_real_ref = c_real.detach()
            c_fake_g = disc.critic_score_soft(fake_probs, pad_mask)
            ce = compound_ce_loss(logits, y, loss_mask)
            loss_g = F.softplus(c_real_ref - c_fake_g).mean() + lambda_ce * ce

            g_opt.zero_grad()
            d_opt.zero_grad()  # discard the D grads this backward deposits
            loss_g.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            g_opt.step()

            g_loss_sum += loss_g.item()
            d_loss_sum += loss_d.item()
            ce_sum     += ce.item()
            batches    += 1
            global_step += 1

        avg_g = g_loss_sum / max(batches, 1)
        avg_d = d_loss_sum / max(batches, 1)
        avg_ce = ce_sum / max(batches, 1)

        # ---- validation: teacher-forced CE (comparable to phase-a curves) ----
        gen.eval()
        val_ce, val_batches = 0.0, 0
        with torch.no_grad():
            for x, y, loss_mask in val_loader:
                x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
                val_ce += compound_ce_loss(gen(x), y, loss_mask).item()
                val_batches += 1
        avg_val_ce = val_ce / max(val_batches, 1)

        on_epoch(epoch, avg_g, avg_val_ce)

        # Record discriminator curve in out_name's config (read-modify-write so
        # we never clobber what on_epoch/append_epoch_to_config just wrote).
        try:
            cfg_now = json.loads(gen_paths["config"].read_text())
            cfg_now.setdefault("gan", {}).setdefault("d_epoches", []).append(
                (avg_d, avg_ce))
            gen_paths["config"].write_text(json.dumps(cfg_now, indent=2))
        except Exception as e:
            log(f"Could not record d_epoches: {e}")

        torch.save(gen.state_dict(),  gen_paths["weights"])
        torch.save(g_opt.state_dict(), gen_paths["optimizer"])
        torch.save(disc.state_dict(), disc_paths["weights"])
        torch.save(d_opt.state_dict(), disc_paths["optimizer"])
        if device == "cuda":
            torch.cuda.empty_cache()
        log(f"Epoch {epoch}/{epochs}  G={avg_g:.4f} (CE {avg_ce:.4f})  D={avg_d:.4f}  "
            f"val_CE={avg_val_ce:.4f}  tau={tau:.2f}  ({out_name} checkpoint saved)")

    log(f"{out_name} GAN fine-tuning complete.")
