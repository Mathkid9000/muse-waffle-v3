import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import random, math, os

from model import MusicTransformer
from model_util import load_val_split_only, build_model, load_full_model, save_model, append_epoch_to_config
from helpers import BATCH_SIZE, VAL_SPLIT, SEQ_LEN
from tokenizer import QuintupleTokenizer, load_dataset
from dataset import QuintupleDataset, make_dataloader


def compound_loss(logits, y, pad_ids, weights):
    """Sum of per-field cross-entropies, ignoring padding rows and weighting
    each target position by ``weights`` (float (B,T).
    ``logits`` is a list of (B,T,V_i); ``y`` is (B,T,5).

    Padding is identified by field 0 == PAD (a whole padding ROW), NOT by a
    per-field ``== pad_ids[i]`` test. In fields 1-4 the "pad" id IS the empty
    string, and '' is a legitimate value: every structural row (SOG, SOS, EOS,
    INSTR) carries '' there. Excluding '' per-field would drop all of those
    targets from the loss, so the field-aware heads would get zero signal to
    emit '' when field 0 is a struct token and would instead produce
    note/chord-like garbage in those fields. Keying off the field-0 PAD row
    keeps genuine padding excluded while supervising ''.

    ``pad_ids[0]`` is currently None: field 0 has no '' entry in the vocab (it
    is always M_*/SOG/SOS/EOS/INSTR_*) and build_vocab_list adds no PAD
    special, so nothing is masked out. That is correct for the present data
    path -- QuintupleDataset concatenates every sequence into one dense stream
    and windows it, so no padding rows exist. The None case MUST be handled
    explicitly rather than left to fall out of the comparison: `tensor == None`
    evaluates to the Python bool False (not an elementwise mask), and `~False`
    is the int -1, which silently negates the whole loss.
    """
    if pad_ids[0] is None:
        row_valid = torch.ones_like(y[..., 0], dtype=torch.float32)
    else:
        row_valid = (y[..., 0] != pad_ids[0]).float()

    total = 0.0
    valid = weights * row_valid                 # loop-invariant
    denom = valid.sum().clamp(min=1)
    for i, logit in enumerate(logits):
        ce = F.cross_entropy(
            logit.reshape(-1, logit.shape[-1]),
            y[..., i].reshape(-1),
            reduction="none",
        ).view(y.shape[0], y.shape[1])
        total = total + (ce * valid).sum() / denom
    return total / len(logits)


@torch.no_grad()
def evaluate_val(model, val_loader, pad_ids, device, max_batches=100):
    """Mean compound loss on the validation windows.
    Restores train mode before returning."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x, target_fields=y)
        weights = torch.ones_like(y[..., 0], dtype=torch.float32)
        total += compound_loss(logits, y, pad_ids, weights).item()
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = 'TEST123'
    MICRO_BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = max(1, BATCH_SIZE // MICRO_BATCH_SIZE)
    MAX_STEPS_PER_EPOCH = 15000   # stride-1 windows make full epochs enormous

    tokenized_events = load_dataset()
    tokenizer = QuintupleTokenizer()
    pad_ids = tokenizer.pad_ids

    # split dataset
    try:
        val_split_indices = load_val_split_only(MODEL_NAME)
    except:
        # shuffle
        indexed = list(range(len(tokenized_events)))
        random.shuffle(indexed)
        val_split_indices = indexed[:math.ceil(len(indexed)*VAL_SPLIT)]
    train_events = [ tokenized_events[i] for i in range(len(tokenized_events)) if i not in val_split_indices ]
    val_events = [ tokenized_events[i] for i in val_split_indices ]

    # # The vocab is built from the train split only; a few valid chorales are
    # # longer than any train chorale (unseen BEAT_x) or hold rare durations.
    # # Drop those sequences rather than crash or corrupt them (~9% of valid).
    # n_val = len(val_events)
    # val_events = [e for e in val_events if tokenizer.covers(e)]
    # if len(val_events) < n_val:
    #     print(f"Dropped {n_val - len(val_events)}/{n_val} val sequences "
    #           f"with out-of-vocab tokens")

    val_dataset = QuintupleDataset(val_events, seq_len=SEQ_LEN)
    val_subset = torch.utils.data.Subset(
        val_dataset, range(0, len(val_dataset), SEQ_LEN))
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=MICRO_BATCH_SIZE, shuffle=False)

    # model_config = get_model_config(tokenizer.field_sizes)
    # model = MusicGenerationModel(**model_config).to(device)
    model, config = build_model(MODEL_NAME)
    model = model.to(device)

    # Mixed precision: ~halves activation memory and speeds up matmuls. T4/Turing
    # has no bf16, so use fp16 + GradScaler when on CUDA.
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training hyperparameters
    N_EPOCHS = 200
    TOTAL_OPT_STEPS = 60000       # the step budget the run is actually given
    LR = 0.0005
    WARMUP_STEPS = 200
    CLIP_NORM = 6.0
    PATIENCE = 60                 # early-stop epochs without val improvement

    # Decay only matmul weights + embeddings; 1-D params (biases, RMSNorm
    # gains) are scale parameters and decaying them just fights the norms.
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}], lr=LR)

    dataloader = make_dataloader(train_events, seq_len=SEQ_LEN, batch_size=MICRO_BATCH_SIZE)

    # LR schedule is stepped once per *optimizer* step (i.e. per accumulation
    # window), so count total steps in those units.
    opt_steps_per_epoch = max(
        1, min(len(dataloader), MAX_STEPS_PER_EPOCH) // GRAD_ACCUM_STEPS)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    # Anneal over TOTAL_OPT_STEPS, NOT N_EPOCHS * opt_steps_per_epoch: the latter is
    # ~1.5M steps (~200 T4-hours), a horizon no run reaches, so cosine stays in its
    # flat opening and the LR never leaves its peak. The budget is decoupled from
    # N_EPOCHS on purpose; N_EPOCHS is only the outer cap.
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, TOTAL_OPT_STEPS - WARMUP_STEPS), eta_min=LR * 0.05)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS])

    # Resume only if a checkpoint exists. NOTE: checkpoints from before the
    # intra-row-chain heads have incompatible head shapes — delete
    # checkpoint.pth (and chorale_model.pth) and re-pretrain from scratch.
    try:
        loaded_model, loaded_config, loaded_scheduler, loaded_optimizer, _  = load_full_model(MODEL_NAME)
        assert loaded_model != None and loaded_scheduler != None and loaded_optimizer != None
        model.load_state_dict(loaded_model)
        config = loaded_config.copy()
        scheduler.load_state_dict(loaded_scheduler)
        optimizer.load_state_dict(loaded_optimizer)
        print(f'Model "{MODEL_NAME}" loaded successfully with {len(config["epoches"])} epoches found.')
    except:
        print(f'Found no complete model with name {MODEL_NAME}. Training will continue with fresh states.')
    print(f'Running with config: {config}')
    
    print(f"Training for up to {N_EPOCHS} epochs with "
          f"{min(len(dataloader), MAX_STEPS_PER_EPOCH)} steps/epoch on {device}")
    print(f"Field vocab sizes: {tokenizer.field_sizes}")
    best_val = float("inf")
    stale_epochs = 0

    for epoch in range(N_EPOCHS):
        # past T_max a CosineAnnealingLR turns back up rather than staying at eta_min,
        # so the budget has to stop the loop as well as shape the curve
        if scheduler.last_epoch >= TOTAL_OPT_STEPS:
            print(f"Reached the {TOTAL_OPT_STEPS}-step budget; stopping.")
            break

        steps_this_epoch = min(len(dataloader), MAX_STEPS_PER_EPOCH)
        model.train()
        epoch_loss = 0
        steps_run = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}", total=steps_this_epoch)
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(progress, start=1):
            if step > steps_this_epoch:
                break
            steps_run = step
            x = x.to(device)          # (B, T, 5)
            y = y.to(device)          # (B, T, 5)
            weights = torch.ones_like(y[..., 0], dtype=torch.float32)

            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                # Teacher forcing for the intra-row chain: head i conditions on
                # the target row's fields < i (see MusicGenerationModel).
                logits = model(x, target_fields=y)
                loss = compound_loss(logits, y, pad_ids, weights)

            # Normalize so accumulated grads match a full-batch average.
            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
            epoch_loss += loss.item()

            if step % GRAD_ACCUM_STEPS == 0 or step == steps_this_epoch:
                scaler.unscale_(optimizer)
                # AMP may produce non-finite grads on some steps; the scaler
                # detects and skips those, so don't hard-error on them here.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CLIP_NORM, error_if_nonfinite=False
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{epoch_loss / step:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_loss = epoch_loss / max(steps_run, 1)
        val_loss = evaluate_val(model, val_loader, pad_ids, device)
        print(f"Epoch {epoch+1}/{N_EPOCHS}; Avg loss: {avg_loss:.4f}; "
              f"Val loss: {val_loss:.4f}")

        append_epoch_to_config(MODEL_NAME, config, avg_loss, val_loss)
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            save_model(MODEL_NAME, config, model, optimizer, scheduler, val_split_indices)
            print(f"Saved checkpoint (val {best_val:.4f})")
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"Early stop: no val improvement in {PATIENCE} epochs "
                      f"(best {best_val:.4f})")
                break


if __name__ == "__main__":
    main()
