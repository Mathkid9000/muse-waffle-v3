from model_util import get_model_config, get_field_sizes, create_causal_mask, build_model, append_epoch_to_config, save_model
import torch
from model import MusicTransformer

if __name__ == "__main__":
    # Sanity check: build a small compound model and run one forward pass.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    name = 'test_model'

    field_sizes, vocab_list = get_field_sizes()
    config = get_model_config(field_sizes)
    config["num_layers"] = 2  # keep the smoke test fast
    model, _ = build_model(name, config)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MusicGenerationModel built with {n_params:,} parameters on {device}")

    B, T = 2, 32
    ids = torch.stack(
        [torch.randint(0, v, (B, T)) for v in field_sizes], dim=-1
    ).to(device)
    tgt = torch.stack(
        [torch.randint(0, v, (B, T)) for v in field_sizes], dim=-1
    ).to(device)
    mask = create_causal_mask(T, device).unsqueeze(0)
    outs = model(ids, mask, target_fields=tgt)
    print(f"Teacher-forced forward OK: input {tuple(ids.shape)} -> "
          f"{len(outs)} heads {[tuple(o.shape) for o in outs]}")

    # Step-wise chain sampling path: trunk once, then heads sequentially.
    h_last = model.hidden(ids, mask)[:, -1]
    prev = []
    for i in range(len(field_sizes)):
        logits = model.field_logits(i, h_last, prev)
        assert logits.shape == (B, field_sizes[i]), logits.shape
        prev.append(logits.argmax(dim=-1))
    print(f"Chain sampling path OK: sampled row {[int(p[0]) for p in prev]}")

    append_epoch_to_config(name, config, 0, 0)
    append_epoch_to_config(name, config, 0.1, 0.2)
    append_epoch_to_config(name, config, 0.4, 0.4)

    save_model(name, config, model, None, None, [])
