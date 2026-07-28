from helpers import *
from model import MusicTransformer
from tokenizer import load_vocab_list


def model_paths(name: str) -> dict[str, Path]:
    model_dir = CHECKPOINT_DIR / name
    return {
        "dir": model_dir,
        "config": model_dir / "config.json",
        "weights": model_dir / "model.pt",
        "optimizer": model_dir / "optimizer.pt",
        "scheduler": model_dir / "scheduler.pt",
        "val_split": model_dir / "val_paths.json",
    }


def append_epoch_to_config(name: str, cfg: dict, training_loss, val_loss):
    if cfg is None:
        return

    if 'epoches' not in cfg:
        cfg['epoches'] = [(training_loss, val_loss)]
    else:
        cfg['epoches'].append((training_loss, val_loss))

    with open(model_paths(name)["config"], "w") as f:
        json.dump(cfg, f, indent=2)


def load_model_only(
    name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[MusicTransformer, dict]:
    paths = model_paths(name)
    config_path = paths["config"]
    
    if not config_path.exists():
        raise FileNotFoundError(f"No config for model '{name}'. Run 'Build Model' first.")
    
    with open(config_path) as f:
        cfg = json.load(f)

    model = MusicTransformer(**cfg).to(device)
    weights = paths["weights"]
    if weights.exists():
        state = torch.load(weights, map_location=device, weights_only=False)
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"No weights for model '{name}'. Train the model first.")
    
    return model, cfg


def load_config_only(
    name: str
) -> dict:
    paths = model_paths(name)
    config_path = paths["config"]
    if not config_path.exists():
        raise FileNotFoundError(f"No config for model '{name}'. Run 'Build Model' first.")
    with open(config_path) as f:
        return json.load(f)


def load_val_split_only(
    name: str
) -> list:
    paths = model_paths(name)
    val_spilt_path = paths["val_split"]
    if not val_spilt_path.exists():
        raise FileNotFoundError(f"No val split path for model '{name}'. Run 'Build Model' first.")
    with open(val_spilt_path) as f:
        return json.load(f)


def save_val_split_only(
    name: str,
    val_splits: list
) -> list:
    paths = model_paths(name)
    val_spilt_path = paths["val_split"]
    with open(val_spilt_path, 'w') as f:
        return json.dump(val_splits, f)


def load_full_model(
    name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    # map_location is required: these are saved from whatever device trained
    # them, so a CUDA-trained checkpoint would otherwise fail to load on a
    # CPU-only box (and would rehydrate the optimizer's exp_avg/exp_avg_sq
    # onto the wrong device even where CUDA exists but the run is on CPU).
    # All three are plain state dicts, so weights_only=True is safe.
    paths = model_paths(name)
    with open(paths['config']) as f:
        cfg = json.load(f)
    model = torch.load(paths['weights'], map_location=device, weights_only=True)
    optimizer = torch.load(paths['optimizer'], map_location=device, weights_only=True)
    scheduler = torch.load(paths['scheduler'], map_location=device, weights_only=True)
    with open(paths['val_split']) as f:
        val_split = json.load(f)

    return model, cfg, scheduler, optimizer, val_split


def build_model(name: str, cfg: dict | None = None):
    """Build a model with expanded vocabulary."""
    paths = model_paths(name)
    if not cfg and paths["config"].exists():
        cfg = load_config_only(name)
    elif not cfg:
        cfg = get_model_config(get_field_sizes()[0])

    paths["dir"].mkdir(parents=True, exist_ok=True)
    # with open(paths["config"], "w") as f:
    #     json.dump(cfg, f, indent=2)

    return MusicTransformer(**cfg), cfg


def save_model(name, cfg, model, optimizer, scheduler, val_split):
    paths = model_paths(name)
    with open(paths['config'], 'w') as f:
        json.dump(cfg, f)
    with open(paths['val_split'], 'w') as f:
        json.dump(val_split, f)
    torch.save(model.state_dict(), paths['weights'])
    if optimizer: torch.save(optimizer.state_dict(), paths['optimizer'])
    if scheduler: torch.save(scheduler.state_dict(), paths['scheduler'])


def get_field_sizes():
    vocab_list = load_vocab_list()
    return [ len(h) for h in vocab_list ], vocab_list


def get_model_config(field_sizes):
    field_sizes = list(field_sizes)
    # ~47M params. The corpus is ~19M unique events, so capacity is not the
    # bottleneck and never was; the win from a smaller trunk is the ~2x cheaper
    # step, which buys more passes over the data in the same T4-hours.
    return {
        "num_layers": 10,
        "num_heads": 8,
        "num_kv_heads": 4,
        "hidden_dim": 640,
        "intermediate_dim": 1728,   # 8/3 x hidden, rounded to a multiple of 64
        "max_seq_len": SEQ_LEN,
        "field_sizes": field_sizes,
        "embed_dims": [160, 128, 96, 96, 48],
        "dropout": 0.1,
        "epoches": []
    }


def create_causal_mask(seq_len, device):
    """Boolean lower-triangular mask (True = attend) for scaled_dot_product_attention."""
    return torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    )
