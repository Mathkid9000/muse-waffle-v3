"""Parse-once token cache for scalable training.

The training loop used to re-parse every score with music21 at the start of each
epoch (fine for ~1.5K files, untenable for tens of thousands).  This module parses
each score **once** and serialises the exact per-score record the trainer consumes
— ``(header, collected, n_parts)`` — into gzipped pickle shards, then streams them
back lazily at train time with no music21 in the loop.

``collected`` is the output of :func:`ml.tokenizer._collect_events_by_measure`
(events plus global-symbol dicts) and is plain Python (ints/lists/dicts), so it
pickles cleanly and still lets partial-staff FIM re-emit arbitrary staff subsets.
``measure_tokens`` is re-derived on access via ``_emit_measure_tokens`` (cheap).

Cache layout (under ``STORAGE_DIR/cache/tokens`` by default):
    index.json              schema + vocab fingerprint + shard list + key order
    shard_00000.pkl.gz ...  each a pickled list of record dicts

The index carries a ``vocab_fingerprint`` (hash of the tokenizer layout).  If the
tokenizer vocabulary changes (e.g. chord-symbol tokens are added) the fingerprint
no longer matches and the cache is rebuilt automatically.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

from ml.datasets.base import STORAGE_DIR

SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = STORAGE_DIR / "cache" / "tokens"
DEFAULT_SHARD_SIZE = 1000


def vocab_fingerprint() -> str:
    """Hash of the tokenizer's vocabulary layout; changes invalidate the cache."""
    import ml.tokenizer as tk

    layout = (
        tk.VOCAB_SIZE, tk.NOTE_OFF, tk.DUR_OFF, tk.REST_OFF, tk.KEY_OFF,
        tk.TIME_OFF, tk.CLEF_OFF, tk.TEXT_OFF, tk.PART_OFF, tk.STEP_OFF,
        tk.GRACE, tk.TEMPO_OFF, tk.INSTR_OFF, tk.FILL_GEN,
        len(tk.DURATIONS), tk.MAX_STEP, tk.MAX_PARTS,
    )
    return hashlib.sha1(repr(layout).encode()).hexdigest()[:16]


def _abs_path(path) -> Path:
    """Canonical absolute path for a score, robust to either input form.

    Callers pass paths in two shapes: raw manifest paths relative to STORAGE_DIR
    (``data/openscore/x.mxl``) and already-resolved paths from ``load_paths``
    (``storage/data/openscore/x.mxl``, relative to CWD).  Both must map to the
    same absolute file so a score has one cache key regardless of how it arrived.
    """
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (STORAGE_DIR / p).resolve()


def _path_key(path) -> str:
    return str(_abs_path(path))


# -- parsing (module-level so it is picklable for multiprocessing) -----------

def _parse_one(item: dict) -> dict | None:
    """Parse one score into a cache record, or None if it fails / is too short.

    `item` is a manifest-style dict: {"path", "source", "meta"}.
    """
    import music21 as m21
    from ml.tokenizer import (
        score_to_header_tokens, _collect_events_by_measure, MAX_REAL_PARTS,
    )

    path = _abs_path(item["path"])
    try:
        score = m21.converter.parse(str(path))
        header = score_to_header_tokens(score)
        collected = _collect_events_by_measure(score)
    except Exception:
        return None

    events_by_measure = collected[0]
    if not events_by_measure or len(events_by_measure) < 3:
        return None

    n_parts = len(list(score.parts)[:MAX_REAL_PARTS])
    return {
        "key": _path_key(item["path"]),
        "source": item.get("source", ""),
        "meta": item.get("meta", {}),
        "header": header,
        "collected": collected,
        "n_parts": n_parts,
    }


# -- shard io ----------------------------------------------------------------

def _write_shard(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_shard(path: Path) -> list[dict]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _normalise_records(items) -> list[dict]:
    """Accept manifest records, plain paths, or (path,...) and normalise to dicts."""
    out = []
    for it in items:
        if isinstance(it, dict):
            out.append({"path": it["path"], "source": it.get("source", ""), "meta": it.get("meta", {})})
        else:
            out.append({"path": str(it), "source": "", "meta": {}})
    return out


# -- build -------------------------------------------------------------------

def build_token_cache(
    items: Iterable,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    n_workers: int = 1,
    force: bool = False,
    shard_size: int = DEFAULT_SHARD_SIZE,
    log: Callable[[str], None] = print,
) -> "CachedMeasureStore":
    """Parse `items` into the token cache (incremental & resumable).

    Args:
        items: manifest records ({"path","source","meta"}) or plain paths.
        cache_dir: where shards + index.json live.
        n_workers: parse processes (>1 uses multiprocessing; guard the caller in
            ``if __name__ == '__main__'`` on Windows). Default 1 (serial).
        force: rebuild from scratch, ignoring any existing cache.
        shard_size: records per shard.

    Returns an open :class:`CachedMeasureStore`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.json"
    fp = vocab_fingerprint()

    # Load existing index unless it's stale or force is set.
    index = {"schema_version": SCHEMA_VERSION, "vocab_fingerprint": fp,
             "shards": [], "order": []}
    if index_path.exists() and not force:
        existing = json.loads(index_path.read_text())
        if existing.get("vocab_fingerprint") == fp and existing.get("schema_version") == SCHEMA_VERSION:
            index = existing
        else:
            log("Tokenizer vocabulary changed since cache was built — rebuilding from scratch.")
            for sh in existing.get("shards", []):
                (cache_dir / sh["file"]).unlink(missing_ok=True)

    records = _normalise_records(items)
    cached_keys = set(index["order"])
    todo = [r for r in records if _path_key(r["path"]) not in cached_keys]
    log(f"Token cache: {len(cached_keys)} cached, {len(todo)} new to parse "
        f"(workers={n_workers}).")

    if not todo:
        index_path.write_text(json.dumps(index))
        return CachedMeasureStore(cache_dir)

    # Parse (serial or parallel), buffering into shards as we go.
    next_shard_idx = len(index["shards"])
    buffer: list[dict] = []
    n_failed = 0

    def flush():
        nonlocal next_shard_idx, buffer
        if not buffer:
            return
        fname = f"shard_{next_shard_idx:05d}.pkl.gz"
        _write_shard(cache_dir / fname, buffer)
        index["shards"].append({"file": fname, "count": len(buffer)})
        index["order"].extend(r["key"] for r in buffer)
        next_shard_idx += 1
        buffer = []
        index_path.write_text(json.dumps(index))  # checkpoint progress

    def handle(rec):
        nonlocal n_failed, buffer
        if rec is None:
            n_failed += 1
            return
        buffer.append(rec)
        if len(buffer) >= shard_size:
            flush()

    if n_workers and n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            for rec in tqdm(pool.imap_unordered(_parse_one, todo, chunksize=8),
                            total=len(todo), desc="Parsing scores"):
                handle(rec)
    else:
        for item in tqdm(todo, desc="Parsing scores"):
            handle(_parse_one(item))

    flush()
    log(f"Token cache built: {len(index['order'])} records across "
        f"{len(index['shards'])} shards ({n_failed} skipped). -> {cache_dir}")
    return CachedMeasureStore(cache_dir)


# -- read / stream -----------------------------------------------------------

class CachedMeasureStore:
    """Lazy, map-style access to a token cache.

    ``store[i]`` returns ``(header, measure_tokens, collected, n_parts)`` —
    identical to what the old in-memory ``_load_measure_sequences`` produced —
    with ``measure_tokens`` re-emitted on the fly.  Shards are loaded on demand
    and kept in a small LRU cache.
    """

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, lru_shards: int = 16):
        self.cache_dir = Path(cache_dir)
        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No token cache at {self.cache_dir} (run build_token_cache).")
        self.index = json.loads(index_path.read_text())
        if self.index.get("vocab_fingerprint") != vocab_fingerprint():
            raise RuntimeError(
                f"Token cache at {self.cache_dir} is stale (tokenizer vocabulary "
                "changed). Rebuild with build_token_cache(..., force=True)."
            )
        # Flat position -> (shard_idx, offset) and key -> position maps.
        self._locations: list[tuple[int, int]] = []
        for shard_idx, sh in enumerate(self.index["shards"]):
            for off in range(sh["count"]):
                self._locations.append((shard_idx, off))
        self.order: list[str] = self.index["order"]
        self._key_to_pos = {k: i for i, k in enumerate(self.order)}
        self._lru: "OrderedDict[int, list[dict]]" = OrderedDict()
        self._lru_max = lru_shards

    def __len__(self) -> int:
        return len(self._locations)

    def keys(self) -> list[str]:
        return self.order

    def has(self, key) -> bool:
        return _path_key(key) in self._key_to_pos

    def _load_shard(self, shard_idx: int) -> list[dict]:
        if shard_idx in self._lru:
            self._lru.move_to_end(shard_idx)
            return self._lru[shard_idx]
        recs = _read_shard(self.cache_dir / self.index["shards"][shard_idx]["file"])
        self._lru[shard_idx] = recs
        if len(self._lru) > self._lru_max:
            self._lru.popitem(last=False)
        return recs

    def _record(self, pos: int) -> dict:
        shard_idx, off = self._locations[pos]
        return self._load_shard(shard_idx)[off]

    def _to_tuple(self, rec: dict):
        from ml.tokenizer import _emit_measure_tokens
        measure_tokens = _emit_measure_tokens(*rec["collected"])
        return rec["header"], measure_tokens, rec["collected"], rec["n_parts"]

    def __getitem__(self, pos: int):
        return self._to_tuple(self._record(pos))

    def get_by_key(self, key):
        return self[self._key_to_pos[_path_key(key)]]

    def subset_view(self, keys) -> "SubsetView":
        """A map-style view over the positions for `keys` (missing keys dropped)."""
        positions = [self._key_to_pos[k] for k in (_path_key(x) for x in keys)
                     if k in self._key_to_pos]
        return SubsetView(self, positions)


class SubsetView:
    """Map-style view exposing a subset (e.g. a train/val split) of a store."""

    def __init__(self, store: CachedMeasureStore, positions: list[int]):
        self.store = store
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        return self.store[self.positions[i]]


def build_or_open_cache(
    items: Iterable,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    n_workers: int = 1,
    log: Callable[[str], None] = print,
) -> CachedMeasureStore:
    """Build the cache for `items` (incremental) and return an open store."""
    return build_token_cache(items, cache_dir=cache_dir, n_workers=n_workers, log=log)
