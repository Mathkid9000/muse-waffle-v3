"""Registry that ties dataset sources together and merges their manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ml.datasets.base import DatasetSource, resolve_path
from ml.datasets.openscore import OpenScoreSource
from ml.datasets.pdmx import PDMXSource
from ml.datasets.wikifonia import WikifoniaSource

# Instantiate once; sources are stateless aside from their storage dirs.
SOURCES: dict[str, DatasetSource] = {
    s.name: s for s in (OpenScoreSource(), PDMXSource(), WikifoniaSource())
}


def _select(names: list[str] | None) -> list[DatasetSource]:
    if names is None:
        return list(SOURCES.values())
    missing = [n for n in names if n not in SOURCES]
    if missing:
        raise KeyError(f"Unknown dataset source(s): {missing}. Known: {list(SOURCES)}")
    return [SOURCES[n] for n in names]


def download_sources(
    names: list[str],
    log: Callable[[str], None] = print,
    progress: Callable[[int, int], None] = lambda i, n: None,
    **opts,
) -> dict[str, int]:
    """Download/locate the named sources. Per-source kwargs go through **opts."""
    counts = {}
    for src in _select(names):
        log(f"=== {src.name} ===")
        counts[src.name] = src.download(log, progress, **opts)
    return counts


def load_records(names: list[str] | None = None) -> list[dict]:
    """Merged manifest records across the selected (default: all) sources."""
    records: list[dict] = []
    for src in _select(names):
        records.extend(src.manifest_records())
    return records


def load_paths(names: list[str] | None = None) -> list[Path]:
    """Merged, resolved score paths across the selected (default: all) sources."""
    return [resolve_path(r["path"]) for r in load_records(names)]


def dataset_status(names: list[str] | None = None) -> str:
    return "\n".join(src.status() for src in _select(names))
