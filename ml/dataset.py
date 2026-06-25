"""Backwards-compatible facade over the multi-source dataset registry.

Historically this module *was* the OpenScore downloader.  That logic now lives in
``ml.datasets`` (one module per source) behind the :class:`DatasetSource`
interface, so new corpora (PDMX, Wikifonia, ...) can be added without touching
callers.  These thin wrappers preserve the original import surface
(``download_dataset``, ``load_cached_paths``, ``dataset_status``) used by the
notebooks, ``similarity.py`` and the round-trip tests.

For multi-source workflows use ``ml.datasets`` directly:

    from ml.datasets import download_sources, load_paths, dataset_status
    download_sources(["wikifonia"])
    paths = load_paths(["openscore", "pdmx"])   # or load_paths() for all
"""

from __future__ import annotations

from pathlib import Path

from ml.datasets import load_paths
from ml.datasets import dataset_status as _registry_status
from ml.datasets.registry import SOURCES


def download_dataset(log, progress, max_files: int = 200) -> int:
    """Download the OpenScore corpus (legacy default source)."""
    return SOURCES["openscore"].download(log, progress, max_files=max_files)


def load_cached_paths() -> list[Path]:
    """All cached score paths, merged across every downloaded source."""
    return load_paths()


def dataset_status() -> str:
    return _registry_status()
