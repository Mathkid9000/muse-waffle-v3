"""Dataset source abstraction.

A `DatasetSource` knows how to acquire a corpus of readable scores (MusicXML/MXL)
and record where they live, plus any per-score metadata (rating, genre, ...).

Each source owns a subdirectory under ``STORAGE_DIR/data/<name>`` and writes a
``manifest.json`` there.  A manifest is a list of records:

    {"path": <str>, "source": <name>, "meta": {...}}

``path`` is either relative to ``STORAGE_DIR`` (for files we download into storage)
or an absolute path (for large corpora we reference in place, e.g. PDMX).
``resolve_path`` handles both.  The registry merges every source's manifest into
one unified path list for training.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv(dotenv_path="env.env")
STORAGE_DIR = Path(os.getenv("STORAGE_DIR"))


def resolve_path(path: str | Path) -> Path:
    """Resolve a manifest path to an absolute path.

    Absolute paths are returned as-is; relative paths are joined onto STORAGE_DIR
    (matching the convention used by the original OpenScore loader).
    """
    p = Path(path)
    return p if p.is_absolute() else STORAGE_DIR / p


class DatasetSource(ABC):
    """Base class for a downloadable/locatable corpus of scores."""

    #: short identifier, also the storage subdirectory name (e.g. "openscore")
    name: str = ""

    @property
    def data_dir(self) -> Path:
        return STORAGE_DIR / "data" / self.name

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @abstractmethod
    def download(
        self,
        log: Callable[[str], None],
        progress: Callable[[int, int], None],
        **opts,
    ) -> int:
        """Acquire/locate scores and write the manifest. Returns file count."""

    # -- manifest helpers -------------------------------------------------

    def _write_manifest(self, records: list[dict]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(records, f, indent=2)

    def manifest_records(self) -> list[dict]:
        """Return this source's manifest records (empty if not downloaded)."""
        if not self.manifest_path.exists():
            return []
        with open(self.manifest_path) as f:
            raw = json.load(f)
        # Tolerate the legacy OpenScore manifest format: a flat list of path strings.
        records = []
        for item in raw:
            if isinstance(item, str):
                records.append({"path": item, "source": self.name, "meta": {}})
            else:
                item.setdefault("source", self.name)
                item.setdefault("meta", {})
                records.append(item)
        return records

    def is_available(self) -> bool:
        return self.manifest_path.exists()

    def status(self) -> str:
        if not self.manifest_path.exists():
            return f"{self.name}: not downloaded"
        return f"{self.name}: {len(self.manifest_records())} files cached"
