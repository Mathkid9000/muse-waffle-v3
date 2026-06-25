"""Wikifonia / EWLD source — lead sheets (melody + chord symbols).

Wikifonia's original public service shut down in 2013; the ~5K-leadsheet corpus
(and the Enhanced Wikifonia Leadsheet Dataset, EWLD) circulates as archives of
MusicXML files.  This source enumerates a directory of ``.mxl``/``.xml`` lead
sheets that the user has obtained, optionally downloading + extracting an archive
from a configurable URL first.  Files are referenced in place.

Lead sheets carry chord symbols (Cmaj7, Dm7/G, ...); capturing those requires the
chord-symbol tokens added in the tokenizer (Phase 3).  Until then the melody line
is still useful training data.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path
from typing import Callable

import requests

from ml.datasets.base import DatasetSource

SCORE_EXTS = (".mxl", ".xml", ".musicxml")


class WikifoniaSource(DatasetSource):
    name = "wikifonia"

    def _download_archive(self, url: str, log: Callable[[str], None]) -> None:
        log(f"Downloading Wikifonia archive: {url}")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        content = resp.content
        if url.endswith(".zip") or zipfile.is_zipfile(io.BytesIO(content)):
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                z.extractall(self.data_dir)
        else:
            with tarfile.open(fileobj=io.BytesIO(content)) as t:
                t.extractall(self.data_dir)
        log(f"Extracted archive into {self.data_dir}/")

    def download(
        self,
        log: Callable[[str], None],
        progress: Callable[[int, int], None],
        wikifonia_root: str | None = None,
        archive_url: str | None = None,
        **opts,
    ) -> int:
        root = Path(wikifonia_root) if wikifonia_root else self.data_dir
        if not root.exists() and archive_url:
            self._download_archive(archive_url, log)
            root = self.data_dir
        if not root.exists():
            raise FileNotFoundError(
                f"Wikifonia root not found at {root}. Obtain the Wikifonia/EWLD "
                "MusicXML archive and place it there, or pass archive_url=... / "
                "wikifonia_root=<dir>."
            )

        files = sorted(
            p for p in root.rglob("*")
            if p.suffix.lower() in SCORE_EXTS and p.is_file()
        )
        records = []
        for i, p in enumerate(files):
            records.append({"path": str(p.resolve()), "source": self.name, "meta": {}})
            progress(i + 1, len(files))

        self._write_manifest(records)
        log(f"Wikifonia cached: {len(records)} lead sheets (manifest in {self.data_dir}/)")
        return len(records)
