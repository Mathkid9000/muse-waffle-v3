"""Wikifonia / EWLD source — lead sheets (melody + chord symbols).

Wikifonia's original public service shut down in 2013, and the full Enhanced
Wikifonia Leadsheet Dataset (EWLD) on Zenodo is restricted-access (manual
approval, non-commercial), so neither can be fetched automatically.  By default
this source therefore downloads **OpenEWLD** — the freely redistributable
public-domain subset (~500 lead sheets in MusicXML, with chord symbols) hosted on
GitHub (00sapo/OpenEWLD) — so a headless/remote notebook works with no manual step.

Override the default to use a larger corpus you have rights to:
  * ``wikifonia_root=<dir>``   enumerate an already-extracted directory of scores
  * ``archive_url=<zip/tar>``  download + extract a specific archive instead

Lead sheets carry chord symbols (Cmaj7, Dm7/G, ...), captured by the chord-symbol
tokens in the tokenizer.
"""

from __future__ import annotations

import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import requests
from tqdm import tqdm

from ml.datasets.base import DatasetSource

SCORE_EXTS = (".mxl", ".xml", ".musicxml")

OPENEWLD_REPO = "00sapo/OpenEWLD"
DEFAULT_ARCHIVE_URL = f"https://github.com/{OPENEWLD_REPO}/archive/refs/heads/master.tar.gz"


class WikifoniaSource(DatasetSource):
    name = "wikifonia"

    def _has_scores(self, root: Path) -> bool:
        return root.exists() and any(
            p.suffix.lower() in SCORE_EXTS for p in root.rglob("*") if p.is_file()
        )

    def _download_archive(self, url: str, log: Callable[[str], None]) -> None:
        """Stream an archive to disk and extract it into the data dir."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        log(f"Downloading lead-sheet archive: {url}")
        suffix = ".zip" if url.endswith(".zip") else ".tar.gz"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=self.data_dir) as tmp:
            tmp_path = Path(tmp.name)
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0))
                with tqdm(total=total, unit="B", unit_scale=True, desc="archive") as bar:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        tmp.write(chunk)
                        bar.update(len(chunk))
        # Extract member-by-member so a few entries with OS-illegal names (e.g. '?'
        # or ':' on Windows) are skipped instead of aborting the whole extraction.
        skipped = 0
        try:
            if zipfile.is_zipfile(tmp_path):
                with zipfile.ZipFile(tmp_path) as z:
                    for info in z.infolist():
                        try:
                            z.extract(info, self.data_dir)
                        except (OSError, ValueError):
                            skipped += 1
            else:
                with tarfile.open(tmp_path) as t:
                    for member in t.getmembers():
                        try:
                            t.extract(member, self.data_dir, filter="data")
                        except (OSError, ValueError):
                            skipped += 1
        finally:
            tmp_path.unlink(missing_ok=True)
        extra = f" ({skipped} entries skipped: invalid filename on this OS)" if skipped else ""
        log(f"Extracted archive into {self.data_dir}/{extra}")

    def download(
        self,
        log: Callable[[str], None],
        progress: Callable[[int, int], None],
        wikifonia_root: str | None = None,
        archive_url: str | None = None,
        download: bool = True,
        **opts,
    ) -> int:
        root = Path(wikifonia_root) if wikifonia_root else self.data_dir

        # Fetch scores if we don't already have any (idempotent across re-runs).
        if not self._has_scores(root):
            url = archive_url or (DEFAULT_ARCHIVE_URL if download else None)
            if url:
                self._download_archive(url, log)
                root = self.data_dir
            else:
                raise FileNotFoundError(
                    f"No lead sheets under {root} and download=False. Provide "
                    "wikifonia_root=<dir>, archive_url=<zip/tar>, or set download=True."
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
