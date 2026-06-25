"""OpenScore/Lieder source — downloads MusicXML scores via the GitHub tree API.

This is the original ``ml/dataset.py`` logic, relocated behind the DatasetSource
interface.  Behaviour is unchanged: it pulls every .mxl/.xml from the repo tree,
flattens paths, and caches into ``STORAGE_DIR/data/openscore``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import requests
from tqdm import tqdm

from ml.datasets.base import DatasetSource

REPO = "OpenScore/Lieder"
BRANCH = "main"
API_TREE = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"

HEADERS = {"Accept": "application/vnd.github+json"}


def _auth_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return {**HEADERS, "Authorization": f"Bearer {token}"}
    return HEADERS


class OpenScoreSource(DatasetSource):
    name = "openscore"

    def _rel(self, github_path: str) -> str:
        """Manifest path (relative to STORAGE_DIR) for a flattened score file."""
        return str(Path("data/openscore") / github_path.replace("/", "_"))

    def fetch_file_list(self, log: Callable[[str], None]) -> list[dict]:
        log("Fetching file tree from GitHub API...")
        resp = requests.get(API_TREE, headers=_auth_headers(), timeout=30)
        if resp.status_code == 403:
            raise RuntimeError(
                "GitHub API rate-limited. Set GITHUB_TOKEN env var to increase limits."
            )
        resp.raise_for_status()
        tree = resp.json().get("tree", [])
        files = [
            {"path": item["path"], "url": f"{RAW_BASE}/{item['path']}"}
            for item in tree
            if item["path"].endswith(".mxl") or item["path"].endswith(".xml")
        ]
        log(f"Found {len(files)} score files.")
        return files

    def download(
        self,
        log: Callable[[str], None],
        progress: Callable[[int, int], None],
        max_files: int = 200,
        **opts,
    ) -> int:
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Reuse the existing cache when the requested count already matches.
        existing = self.manifest_records()
        if existing and len(existing) == max_files:
            log(f"Cache found: {len(existing)} files already downloaded.")
            return len(existing)

        files = self.fetch_file_list(log)[:max_files]
        records: list[dict] = []

        for i, item in tqdm(enumerate(files), total=len(files), desc="Downloading"):
            dest = self.data_dir / item["path"].replace("/", "_")
            progress(i, len(files))
            rel = self._rel(item["path"])
            if dest.exists():
                records.append({"path": rel, "source": self.name, "meta": {}})
                continue
            try:
                resp = requests.get(item["url"], headers=_auth_headers(), timeout=20)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                records.append({"path": rel, "source": self.name, "meta": {}})
                time.sleep(0.05)
            except Exception as e:
                log(f"  Skip {item['path']}: {e}")

        self._write_manifest(records)
        progress(len(files), len(files))
        log(f"Dataset cached: {len(records)} files in {self.data_dir}/")
        return len(records)
