"""Downloads and caches the OpenScore/Lieder dataset from GitHub."""

import os
import json
import time
import requests
from pathlib import Path
from typing import Callable, Optional
from dotenv import load_dotenv
from tqdm import tqdm

REPO = "OpenScore/Lieder"
BRANCH = "main"
API_TREE = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
DATA_DIR = Path("/storage/data/openscore")
MANIFEST = DATA_DIR / "manifest.json"

HEADERS = {"Accept": "application/vnd.github+json"}


def _get_token() -> dict:
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return {**HEADERS, "Authorization": f"Bearer {token}"}
    return HEADERS


def fetch_file_list(log: Callable[[str], None]) -> list[dict]:
    log("Fetching file tree from GitHub API...")
    resp = requests.get(API_TREE, headers=_get_token(), timeout=30)
    if resp.status_code == 403:
        raise RuntimeError("GitHub API rate-limited. Set GITHUB_TOKEN env var to increase limits.")
    resp.raise_for_status()
    tree = resp.json().get("tree", [])
    mxl_files = [
        {"path": item["path"], "url": f"{RAW_BASE}/{item['path']}"}
        for item in tree
        if item["path"].endswith(".mxl") or item["path"].endswith(".xml")
    ]
    log(f"Found {len(mxl_files)} score files.")
    return mxl_files


def download_dataset(
    log: Callable[[str], None],
    progress: Callable[[int, int], None],
    max_files: int = 200,
) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST.exists():
        with open(MANIFEST) as f:
            cached = json.load(f)
        if max_files == len(cached):
            log(f"Cache found: {len(cached)} files already downloaded.")
            return len(cached)

    files = fetch_file_list(log)
    files = files[:max_files]
    downloaded = []

    for i, item in tqdm(enumerate(files), total=len(files), desc="Downloading"):
        dest = DATA_DIR / item["path"].replace("/", "_")
        progress(i, len(files))
        if dest.exists():
            downloaded.append(str(dest))
            continue
        try:
            resp = requests.get(item["url"], headers=_get_token(), timeout=20)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            downloaded.append(str(dest))
            time.sleep(0.05)
        except Exception as e:
            log(f"  Skip {item['path']}: {e}")

    with open(MANIFEST, "w") as f:
        json.dump(downloaded, f, indent=2)

    progress(len(files), len(files))
    log(f"Dataset cached: {len(downloaded)} files in {DATA_DIR}/")
    return len(downloaded)


def load_cached_paths() -> list[str]:
    if not MANIFEST.exists():
        return []
    with open(MANIFEST) as f:
        return json.load(f)


def dataset_status() -> str:
    if not MANIFEST.exists():
        return "Not downloaded"
    paths = load_cached_paths()
    return f"{len(paths)} files cached in {DATA_DIR}/"
