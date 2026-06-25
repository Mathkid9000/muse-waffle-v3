"""PDMX source — a capped, high-quality subset of the PDMX corpus.

PDMX (~250K public-domain MusicXML scores from MuseScore) is hosted on Zenodo
(record 14648209, CC-BY) as several separate archives.  We only need two of them:

  * ``PDMX.csv``      (~0.2 GB) — the master metadata table we filter on
  * ``mxl.tar.gz``    (~1.8 GB) — the compressed MusicXML files

so the ``pdf.tar.gz`` (~9 GB) and ``data.tar.gz`` (~2 GB) archives are skipped.
``download()`` fetches these from Zenodo directly (good for headless/remote
notebooks) and, to save disk, extracts **only the .mxl files of the selected
subset** from the tarball rather than all ~250K.

Selection uses PDMX's own metadata: keep rows in the rated+deduplicated and
all-valid subsets, sort by the 5-star ``rating``, and take the top ``max_files``.

Set ``download=False`` (and pass ``pdmx_root``) to use an already-extracted copy
instead of downloading.  Column names/path formats are auto-detected with
overrides, since they can drift between releases.
"""

from __future__ import annotations

import csv
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import requests
from tqdm import tqdm

from ml.datasets.base import DatasetSource

ZENODO_RECORD = "14648209"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD}"

# Only these Zenodo files are needed for MusicXML training.
METADATA_FILE = "PDMX.csv"
MXL_ARCHIVE = "mxl.tar.gz"

# Subset boolean columns to require (only applied when present in the table).
DEFAULT_SUBSET_FILTERS = ["subset:rated_deduplicated", "subset:all_valid"]
# Candidate column names holding the path to each score's .mxl file.
MXL_COLUMN_CANDIDATES = ["mxl", "mxl_path", "path_mxl"]
RATING_COLUMN_CANDIDATES = ["rating", "rating_avg", "stars"]


def _truthy(val: str) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "t")


def _first_present(fieldnames, candidates):
    for c in candidates:
        if c in fieldnames:
            return c
    return None


def _norm_rel(raw: str) -> str:
    """Normalise a CSV path like './mxl/a/b.mxl' to 'mxl/a/b.mxl'."""
    return str(raw).replace("\\", "/").lstrip("./")


class PDMXSource(DatasetSource):
    name = "pdmx"

    # -- Zenodo download --------------------------------------------------

    def _zenodo_files(self, log: Callable[[str], None]) -> dict:
        """Map Zenodo filename -> {"url", "size"} for this record."""
        resp = requests.get(ZENODO_API, timeout=30)
        resp.raise_for_status()
        files = {}
        for entry in resp.json().get("files", []):
            files[entry["key"]] = {
                "url": entry["links"]["self"],
                "size": entry.get("size"),
            }
        return files

    def _download_file(self, url: str, dest: Path, size: int | None, log) -> None:
        if dest.exists() and size is not None and dest.stat().st_size == size:
            log(f"  {dest.name} already present ({size/1e9:.2f} GB).")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", size or 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=dest.name
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))

    def _ensure_csv(self, root: Path, download: bool, log) -> Path:
        csv_path = root / METADATA_FILE
        if csv_path.exists():
            return csv_path
        if not download:
            raise FileNotFoundError(
                f"{csv_path} not found and download=False. Download {METADATA_FILE} "
                f"from {ZENODO_URL}, or call with download=True."
            )
        files = self._zenodo_files(log)
        if METADATA_FILE not in files:
            raise RuntimeError(f"{METADATA_FILE} not found in Zenodo record {ZENODO_RECORD}.")
        log(f"Downloading {METADATA_FILE} from Zenodo...")
        self._download_file(files[METADATA_FILE]["url"], csv_path, files[METADATA_FILE]["size"], log)
        return csv_path

    def _ensure_mxl(self, root: Path, wanted_rel: list[str], download: bool,
                    keep_archive: bool, log, progress) -> None:
        """Make sure the selected .mxl files exist under root (extract subset)."""
        missing = [r for r in wanted_rel if not (root / r).exists()]
        if not missing:
            return
        if not download:
            raise FileNotFoundError(
                f"{len(missing)} selected .mxl files missing under {root} and "
                f"download=False. Extract {MXL_ARCHIVE} there, or call with download=True."
            )
        wanted = set(missing)
        archive = root / MXL_ARCHIVE
        files = self._zenodo_files(log)
        if MXL_ARCHIVE not in files:
            raise RuntimeError(f"{MXL_ARCHIVE} not found in Zenodo record {ZENODO_RECORD}.")
        log(f"Downloading {MXL_ARCHIVE} (~{(files[MXL_ARCHIVE]['size'] or 0)/1e9:.1f} GB) from Zenodo...")
        self._download_file(files[MXL_ARCHIVE]["url"], archive, files[MXL_ARCHIVE]["size"], log)

        # Match tar members to wanted paths by basename (PDMX filenames are unique
        # MuseScore ids), disambiguating with a suffix check. This is robust to any
        # internal prefix the archive may use ('./', 'mxl/', 'PDMX/mxl/', ...). Each
        # file is written to its canonical root/<rel> path via extractfile(), so the
        # location never depends on the archive's layout.
        by_base: dict[str, list[str]] = defaultdict(list)
        for rel in wanted:
            by_base[rel.rsplit("/", 1)[-1]].append(rel)

        log(f"Extracting {len(wanted)} selected .mxl files from {MXL_ARCHIVE}...")
        extracted = 0
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = _norm_rel(member.name)
                cands = by_base.get(name.rsplit("/", 1)[-1])
                if not cands:
                    continue
                rel = next((c for c in cands if name.endswith(c)), None)
                if rel is None:
                    continue
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is None:
                    continue
                with src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                extracted += 1
                progress(extracted, len(wanted))
                if extracted == len(wanted):
                    break
        log(f"  Extracted {extracted}/{len(wanted)} files.")
        if not keep_archive:
            archive.unlink(missing_ok=True)
            log(f"  Removed {MXL_ARCHIVE} to save disk (keep_archive=True to retain).")

    # -- metadata table ---------------------------------------------------

    def _find_metadata_csv(self, root: Path, override: str | None) -> Path:
        if override:
            p = Path(override)
            return p if p.is_absolute() else root / p
        if (root / METADATA_FILE).exists():
            return root / METADATA_FILE
        for pat in ("*.csv", "*.tsv"):
            hits = sorted(root.glob(pat))
            if hits:
                return hits[0]
        raise FileNotFoundError(
            f"No PDMX metadata table (.csv/.tsv) found under {root}. "
            "Pass metadata_csv=... to point at it."
        )

    def _select_rows(self, csv_path: Path, max_files: int, min_rating: float,
                     subset_filters, mxl_column, log) -> list[tuple]:
        """Return up to max_files (rating, mxl_rel, meta) rows, best rating first."""
        with open(csv_path, newline="", encoding="utf-8") as f:
            sample = f.read(8192)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            fields = reader.fieldnames or []

            filters = subset_filters if subset_filters is not None else DEFAULT_SUBSET_FILTERS
            active = [c for c in filters if c in fields]
            log(f"Applying subset filters present in table: {active or '(none)'}")

            mxl_col = mxl_column or _first_present(fields, MXL_COLUMN_CANDIDATES)
            rating_col = _first_present(fields, RATING_COLUMN_CANDIDATES)
            if not mxl_col:
                raise RuntimeError(
                    f"Could not find an mxl path column in {fields}. Pass mxl_column=..."
                )

            rows = []
            for row in reader:
                if not all(_truthy(row.get(c, "")) for c in active):
                    continue
                rating = 0.0
                if rating_col:
                    try:
                        rating = float(row.get(rating_col) or 0.0)
                    except ValueError:
                        rating = 0.0
                if rating < min_rating or not row.get(mxl_col):
                    continue
                meta = {"rating": rating}
                gcol = _first_present(row.keys(), ["genres", "genre"])
                if gcol and row.get(gcol):
                    meta["genre"] = row[gcol]
                rows.append((rating, _norm_rel(row[mxl_col]), meta))

        rows.sort(key=lambda r: r[0], reverse=True)
        log(f"{len(rows)} rows pass filters; taking top {min(max_files, len(rows))}.")
        return rows[:max_files]

    # -- main entry -------------------------------------------------------

    def download(
        self,
        log: Callable[[str], None],
        progress: Callable[[int, int], None],
        max_files: int = 20000,
        min_rating: float = 0.0,
        pdmx_root: str | None = None,
        metadata_csv: str | None = None,
        mxl_column: str | None = None,
        subset_filters: list[str] | None = None,
        download: bool = True,
        keep_archive: bool = False,
        **opts,
    ) -> int:
        # Pre-extracted copy: use it as-is; otherwise download into our data dir.
        if pdmx_root:
            root = Path(pdmx_root)
            csv_path = self._find_metadata_csv(root, metadata_csv)
            do_download = False
        else:
            root = self.data_dir
            root.mkdir(parents=True, exist_ok=True)
            csv_path = self._ensure_csv(root, download, log)
            do_download = download

        selected = self._select_rows(csv_path, max_files, min_rating,
                                     subset_filters, mxl_column, log)
        if not pdmx_root:
            self._ensure_mxl(root, [rel for _, rel, _ in selected], do_download,
                             keep_archive, log, progress)

        records, missing = [], 0
        for rating, rel, meta in selected:
            p = (root / rel)
            if not p.exists():
                missing += 1
                continue
            records.append({"path": str(p.resolve()), "source": self.name, "meta": meta})

        if missing:
            log(f"  ({missing} selected rows skipped: .mxl not found on disk)")
        self._write_manifest(records)
        log(f"PDMX subset cached: {len(records)} files (manifest in {self.data_dir}/)")
        return len(records)
