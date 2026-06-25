"""PDMX source — a capped, high-quality subset of the PDMX corpus.

PDMX (~250K public-domain MusicXML scores from MuseScore) is distributed as a
large archive on Zenodo (record 14648209, CC-BY) with a companion GitHub repo
(pnlong/PDMX).  It is far too large to download file-by-file, so this source
expects the archive to be **downloaded and extracted once** to a root directory,
then selects a high-quality subset using PDMX's own metadata table:

  * keep rows flagged as both rated AND deduplicated  (best unique arrangements)
  * keep rows whose MXL/PDF validated
  * sort by the 5-star ``rating`` and take the top ``max_files``

Selected ``.mxl`` files are referenced **in place** (absolute paths in the
manifest) to avoid copying tens of thousands of files into storage.

The exact column names in PDMX's table can drift between releases, so filtering
is defensive: a filter is only applied when its column exists, and the mxl-path
column is configurable.  Tune via the ``download`` kwargs if a release differs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from ml.datasets.base import STORAGE_DIR, DatasetSource

ZENODO_URL = "https://zenodo.org/records/14648209"

# Subset boolean columns to require (only applied if present in the table).
DEFAULT_SUBSET_FILTERS = ["rated_deduplicated", "valid_mxl_pdf"]
# Candidate column names holding the path to each score's .mxl file.
MXL_COLUMN_CANDIDATES = ["mxl", "mxl_path", "path_mxl"]
# Candidate column names holding a path we can derive the .mxl from.
PATH_COLUMN_CANDIDATES = ["path", "data", "json", "metadata"]
RATING_COLUMN_CANDIDATES = ["rating", "rating_avg", "stars"]


def _truthy(val: str) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "t")


def _first_present(fieldnames, candidates):
    for c in candidates:
        if c in fieldnames:
            return c
    return None


class PDMXSource(DatasetSource):
    name = "pdmx"

    def _find_metadata_csv(self, root: Path, override: str | None) -> Path:
        if override:
            p = Path(override)
            return p if p.is_absolute() else root / p
        # Prefer a top-level CSV/TSV; fall back to any csv in the tree.
        for pat in ("*.csv", "*.tsv"):
            hits = sorted(root.glob(pat))
            if hits:
                return hits[0]
        hits = sorted(root.rglob("*.csv"))
        if hits:
            return hits[0]
        raise FileNotFoundError(
            f"No PDMX metadata table (.csv/.tsv) found under {root}. "
            "Pass metadata_csv=... to point at it."
        )

    def _resolve_mxl(self, root: Path, row: dict, mxl_col, path_col) -> Path | None:
        if mxl_col and row.get(mxl_col):
            raw = row[mxl_col]
        elif path_col and row.get(path_col):
            raw = str(Path(row[path_col]).with_suffix(".mxl"))
        else:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        return p

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
        **opts,
    ) -> int:
        root = Path(pdmx_root) if pdmx_root else (self.data_dir / "PDMX")
        if not root.exists():
            raise FileNotFoundError(
                f"PDMX root not found at {root}.\n"
                f"Download + extract the PDMX archive from {ZENODO_URL} (CC-BY), then\n"
                f"either place it at {self.data_dir}/PDMX or pass pdmx_root=<extracted dir>."
            )

        csv_path = self._find_metadata_csv(root, metadata_csv)
        log(f"Reading PDMX metadata: {csv_path}")

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
            active_filters = [c for c in filters if c in fields]
            log(f"Applying subset filters present in table: {active_filters or '(none)'}")

            mxl_col = mxl_column or _first_present(fields, MXL_COLUMN_CANDIDATES)
            path_col = _first_present(fields, PATH_COLUMN_CANDIDATES)
            rating_col = _first_present(fields, RATING_COLUMN_CANDIDATES)
            if not mxl_col and not path_col:
                raise RuntimeError(
                    f"Could not find an mxl/path column in {fields}. "
                    "Pass mxl_column=... to specify it."
                )

            rows = []
            for row in reader:
                if not all(_truthy(row.get(c, "")) for c in active_filters):
                    continue
                rating = 0.0
                if rating_col:
                    try:
                        rating = float(row.get(rating_col) or 0.0)
                    except ValueError:
                        rating = 0.0
                if rating < min_rating:
                    continue
                rows.append((rating, row))

        # Best-rated first, then resolve to existing .mxl files until we hit max_files.
        rows.sort(key=lambda r: r[0], reverse=True)
        log(f"{len(rows)} rows pass filters; resolving up to {max_files} .mxl files...")

        records: list[dict] = []
        missing = 0
        for rating, row in rows:
            if len(records) >= max_files:
                break
            mxl = self._resolve_mxl(root, row, mxl_col, path_col)
            if mxl is None or not mxl.exists():
                missing += 1
                continue
            meta = {"rating": rating}
            genre_col = _first_present(row.keys(), ["genres", "genre"])
            if genre_col and row.get(genre_col):
                meta["genre"] = row[genre_col]
            records.append({"path": str(mxl.resolve()), "source": self.name, "meta": meta})
            progress(len(records), max_files)

        if missing:
            log(f"  ({missing} selected rows skipped: .mxl file not found on disk)")
        self._write_manifest(records)
        log(f"PDMX subset cached: {len(records)} files referenced (manifest in {self.data_dir}/)")
        return len(records)
