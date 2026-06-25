"""Multi-source dataset ingestion for the music transformer.

Sources implement :class:`~ml.datasets.base.DatasetSource`.  The registry merges
their manifests into one path list for preprocessing/training.
"""

from ml.datasets.base import DatasetSource, resolve_path, STORAGE_DIR
from ml.datasets.registry import (
    SOURCES,
    download_sources,
    load_records,
    load_paths,
    dataset_status,
)

__all__ = [
    "DatasetSource",
    "resolve_path",
    "STORAGE_DIR",
    "SOURCES",
    "download_sources",
    "load_records",
    "load_paths",
    "dataset_status",
]
