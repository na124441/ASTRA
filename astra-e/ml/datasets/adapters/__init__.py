"""Dataset adapters for external video benchmarks and lab recordings."""

from ml.datasets.adapters.base import BaseDatasetAdapter
from ml.datasets.adapters.microg import MicroGDatasetAdapter

__all__ = ["BaseDatasetAdapter", "MicroGDatasetAdapter"]
