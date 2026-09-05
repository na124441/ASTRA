"""Recording-level split management and leakage protection."""

from __future__ import annotations

import json
from pathlib import Path
from ml.datasets.schemas import DatasetManifest


class SplitManager:
    """Manages recording-level splits strictly by run_id to avoid temporal frame leakage."""

    def __init__(self, manifest_path: str | Path = "data/manifests/dataset_manifest.json") -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest: DatasetManifest | None = None
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.manifest = DatasetManifest.model_validate(data)

    def get_split_runs(self, split_name: str) -> list[str]:
        """Return list of run IDs assigned to split ('train', 'val', 'test')."""
        if self.manifest is None:
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
        return self.manifest.splits.get(split_name, [])

    def verify_no_leakage(self) -> bool:
        """Verify that train, val, and test run IDs are completely disjoint."""
        if self.manifest is None:
            return True
        train_set = set(self.manifest.splits.get("train", []))
        val_set = set(self.manifest.splits.get("val", []))
        test_set = set(self.manifest.splits.get("test", []))

        assert len(train_set & val_set) == 0, "Leakage detected between train and val!"
        assert len(train_set & test_set) == 0, "Leakage detected between train and test!"
        assert len(val_set & test_set) == 0, "Leakage detected between val and test!"
        return True
