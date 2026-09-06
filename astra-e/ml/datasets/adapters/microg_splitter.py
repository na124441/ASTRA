"""Grouped Video-Level Dataset Splitter for MicroG-4M.

Enforces zero temporal data leakage across train, validation, and test splits.
All sliding windows belonging to the same video_id (or person_id group) are
strictly partitioned together into exactly one split.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger("astra.ml.microg_splitter")


class GroupedSplitError(ValueError):
    """Raised when grouped splitting invariants are violated."""
    pass


class MicroGGroupedSplitter:
    """
    Deterministic grouped split generator for MicroG-4M.
    Guarantees that no video_id appears across multiple partitions.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        if not (0.999 <= total <= 1.001):
            raise GroupedSplitError(f"Split ratios must sum to 1.0, got {total}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split_annotations(
        self,
        annotations: Sequence[dict[str, Any]],
        group_by: str = "video_id",
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Group annotations by group_by (default: video_id) and partition deterministically.
        Returns:
            {"train": [...], "val": [...], "test": [...]}
        """
        if not annotations:
            raise GroupedSplitError("Cannot split empty annotations list.")

        # Group rows by key
        grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in annotations:
            key = str(r.get(group_by, "unknown"))
            grouped_rows[key].append(r)

        unique_groups = sorted(grouped_rows.keys())
        n_groups = len(unique_groups)

        if n_groups < 3:
            raise GroupedSplitError(f"Insufficient groups ({n_groups}) to create 3 partitions.")

        # Deterministic shuffle using configured seed
        rng = random.Random(self.seed)
        shuffled_groups = list(unique_groups)
        rng.shuffle(shuffled_groups)

        n_train = max(1, int(round(n_groups * self.train_ratio)))
        n_val = max(1, int(round(n_groups * self.val_ratio)))
        n_test = n_groups - (n_train + n_val)
        if n_test < 1:
            n_test = 1
            if n_train > 1:
                n_train -= 1

        train_groups = set(shuffled_groups[:n_train])
        val_groups = set(shuffled_groups[n_train : n_train + n_val])
        test_groups = set(shuffled_groups[n_train + n_val :])

        # Strict leakage assertion
        assert train_groups.isdisjoint(val_groups), "Data leakage: train and val overlap!"
        assert train_groups.isdisjoint(test_groups), "Data leakage: train and test overlap!"
        assert val_groups.isdisjoint(test_groups), "Data leakage: val and test overlap!"

        partitioned: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for grp in train_groups:
            partitioned["train"].extend(grouped_rows[grp])
        for grp in val_groups:
            partitioned["val"].extend(grouped_rows[grp])
        for grp in test_groups:
            partitioned["test"].extend(grouped_rows[grp])

        return partitioned

    def export_split_manifest(
        self,
        partitioned: dict[str, list[dict[str, Any]]],
        output_path: str | Path,
        group_by: str = "video_id",
    ) -> Path:
        """
        Saves split manifest with audit metadata, group lists, and sample counts.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "seed": self.seed,
            "grouping_key": group_by,
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "counts": {
                "train_samples": len(partitioned["train"]),
                "val_samples": len(partitioned["val"]),
                "test_samples": len(partitioned["test"]),
                "train_groups": len(set(r[group_by] for r in partitioned["train"])),
                "val_groups": len(set(r[group_by] for r in partitioned["val"])),
                "test_groups": len(set(r[group_by] for r in partitioned["test"])),
            },
            "groups": {
                "train": sorted(list(set(r[group_by] for r in partitioned["train"]))),
                "val": sorted(list(set(r[group_by] for r in partitioned["val"]))),
                "test": sorted(list(set(r[group_by] for r in partitioned["test"]))),
            },
        }

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return out_p
