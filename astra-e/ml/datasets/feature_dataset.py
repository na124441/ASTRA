"""PyTorch FeatureSequenceDataset for causal temporal training and evaluation on 26-D sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml.datasets.schemas import (
    OBJECT_TO_IDX,
    TARGET_TO_IDX,
    VERB_TO_IDX,
    VIOLATION_VOCAB,
    NUM_FEATURES,
)


class FeatureSequenceDataset(Dataset):
    """
    Causal Sliding-Window Dataset consuming pre-extracted 26-D kinematic feature sequences.
    
    Each sample represents:
      - X_t: Sliding window of shape (T=window_size, D=26), representing historical frames [t - window_size + 1, ..., t].
      - y_verb: Action verb index at current endpoint frame t.
      - y_object: Interacted object index at current endpoint frame t.
      - y_target: Target zone index at current endpoint frame t.
      - y_violation: Ground-truth procedural deviation index at current frame t.
      - t_end: Frame timestamp in seconds.
    
    Guarantees strict temporal causality (zero future lookahead).
    """

    def __init__(
        self,
        sequence_files: Sequence[str | Path],
        window_size: int = 30,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.window_size = window_size
        self.transform = transform
        self.samples: list[dict[str, Any]] = []

        self._load_sequences(sequence_files)

    def _load_sequences(self, sequence_files: Sequence[str | Path]) -> None:
        """Load and index all valid sliding windows across sequence files."""
        for seq_file in sequence_files:
            p = Path(seq_file)
            if not p.exists():
                continue

            with np.load(p, allow_pickle=True) as data:
                features = data["features"]  # (N, 26)
                verbs = data["verbs"]        # (N,)
                objects = data["objects"]    # (N,)
                targets = data["targets"]    # (N,)

                # Optional fields
                violations = data["violations"] if "violations" in data else np.zeros(len(features), dtype=np.int64)
                timestamps = data["timestamps"] if "timestamps" in data else np.arange(len(features)) * (1.0 / 30.0)

            n_frames = len(features)
            if n_frames < self.window_size:
                continue

            for t in range(self.window_size - 1, n_frames):
                # Window spans [t - window_size + 1 : t + 1] -> shape (window_size, 26)
                window = features[t - self.window_size + 1 : t + 1]
                self.samples.append({
                    "window": window,
                    "verb": int(verbs[t]),
                    "object": int(objects[t]),
                    "target": int(targets[t]),
                    "violation": int(violations[t]),
                    "timestamp": float(timestamps[t]),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        window = sample["window"].copy()

        if self.transform is not None:
            window = self.transform(window)

        return {
            "features": torch.from_numpy(window).float(),  # (30, 26)
            "verb": torch.tensor(sample["verb"], dtype=torch.long),
            "object": torch.tensor(sample["object"], dtype=torch.long),
            "target": torch.tensor(sample["target"], dtype=torch.long),
            "violation": torch.tensor(sample["violation"], dtype=torch.long),
            "timestamp": torch.tensor(sample["timestamp"], dtype=torch.float32),
        }


def create_feature_dataloaders(
    sequences_dir: str | Path,
    manifest_path: str | Path,
    batch_size: int = 32,
    window_size: int = 30,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory creating PyTorch DataLoaders for train, val, and test splits from a dataset manifest.
    """
    seq_dir = Path(sequences_dir)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    splits = manifest_data.get("splits", {})
    train_ids = splits.get("train", [])
    val_ids = splits.get("val", [])
    test_ids = splits.get("test", [])

    train_files = [seq_dir / f"{rid}.npz" for rid in train_ids]
    val_files = [seq_dir / f"{rid}.npz" for rid in val_ids]
    test_files = [seq_dir / f"{rid}.npz" for rid in test_ids]

    train_ds = FeatureSequenceDataset(train_files, window_size=window_size)
    val_ds = FeatureSequenceDataset(val_files, window_size=window_size)
    test_ds = FeatureSequenceDataset(test_files, window_size=window_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


class MmapFeatureDataset(Dataset):
    """
    High-Performance Memory-Mapped Sequence Dataset for the frozen astra-e-features format:
      features.npy: shape [N, 30, 26] float32
      labels.json: list of dicts with sequence metadata & multi-head labels

    Zero-copy, near-zero RAM footprint, instant startup time.
    """

    def __init__(
        self,
        split_dir: str | Path,
        mmap_mode: str = "r",
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.features_file = self.split_dir / "features.npy"
        self.labels_file = self.split_dir / "labels.json"
        self.transform = transform

        if not self.features_file.exists():
            raise FileNotFoundError(f"Missing features.npy in {self.split_dir}")
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Missing labels.json in {self.split_dir}")

        # Memory-map the 3D tensor: [N, 30, 26]
        self.features = np.load(self.features_file, mmap_mode=mmap_mode)

        with open(self.labels_file, "r", encoding="utf-8") as f:
            self.labels: list[dict[str, Any]] = json.load(f)

        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Mismatch in {self.split_dir}: features.npy has {len(self.features)} samples "
                f"but labels.json has {len(self.labels)} items"
            )

        # Pre-extract label arrays into contiguous int64 tensors for zero-overhead collation
        self.verbs = torch.tensor([item["verb"] for item in self.labels], dtype=torch.long)
        self.objects = torch.tensor([item["object"] for item in self.labels], dtype=torch.long)
        self.targets = torch.tensor([item["target"] for item in self.labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Window of shape [30, 26]
        window = np.array(self.features[idx], dtype=np.float32)
        if self.transform is not None:
            window = self.transform(window)

        return {
            "features": torch.from_numpy(window).float(),  # [30, 26]
            "verb": self.verbs[idx],                       # scalar long
            "object": self.objects[idx],                   # scalar long
            "target": self.targets[idx],                   # scalar long
            "sequence_id": self.labels[idx]["sequence_id"],
        }

    def to_logical_sample(self, idx: int) -> dict[str, Any]:
        """
        Reconstructs the full logical sample schema matching the user specification:
        {
          "sequence_id": "EXP001_RUN_001_CAM01_000001",
          "run_id": "RUN-0001",
          "subject_id": "ASTRONAUT-01",
          "video_id": "EXP001_RUN_001_CAM01",
          "start_frame": 0,
          "end_frame": 29,
          "features": [[26 values], "... 30 frames ..."],
          "verb": 1,
          "object": 1,
          "target": 0
        }
        """
        meta = self.labels[idx]
        feat_window = np.array(self.features[idx], dtype=np.float32).tolist()
        return {
            "sequence_id": meta["sequence_id"],
            "run_id": meta.get("run_id", "RUN-0001"),
            "subject_id": meta.get("subject_id", "ASTRONAUT-01"),
            "video_id": meta.get("video_id", "EXP001_RUN_001_CAM01"),
            "start_frame": int(meta.get("start_frame", 0)),
            "end_frame": int(meta.get("end_frame", 29)),
            "features": feat_window,
            "verb": int(meta["verb"]),
            "object": int(meta["object"]),
            "target": int(meta["target"]),
        }


def create_mmap_dataloaders(
    dataset_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory creating PyTorch DataLoaders for train, validation, and test splits
    from the packed astra-e-features layout.
    """
    base_dir = Path(dataset_dir)
    train_ds = MmapFeatureDataset(base_dir / "train")
    val_ds = MmapFeatureDataset(base_dir / "validation")
    test_ds = MmapFeatureDataset(base_dir / "test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

