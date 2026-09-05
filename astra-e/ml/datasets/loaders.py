"""PyTorch Dataset and DataLoader for causal sliding-window feature sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ml.datasets.splits import SplitManager


class CausalWindowDataset(Dataset):
    """
    Causal Sliding Window Dataset.
    Each sample is a 30-frame historical window of 26-D physical features X[t-29:t],
    with target labels aligned strictly at the endpoint frame t.
    Zero future lookahead.
    """

    def __init__(
        self,
        run_ids: Sequence[str],
        data_dir: str | Path = "data/processed/EXP001",
        window_size: int = 30,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.samples: list[tuple[np.ndarray, int, int, int]] = []

        self._load_runs(run_ids)

    def _load_runs(self, run_ids: Sequence[str]) -> None:
        for rid in run_ids:
            npz_file = self.data_dir / f"{rid}.npz"
            if not npz_file.exists():
                continue

            with np.load(npz_file) as data:
                features = data["features"]  # (T, 26)
                verbs = data["verbs"]        # (T,)
                objects = data["objects"]    # (T,)
                targets = data["targets"]    # (T,)

            t_total = len(features)
            for t in range(self.window_size - 1, t_total):
                window = features[t - self.window_size + 1 : t + 1]  # shape: (30, 26)
                v_lbl = int(verbs[t])
                o_lbl = int(objects[t])
                t_lbl = int(targets[t])
                self.samples.append((window, v_lbl, o_lbl, t_lbl))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        window, v, o, t = self.samples[idx]
        return (
            torch.from_numpy(window).float(),
            torch.tensor(v, dtype=torch.long),
            torch.tensor(o, dtype=torch.long),
            torch.tensor(t, dtype=torch.long),
        )


def create_dataloaders(
    data_dir: str | Path = "data/processed/EXP001",
    manifest_path: str | Path = "data/manifests/dataset_manifest.json",
    batch_size: int = 32,
    window_size: int = 30,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Factory creating PyTorch DataLoaders for train, val, and test splits."""
    manager = SplitManager(manifest_path=manifest_path)
    manager.verify_no_leakage()

    train_runs = manager.get_split_runs("train")
    val_runs = manager.get_split_runs("val")
    test_runs = manager.get_split_runs("test")

    train_ds = CausalWindowDataset(train_runs, data_dir=data_dir, window_size=window_size)
    val_ds = CausalWindowDataset(val_runs, data_dir=data_dir, window_size=window_size)
    test_ds = CausalWindowDataset(test_runs, data_dir=data_dir, window_size=window_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
