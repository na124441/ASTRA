"""Cloud Split Generator: Partitions supervised sequence files into disjoint train/val/test splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.datasets.schemas import DatasetManifest
from ml.datasets.splits import SplitManager


def generate_dataset_splits(
    sequences_dir: str | Path,
    output_manifest: str | Path = "data/manifests/dataset_manifest.json",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    window_size: int = 30,
) -> DatasetManifest:
    """
    Generate deterministic, leak-free train/val/test splits at the recording level.
    """
    seq_dir = Path(sequences_dir)
    out_manifest = Path(output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(seq_dir.glob("*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz sequence files found in {seq_dir}")

    random.seed(seed)
    run_ids = [f.stem for f in npz_files]
    shuffled = list(run_ids)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = max(1, int(train_ratio * n_total))
    n_val = max(1, int(val_ratio * n_total)) if n_total >= 3 else 0
    
    train_runs = sorted(shuffled[:n_train])
    val_runs = sorted(shuffled[n_train : n_train + n_val]) if n_val > 0 else []
    test_runs = sorted(shuffled[n_train + n_val :])

    # If test is empty due to small dataset, take from train
    if not test_runs and len(train_runs) > 1:
        test_runs = [train_runs.pop()]

    # Calculate total windows
    total_windows = 0
    for f in npz_files:
        with np.load(f, allow_pickle=True) as d:
            n_f = len(d["features"])
            total_windows += max(0, n_f - window_size + 1)

    manifest = DatasetManifest(
        dataset_version=time.strftime("%Y.%m.%d"),
        generator_version="1.0.0",
        feature_schema_version="kinematic-26d-v1.0",
        random_seed=seed,
        recordings_count=n_total,
        total_windows=total_windows,
        splits={
            "train": train_runs,
            "val": val_runs,
            "test": test_runs,
        },
        created_at=time.time(),
        metadata={
            "sequences_dir": str(seq_dir),
            "train_count": len(train_runs),
            "val_count": len(val_runs),
            "test_count": len(test_runs),
            "window_size": window_size,
        },
    )

    with open(out_manifest, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    print("\n" + "=" * 68)
    print("║" + "DATASET SPLIT GENERATION SUMMARY".center(66) + "║")
    print("=" * 68)
    print(f"  Sequences Count:    {n_total}")
    print(f"  Total 30-f Windows: {total_windows:,}")
    print(f"  Train Runs:         {len(train_runs)} ({len(train_runs)/n_total*100:.1f}%)")
    print(f"  Val Runs:           {len(val_runs)} ({len(val_runs)/n_total*100:.1f}%)")
    print(f"  Test Runs:          {len(test_runs)} ({len(test_runs)/n_total*100:.1f}%)")
    print(f"  Manifest Path:      {out_manifest}")
    print("=" * 68 + "\n")

    # Run leakage verification
    manager = SplitManager(manifest_path=out_manifest)
    manager.verify_no_leakage()
    print("✓ Split leakage verification PASSED: 0% data overlap detected.")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA-E Split Generator")
    parser.add_argument("--sequences-dir", default="data/processed/EXP001", help="Directory with processed .npz sequences")
    parser.add_argument("--output-manifest", default="data/manifests/dataset_manifest.json", help="Target manifest JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default: 0.15)")
    args = parser.parse_args()

    generate_dataset_splits(
        sequences_dir=args.sequences_dir,
        output_manifest=args.output_manifest,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
