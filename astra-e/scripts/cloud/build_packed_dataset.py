"""Cloud Dataset Compiler: Packages extracted sequences into the frozen high-throughput astra-e-features layout.

Layout:
astra-e-features/
├── train/
│   ├── features.npy       # shape: (N_train, 30, 26), float32
│   └── labels.json        # list of metadata & labels
├── validation/
│   ├── features.npy       # shape: (N_val, 30, 26), float32
│   └── labels.json
├── test/
│   ├── features.npy       # shape: (N_test, 30, 26), float32
│   └── labels.json
└── metadata/
    ├── dataset_manifest.json
    └── feature_contract.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.datasets.schemas import (
    DatasetManifest,
    NUM_FEATURES,
    WINDOW_SIZE,
    export_feature_contract_dict,
)


def compile_packed_dataset(
    sequences_dir: str | Path,
    output_dir: str | Path = "data/astra-e-features",
    manifest_path: str | Path | None = None,
    window_size: int = WINDOW_SIZE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compiles raw/aligned sequence files into the standardized memory-mapped dataset layout.
    """
    seq_dir = Path(sequences_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve sequence files
    seq_files = sorted(list(seq_dir.glob("*.npz")))
    if not seq_files:
        raise FileNotFoundError(f"No .npz sequence files found in {seq_dir}")

    # 2. Resolve or generate splits
    splits: dict[str, list[str]] = {}
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            splits = manifest_data.get("splits", {})

    if not splits or "train" not in splits:
        random.seed(seed)
        all_ids = [f.stem for f in seq_files]
        shuffled = list(all_ids)
        random.shuffle(shuffled)
        n_total = len(shuffled)
        n_tr = max(1, int(train_ratio * n_total))
        n_va = max(1, int(val_ratio * n_total)) if n_total >= 3 else 0
        train_ids = sorted(shuffled[:n_tr])
        val_ids = sorted(shuffled[n_tr : n_tr + n_va]) if n_va > 0 else []
        test_ids = sorted(shuffled[n_tr + n_va :])
        if not test_ids and len(train_ids) > 1:
            test_ids = [train_ids.pop()]
        splits = {
            "train": train_ids,
            "validation": val_ids,
            "test": test_ids,
        }
    else:
        # Standardize split key names ("val" -> "validation")
        if "val" in splits and "validation" not in splits:
            splits["validation"] = splits.pop("val")

    file_map = {f.stem: f for f in seq_files}

    split_counts: dict[str, int] = {}
    total_samples = 0

    # 3. Process each split
    for split_name in ("train", "validation", "test"):
        split_ids = splits.get(split_name, [])
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        features_windows: list[np.ndarray] = []
        labels_list: list[dict[str, Any]] = []

        global_sample_idx = 0
        for run_id in split_ids:
            if run_id not in file_map:
                continue
            npz_path = file_map[run_id]

            with np.load(npz_path, allow_pickle=True) as data:
                features = data["features"]  # (T, 26)
                verbs = data["verbs"]        # (T,)
                objects = data["objects"]    # (T,)
                targets = data["targets"]    # (T,)

            t_total = len(features)
            if t_total < window_size:
                continue

            # Standardized sample identifiers
            video_id = f"EXP001_{run_id}_CAM01"
            subject_id = "ASTRONAUT-01"

            for t in range(window_size - 1, t_total):
                start_f = t - window_size + 1
                end_f = t
                seq_id = f"{video_id}_{global_sample_idx + 1:06d}"

                window = features[start_f : end_f + 1]  # shape: (30, 26)
                features_windows.append(window)

                label_item = {
                    "sample_idx": global_sample_idx,
                    "sequence_id": seq_id,
                    "run_id": run_id,
                    "subject_id": subject_id,
                    "video_id": video_id,
                    "start_frame": int(start_f),
                    "end_frame": int(end_f),
                    "verb": int(verbs[t]),
                    "object": int(objects[t]),
                    "target": int(targets[t]),
                }
                labels_list.append(label_item)
                global_sample_idx += 1

        # Save binary memory-mappable tensor: [N, 30, 26] float32
        if features_windows:
            feats_arr = np.array(features_windows, dtype=np.float32)
        else:
            feats_arr = np.empty((0, window_size, NUM_FEATURES), dtype=np.float32)

        features_npy_path = split_dir / "features.npy"
        np.save(features_npy_path, feats_arr)

        labels_json_path = split_dir / "labels.json"
        with open(labels_json_path, "w", encoding="utf-8") as f:
            json.dump(labels_list, f, indent=2)

        split_counts[split_name] = len(feats_arr)
        total_samples += len(feats_arr)

    # 4. Write Metadata
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    feature_contract_path = meta_dir / "feature_contract.json"
    with open(feature_contract_path, "w", encoding="utf-8") as f:
        json.dump(export_feature_contract_dict(), f, indent=2)

    manifest_path_out = meta_dir / "dataset_manifest.json"
    manifest_data = {
        "dataset_name": "astra-e-features",
        "dataset_version": time.strftime("%Y.%m.%d"),
        "created_at": time.time(),
        "num_features": NUM_FEATURES,
        "window_size": window_size,
        "tensor_layout": "X.shape = [N, 30, 26], float32",
        "labels_layout": "verb = [N], object = [N], target = [N], int64",
        "total_samples": total_samples,
        "split_counts": split_counts,
        "splits_provenance": splits,
        "feature_contract_file": "metadata/feature_contract.json",
    }
    with open(manifest_path_out, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile aligned sequences into high-performance astra-e-features format.")
    parser.add_argument("--sequences-dir", type=str, default="data/processed/EXP001", help="Path to input sequence .npz files")
    parser.add_argument("--output-dir", type=str, default="data/astra-e-features", help="Output directory for packed dataset")
    parser.add_argument("--manifest", type=str, default=None, help="Optional existing manifest containing split IDs")
    parser.add_argument("--window-size", type=int, default=30, help="Sliding window size (frames)")
    args = parser.parse_args()

    result = compile_packed_dataset(
        sequences_dir=args.sequences_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        window_size=args.window_size,
    )
    print(json.dumps(result, indent=2))
