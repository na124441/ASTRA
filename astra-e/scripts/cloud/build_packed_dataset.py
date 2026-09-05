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
from ml.datasets.sequence_generator import TemporalSequenceGenerator



def compile_packed_dataset(
    sequences_dir: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path = "data/astra-e-features",
    window_size: int = WINDOW_SIZE,
) -> dict[str, Any]:
    """
    Compiles raw/aligned sequence files into the standardized memory-mapped dataset layout.
    Fails closed: strictly requires a validated, leakage-free split manifest by run/subject.
    """
    seq_dir = Path(sequences_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve sequence files
    seq_files = sorted(list(seq_dir.glob("*.npz")))
    if not seq_files:
        raise FileNotFoundError(f"No .npz sequence files found in {seq_dir}")

    file_map: dict[str, Path] = {}
    run_to_files: dict[str, list[Path]] = {}
    for f in seq_files:
        file_map[f.stem] = f
        canonical_run = f.stem

        meta_f = f.parent / f"{f.stem}_meta.json"
        if not meta_f.exists():
            meta_f = f.parent / f"{f.stem}.json"

        if meta_f.exists():
            try:
                with open(meta_f, "r", encoding="utf-8") as mf:
                    m = json.load(mf)
                    if "run_id" in m and m["run_id"]:
                        canonical_run = str(m["run_id"])
            except Exception:
                pass
        else:
            parts = f.stem.split("_CAM")[0].split("-CAM")[0]
            if "RUN" in parts:
                canonical_run = parts

        if canonical_run not in run_to_files:
            run_to_files[canonical_run] = []
        run_to_files[canonical_run].append(f)
        if canonical_run not in file_map:
            file_map[canonical_run] = f


    # 2. Strict Fail-Closed Validation of Split Manifest
    if manifest_path is None:
        raise ValueError(
            "A leakage-safe split manifest is strictly required to compile a packed dataset. "
            "Never split windows randomly. Generate a validated recording/subject-level split "
            "manifest first using scripts/cloud/create_splits.py."
        )

    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(
            f"Specified split manifest does not exist at: {m_path}. "
            "Generate one first using scripts/cloud/create_splits.py."
        )

    with open(m_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    raw_splits = manifest_data.get("splits")
    if not isinstance(raw_splits, dict):
        if "train" in manifest_data and isinstance(manifest_data["train"], dict) and "runs" in manifest_data["train"]:
            raw_splits = {
                "train": manifest_data["train"].get("runs", []),
                "validation": manifest_data.get("validation", {}).get("runs", []),
                "test": manifest_data.get("test", {}).get("runs", []),
            }
        else:
            raise ValueError(f"Invalid manifest format at {m_path}: 'splits' field is missing or not a dict.")

    # Normalize split keys ("val" -> "validation")
    splits: dict[str, list[str]] = {
        "train": list(raw_splits.get("train", [])),
        "validation": list(raw_splits.get("validation", raw_splits.get("val", []))),
        "test": list(raw_splits.get("test", [])),
    }

    # Verify train split is non-empty
    if not splits["train"]:
        raise ValueError(f"Manifest at {m_path} has an empty 'train' split. Refusing to compile.")

    # Strict Leakage Checks (Disjointness Verification across runs/subjects)
    train_set = set(splits["train"])
    val_set = set(splits["validation"])
    test_set = set(splits["test"])

    train_val_leak = train_set & val_set
    train_test_leak = train_set & test_set
    val_test_leak = val_set & test_set

    if train_val_leak or train_test_leak or val_test_leak:
        raise ValueError(
            f"CRITICAL: Data leakage detected across splits in manifest {m_path}!\n"
            f"  - Train & Validation overlap: {train_val_leak or 'None'}\n"
            f"  - Train & Test overlap: {train_test_leak or 'None'}\n"
            f"  - Validation & Test overlap: {val_test_leak or 'None'}\n"
            "All splits must be mutually disjoint by recording/run/subject. Compilation aborted."
        )

    # Subject Disjointness Verification (when group_by='subject')
    if manifest_data.get("group_by") == "subject":
        train_subjs = set(manifest_data.get("train", {}).get("subjects", []))
        val_subjs = set(manifest_data.get("validation", {}).get("subjects", []))
        test_subjs = set(manifest_data.get("test", {}).get("subjects", []))
        subj_leak = (train_subjs & val_subjs) or (train_subjs & test_subjs) or (val_subjs & test_subjs)
        if subj_leak:
            raise ValueError(
                f"CRITICAL: Subject leakage detected under group_by='subject' in manifest {m_path}!\n"
                f"  - Overlapping subjects: {subj_leak}\n"
                "All splits must be mutually disjoint by subject. Compilation aborted."
            )


    # Verify all referenced runs exist on disk
    missing_train = [r for r in splits["train"] if r not in file_map]
    missing_val = [r for r in splits["validation"] if r not in file_map]
    missing_test = [r for r in splits["test"] if r not in file_map]
    total_missing = missing_train + missing_val + missing_test
    if total_missing:
        raise FileNotFoundError(
            f"Manifest {m_path} references {len(total_missing)} run(s) not found in {seq_dir}:\n"
            f"  Missing: {total_missing[:10]}{'...' if len(total_missing) > 10 else ''}\n"
            "Every run in the manifest must exist in the sequences directory. Compilation aborted."
        )

    split_counts: dict[str, int] = {}
    total_samples = 0

    # 3. Process each split
    for split_name in ("train", "validation", "test"):
        split_ids = splits.get(split_name, [])
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        features_windows: list[np.ndarray] = []
        labels_list: list[dict[str, Any]] = []

        generator = TemporalSequenceGenerator(window_size=window_size)
        global_sample_idx = 0
        for run_id in split_ids:
            run_files = run_to_files.get(run_id, [file_map[run_id]] if run_id in file_map else [])
            if not run_files:
                continue

            for npz_path in run_files:
                with np.load(npz_path, allow_pickle=True) as data:
                    if "X" in data:
                        run_X = data["X"]
                        run_verbs = data["verbs"]
                        run_objects = data["objects"]
                        run_targets = data["targets"]
                        meta_raw = data.get("metadata")
                        if meta_raw is not None:
                            if isinstance(meta_raw, np.ndarray):
                                meta_raw = meta_raw.item()
                            run_metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else list(meta_raw)
                        else:
                            run_metadata = []
                            for idx_w in range(len(run_X)):
                                run_metadata.append({
                                    "sample_idx": global_sample_idx + idx_w,
                                    "sequence_id": f"EXP001_{run_id}_CAM01_{global_sample_idx + idx_w + 1:06d}",
                                    "run_id": run_id,
                                    "subject_id": "ASTRONAUT-01",
                                    "video_id": f"EXP001_{run_id}_CAM01",
                                    "start_frame": idx_w,
                                    "end_frame": idx_w + window_size - 1,
                                    "verb": int(run_verbs[idx_w]),
                                    "object": int(run_objects[idx_w]),
                                    "target": int(run_targets[idx_w]),
                                })

                        for idx_w in range(len(run_X)):
                            features_windows.append(run_X[idx_w])
                            item = dict(run_metadata[idx_w])
                            item["sample_idx"] = global_sample_idx
                            labels_list.append(item)
                            global_sample_idx += 1
                    else:
                        features = data["features"]  # (T, 26)
                        verbs = data["verbs"]        # (T,)
                        objects = data["objects"]    # (T,)
                        targets = data["targets"]    # (T,)

                        t_total = len(features)
                        if t_total < window_size:
                            continue

                        gen_seqs = generator.generate_sequences(
                            features=features.astype(np.float32),
                            verbs=verbs,
                            objects=objects,
                            targets=targets,
                            run_id=run_id,
                        )
                        for idx_w in range(gen_seqs.num_sequences):
                            features_windows.append(gen_seqs.X[idx_w])
                            item = dict(gen_seqs.sample_metadata[idx_w])
                            item["sample_idx"] = global_sample_idx
                            labels_list.append(item)
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
    parser.add_argument("--manifest", type=str, required=True, help="Path to leakage-safe dataset split manifest JSON (REQUIRED)")
    parser.add_argument("--window-size", type=int, default=30, help="Sliding window size (frames)")
    args = parser.parse_args()

    result = compile_packed_dataset(
        sequences_dir=args.sequences_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        window_size=args.window_size,
    )
    print(json.dumps(result, indent=2))
