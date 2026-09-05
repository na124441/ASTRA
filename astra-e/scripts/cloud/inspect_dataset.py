"""Cloud Dataset Inspector: Audits video archives, sequence files, and annotation manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.datasets.schemas import VERB_VOCAB, OBJECT_VOCAB, TARGET_VOCAB, VIOLATION_VOCAB


def inspect_dataset(data_dir: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Inspect dataset directory containing videos, NPZ sequences, or annotation JSONs."""
    p_dir = Path(data_dir)
    print("\n" + "=" * 76)
    print("║" + "ASTRA-E CLOUD DATASET AUDITOR".center(74) + "║")
    print("=" * 76)
    print(f"  Target Directory: {p_dir}")

    if not p_dir.exists():
        print(f"  [ERROR] Directory not found: {p_dir}")
        return {}

    # Scan files
    mp4_files = list(p_dir.glob("*.mp4")) + list(p_dir.glob("**/*.mp4"))
    npz_files = list(p_dir.glob("*.npz")) + list(p_dir.glob("**/*.npz"))
    json_files = list(p_dir.glob("*meta.json")) + list(p_dir.glob("**/*meta.json"))

    print(f"  Raw Video Files (.mp4):      {len(mp4_files)}")
    print(f"  Extracted Sequences (.npz):   {len(npz_files)}")
    print(f"  Metadata Records (.json):     {len(json_files)}")

    # Parse NPZ sequences if available
    total_frames = 0
    verb_counts = {v: 0 for v in VERB_VOCAB}
    obj_counts = {o: 0 for o in OBJECT_VOCAB}
    tgt_counts = {t: 0 for t in TARGET_VOCAB}
    viol_counts = {vi: 0 for vi in VIOLATION_VOCAB}

    import numpy as np
    from ml.datasets.schemas import IDX_TO_VERB, IDX_TO_OBJECT, IDX_TO_TARGET

    for npz_p in npz_files:
        try:
            with np.load(npz_p, allow_pickle=True) as d:
                feats = d["features"]
                total_frames += len(feats)
                if "verbs" in d:
                    for v_idx in d["verbs"]:
                        name = IDX_TO_VERB.get(int(v_idx), "UNKNOWN")
                        verb_counts[name] = verb_counts.get(name, 0) + 1
                if "objects" in d:
                    for o_idx in d["objects"]:
                        name = IDX_TO_OBJECT.get(int(o_idx), "NONE")
                        obj_counts[name] = obj_counts.get(name, 0) + 1
                if "targets" in d:
                    for t_idx in d["targets"]:
                        name = IDX_TO_TARGET.get(int(t_idx), "NONE")
                        tgt_counts[name] = tgt_counts.get(name, 0) + 1
                if "violations" in d:
                    for vi_idx in d["violations"]:
                        vi_name = VIOLATION_VOCAB[int(vi_idx)] if int(vi_idx) < len(VIOLATION_VOCAB) else "UNKNOWN"
                        viol_counts[vi_name] = viol_counts.get(vi_name, 0) + 1
        except Exception as e:
            print(f"  [WARN] Could not read {npz_p.name}: {e}")

    duration_sec = total_frames / 30.0 if total_frames > 0 else 0.0

    print("-" * 76)
    print("DATASET AGGREGATE SUMMARY:")
    print(f"  Total Processed Frames:     {total_frames:,}")
    print(f"  Total Temporal Duration:     {duration_sec:.1f} seconds ({duration_sec / 60.0:.2f} mins)")
    print(f"  Causal 30-Frame Windows:     {max(0, total_frames - (29 * len(npz_files))):,}")

    # Display Action Verb Distribution
    print("\nACTION VERB DISTRIBUTION:")
    print(f"  {'Verb':<20} | {'Frames':<12} | {'Percentage'}")
    print(f"  {'-'*20}-+-{'-'*12}-+-----------")
    for verb in VERB_VOCAB:
        count = verb_counts.get(verb, 0)
        pct = (count / max(1, total_frames)) * 100.0
        print(f"  {verb:<20} | {count:<12,} | {pct:>8.2f}%")

    # Display Object Distribution
    print("\nOBJECT DISTRIBUTION:")
    for obj in OBJECT_VOCAB:
        count = obj_counts.get(obj, 0)
        pct = (count / max(1, total_frames)) * 100.0
        print(f"  {obj:<20} | {count:<12,} | {pct:>8.2f}%")

    # Display Violations (if present)
    non_zero_viols = {k: v for k, v in viol_counts.items() if v > 0 and k != "NONE"}
    if non_zero_viols:
        print("\nPROCEDURAL VIOLATIONS DETECTED IN GROUND TRUTH:")
        for v_name, count in non_zero_viols.items():
            print(f"  ⚠️  {v_name:<20}: {count:,} frames")

    print("=" * 76 + "\n")

    return {
        "mp4_count": len(mp4_files),
        "npz_count": len(npz_files),
        "total_frames": total_frames,
        "duration_seconds": duration_sec,
        "verb_distribution": verb_counts,
        "object_distribution": obj_counts,
        "target_distribution": tgt_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA-E Cloud Dataset Inspector")
    parser.add_argument("--data-dir", default="data/processed/EXP001", help="Path to video or sequence directory")
    parser.add_argument("--manifest", default=None, help="Optional dataset_manifest.json")
    args = parser.parse_args()

    inspect_dataset(args.data_dir, args.manifest)


if __name__ == "__main__":
    main()
