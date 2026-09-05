"""Cloud Split Generator: Partitions supervised sequence files into disjoint train/val/test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.datasets.schemas import WINDOW_SIZE, SplitManifest
from ml.datasets.splits import SplitManager, generate_leakage_safe_splits


def print_split_summary(manifest: SplitManifest, output_manifest: str | Path) -> None:
    """Pretty prints detailed partitioning statistics and auditing results."""
    print("\n" + "=" * 74)
    print("║" + "ASTRA-E LEAKAGE-SAFE DATASET SPLIT SUMMARY (PHASE 2.8)".center(72) + "║")
    print("=" * 74)
    print(f"  Schema Version:      {manifest.schema_version}")
    print(f"  Grouping Strategy:   {manifest.group_by.upper()}-DISJOINT")
    print(f"  Random Seed:         {manifest.seed}")
    print(f"  Configured Ratios:   Train={manifest.ratios['train']:.2f}, Val={manifest.ratios['validation']:.2f}, Test={manifest.ratios['test']:.2f}")
    print("-" * 74)

    total_groups = manifest.metadata.get("total_groups", 0)
    print(f"  Total Partition Groups: {total_groups}")
    print(f"  {'Partition':<12} | {'Subjects':<8} | {'Runs':<6} | {'Recordings':<10} | {'30-f Windows':<12}")
    print("  " + "-" * 62)

    for s_name in ("train", "validation", "test"):
        stats = manifest.statistics.get(s_name, {})
        n_subj = stats.get("num_subjects", 0)
        n_run = stats.get("num_runs", 0)
        n_rec = stats.get("num_recordings", 0)
        n_win = stats.get("num_windows", 0)
        print(f"  {s_name:<12} | {n_subj:<8} | {n_run:<6} | {n_rec:<10} | {n_win:<12,}")

    print("-" * 74)
    # Rare class warnings
    if manifest.rare_classes:
        print(f"  ⚠️  RARE-CLASS ALERTS ({len(manifest.rare_classes)} detected):")
        for rc in manifest.rare_classes[:5]:
            print(f"      - [{rc['category']}] '{rc['class_name']}': missing from {rc['missing_splits']}")
        if len(manifest.rare_classes) > 5:
            print(f"      ... and {len(manifest.rare_classes) - 5} more.")
    else:
        print("  ✓ All active vocabulary classes represented across all splits.")

    print("-" * 74)
    audit = manifest.disjointness_audit
    print(f"  Disjoint Runs:       {'✓ PASSED' if audit.get('mutually_disjoint_runs') else '❌ FAILED'}")
    print(f"  Disjoint Recordings: {'✓ PASSED' if audit.get('mutually_disjoint_recordings') else '❌ FAILED'}")
    if audit.get("subject_disjoint_enforced"):
        print(f"  Disjoint Subjects:   {'✓ PASSED (Subject-Disjoint Enforced)' if audit.get('mutually_disjoint_subjects') else '❌ FAILED'}")
    print(f"  Manifest Written:    {output_manifest}")
    print("=" * 74 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA-E Leakage-Safe Split Generator")
    parser.add_argument("--sequences-dir", default="data/processed/EXP001", help="Directory with processed .npz sequences")
    parser.add_argument("--metadata-dir", default=None, help="Directory containing metadata JSON files (default: sequences-dir)")
    parser.add_argument("--output-manifest", default="data/manifests/dataset_manifest.json", help="Target manifest JSON path")
    parser.add_argument("--group-by", choices=["subject", "run"], default="subject", help="Grouping level: 'subject' (preferred) or 'run'")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val-ratio", "--validation-ratio", dest="val_ratio", type=float, default=0.15, help="Val ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Sliding window size (default: 30)")
    args = parser.parse_args()

    manifest = generate_leakage_safe_splits(
        sequences_dir=args.sequences_dir,
        metadata_dir=args.metadata_dir,
        output_manifest=args.output_manifest,
        group_by=args.group_by,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        window_size=args.window_size,
    )

    print_split_summary(manifest, args.output_manifest)


if __name__ == "__main__":
    main()
