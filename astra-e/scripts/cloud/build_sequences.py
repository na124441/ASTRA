"""Cloud Sequence Builder: Merges extracted 26-D feature streams with ground-truth action segment annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.datasets.schemas import WINDOW_SIZE
from ml.datasets.sequence_generator import (
    TemporalSequenceGenerator,
    align_segments_to_frame_labels,
)


def build_aligned_sequence(
    feature_npz_path: str | Path,
    annotation_json_path: str | Path | None,
    output_npz_path: str | Path,
    output_format: str = "aligned_stream",
    window_size: int = WINDOW_SIZE,
) -> dict[str, Any]:
    """
    Combines continuous 26-D features with frame-level action labels.
    Delegates alignment and sequence generation to TemporalSequenceGenerator.
    """
    f_path = Path(feature_npz_path)
    out_path = Path(output_npz_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(f_path, allow_pickle=True) as data:
        if "features" not in data:
            raise ValueError(f"Missing 'features' array in NPZ: {f_path}")
        features = data["features"].astype(np.float32)
        timestamps = data["timestamps"] if "timestamps" in data else np.arange(len(features)) / 30.0

    n_frames = len(features)

    # Load annotations if provided
    segments: list[dict[str, Any]] = []
    meta_dict: dict[str, Any] = {}
    if annotation_json_path and Path(annotation_json_path).exists():
        with open(annotation_json_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
            segments = meta_dict.get("segments", [])

    # Strict alignment with conflict detection via Phase 2.7 sequence generator
    verbs, objects, targets, violations = align_segments_to_frame_labels(
        segments=segments,
        total_frames=n_frames,
    )

    if output_format == "windows":
        generator = TemporalSequenceGenerator(window_size=window_size)
        gen_seqs = generator.generate_sequences(
            features=features,
            verbs=verbs,
            objects=objects,
            targets=targets,
            recording_meta=meta_dict,
            run_id=out_path.stem,
        )
        np.savez_compressed(
            out_path,
            X=gen_seqs.X,
            verbs=gen_seqs.verbs,
            objects=gen_seqs.objects,
            targets=gen_seqs.targets,
            metadata=json.dumps(gen_seqs.sample_metadata),
            total_frames=n_frames,
            window_size=window_size,
        )
        return {
            "sequence_id": out_path.stem,
            "total_frames": n_frames,
            "num_sequences": gen_seqs.num_sequences,
            "segments_applied": len(segments),
            "output_file": str(out_path),
        }

    # Default: save aligned continuous stream [T, 26]
    np.savez_compressed(
        out_path,
        features=features,
        verbs=verbs,
        objects=objects,
        targets=targets,
        violations=violations,
        timestamps=timestamps.astype(np.float32),
    )

    return {
        "sequence_id": out_path.stem,
        "total_frames": n_frames,
        "segments_applied": len(segments),
        "output_file": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA-E Sequence Builder")
    parser.add_argument("--features-dir", required=True, help="Directory containing extracted 26-D .npz files")
    parser.add_argument("--annotations-dir", default=None, help="Directory containing annotation JSON files")
    parser.add_argument("--output-dir", default="data/cloud/sequences", help="Output directory for supervised sequence NPZs")
    parser.add_argument(
        "--output-format",
        choices=["aligned_stream", "windows"],
        default="aligned_stream",
        help="Output format: 'aligned_stream' (continuous [T, 26]) or 'windows' (causal [N, 30, 26])",
    )
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Sliding window size for windows format")
    args = parser.parse_args()

    feat_dir = Path(args.features_dir)
    annot_dir = Path(args.annotations_dir) if args.annotations_dir else feat_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(feat_dir.glob("*.npz")))
    print(f"\nProcessing {len(npz_files)} feature files into supervised temporal training sequences (format={args.output_format})...")

    for idx, f in enumerate(npz_files, start=1):
        stem = f.stem
        # Look for matching annotation file: {stem}_meta.json, {stem}.json, or annotations/{stem}.json
        annot_file = annot_dir / f"{stem}_meta.json"
        if not annot_file.exists():
            annot_file = annot_dir / f"{stem}.json"
        if not annot_file.exists():
            annot_file = None

        out_npz = out_dir / f"{stem}.npz"
        res = build_aligned_sequence(
            f,
            annot_file,
            out_npz,
            output_format=args.output_format,
            window_size=args.window_size,
        )
        print(f"[{idx}/{len(npz_files)}] {stem} -> {res['total_frames']} frames, {res['segments_applied']} segments")

    print(f"\nSequence generation complete. Ready for split generation: {out_dir}\n")


if __name__ == "__main__":
    main()
