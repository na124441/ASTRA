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

from ml.datasets.schemas import (
    ActionSegmentAnnotation,
    OBJECT_TO_IDX,
    RecordingMetadata,
    TARGET_TO_IDX,
    VERB_TO_IDX,
    VIOLATION_VOCAB,
)


def build_aligned_sequence(
    feature_npz_path: str | Path,
    annotation_json_path: str | Path | None,
    output_npz_path: str | Path,
) -> dict[str, Any]:
    """
    Combines continuous 26-D features with frame-level action labels.
    """
    f_path = Path(feature_npz_path)
    out_path = Path(output_npz_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(f_path, allow_pickle=True) as data:
        features = data["features"]  # (T, 26)
        timestamps = data["timestamps"] if "timestamps" in data else np.arange(len(features)) / 30.0

    n_frames = len(features)
    verb_indices = np.full(n_frames, VERB_TO_IDX["IDLE"], dtype=np.int64)
    object_indices = np.full(n_frames, OBJECT_TO_IDX["NONE"], dtype=np.int64)
    target_indices = np.full(n_frames, TARGET_TO_IDX["NONE"], dtype=np.int64)
    violation_indices = np.zeros(n_frames, dtype=np.int64)

    # Load annotations if provided
    segments: list[dict[str, Any]] = []
    if annotation_json_path and Path(annotation_json_path).exists():
        with open(annotation_json_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
            segments = meta_dict.get("segments", [])

    for seg in segments:
        s_frame = seg.get("start_frame")
        e_frame = seg.get("end_frame")
        s_time = seg.get("start_time", 0.0)
        e_time = seg.get("end_time", 0.0)

        # Resolve frame indices from timestamps if frame numbers not set
        if s_frame is None or e_frame is None or s_frame == 0 and e_frame == 0:
            s_idx = int(s_time * 30.0)
            e_idx = int(e_time * 30.0)
        else:
            s_idx = int(s_frame)
            e_idx = int(e_frame)

        s_idx = max(0, min(n_frames - 1, s_idx))
        e_idx = max(s_idx, min(n_frames, e_idx + 1))

        v_name = seg.get("verb", "IDLE")
        o_name = seg.get("object") or "NONE"
        t_name = seg.get("target") or "NONE"
        vi_name = seg.get("violation_type", "NONE")

        v_code = VERB_TO_IDX.get(v_name, VERB_TO_IDX["UNKNOWN"])
        o_code = OBJECT_TO_IDX.get(o_name, OBJECT_TO_IDX["NONE"])
        t_code = TARGET_TO_IDX.get(t_name, TARGET_TO_IDX["NONE"])
        vi_code = VIOLATION_VOCAB.index(vi_name) if vi_name in VIOLATION_VOCAB else 0

        verb_indices[s_idx:e_idx] = v_code
        object_indices[s_idx:e_idx] = o_code
        target_indices[s_idx:e_idx] = t_code
        violation_indices[s_idx:e_idx] = vi_code

    # Save aligned supervised sequence
    np.savez_compressed(
        out_path,
        features=features.astype(np.float32),
        verbs=verb_indices,
        objects=object_indices,
        targets=target_indices,
        violations=violation_indices,
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
    args = parser.parse_args()

    feat_dir = Path(args.features_dir)
    annot_dir = Path(args.annotations_dir) if args.annotations_dir else feat_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(feat_dir.glob("*.npz")))
    print(f"\nProcessing {len(npz_files)} feature files into supervised temporal training sequences...")

    for idx, f in enumerate(npz_files, start=1):
        stem = f.stem
        # Look for matching annotation file: {stem}_meta.json, {stem}.json, or annotations/{stem}.json
        annot_file = annot_dir / f"{stem}_meta.json"
        if not annot_file.exists():
            annot_file = annot_dir / f"{stem}.json"
        if not annot_file.exists():
            annot_file = None

        out_npz = out_dir / f"{stem}.npz"
        res = build_aligned_sequence(f, annot_file, out_npz)
        print(f"[{idx}/{len(npz_files)}] {stem} -> {res['total_frames']} frames, {res['segments_applied']} segments")

    print(f"\nSequence generation complete. Ready for split generation: {out_dir}\n")


if __name__ == "__main__":
    main()
