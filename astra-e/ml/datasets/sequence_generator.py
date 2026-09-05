"""ASTRA-E Phase 2.7: Causal Temporal Sequence Generation Engine.

Converts run-level continuous frame feature streams [T, 26] and segment annotations
into causal sliding-window training samples [N, 30, 26] with endpoint multi-head supervision.
Strict physical causality: X_i = F[i : i + 30], y_i = Y[i + 29].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from ml.datasets.schemas import (
    ActionSegmentAnnotation,
    NUM_FEATURES,
    OBJECT_TO_IDX,
    OBJECT_VOCAB,
    RecordingMetadata,
    TARGET_TO_IDX,
    TARGET_VOCAB,
    VERB_TO_IDX,
    VERB_VOCAB,
    VIOLATION_VOCAB,
    WINDOW_SIZE,
)


class SequenceGenerationError(ValueError):
    """Base exception for sequence generation errors."""
    pass


class AnnotationConflictError(SequenceGenerationError):
    """Raised when segment annotations overlap with contradictory labels."""
    pass


class SequenceValidationError(SequenceGenerationError):
    """Raised when feature dimensions, types, or boundaries violate contracts."""
    pass


@dataclass(frozen=True)
class GeneratedSequences:
    """Container holding generated causal sliding-window arrays and provenance metadata."""
    X: np.ndarray             # [N, 30, 26] float32
    verbs: np.ndarray         # [N] int64
    objects: np.ndarray       # [N] int64
    targets: np.ndarray       # [N] int64
    sample_metadata: list[dict[str, Any]]
    run_id: str
    video_id: str
    subject_id: str
    total_frames: int
    window_size: int
    num_sequences: int


def align_segments_to_frame_labels(
    segments: Sequence[ActionSegmentAnnotation | dict[str, Any]],
    total_frames: int,
    default_verb: str = "IDLE",
    default_object: str = "NONE",
    default_target: str = "NONE",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts interval-based ActionSegmentAnnotations into frame-level label arrays.

    Fails closed on:
      - start_frame < 0
      - end_frame < start_frame
      - end_frame >= total_frames
      - Overlapping conflicting segments with contradictory labels

    Unannotated frames default deterministically to:
      verb=default_verb ("IDLE"), object=default_object ("NONE"), target=default_target ("NONE").

    Returns:
      (verbs, objects, targets, violations) as 1D numpy arrays of length total_frames.
    """
    if total_frames <= 0:
        raise SequenceValidationError(f"total_frames must be > 0, got {total_frames}")

    verbs = np.full(total_frames, VERB_TO_IDX[default_verb], dtype=np.int64)
    objects = np.full(total_frames, OBJECT_TO_IDX[default_object], dtype=np.int64)
    targets = np.full(total_frames, TARGET_TO_IDX[default_target], dtype=np.int64)
    violations = np.zeros(total_frames, dtype=np.int64)

    # Frame assignment tracking: frame_idx -> (verb_str, object_str, target_str, segment_id)
    assigned_frames: dict[int, tuple[str, str, str, str | None]] = {}

    for seg in segments:
        if isinstance(seg, dict):
            s_frame = seg.get("start_frame", 0)
            e_frame = seg.get("end_frame", 0)
            v_name = seg.get("verb", default_verb)
            o_name = seg.get("object") or default_object
            t_name = seg.get("target") or default_target
            vi_name = seg.get("violation_type", "NONE")
            seg_id = seg.get("segment_id")
        else:
            s_frame = seg.start_frame
            e_frame = seg.end_frame
            v_name = seg.verb
            o_name = seg.object or default_object
            t_name = seg.target or default_target
            vi_name = seg.violation_type
            seg_id = seg.segment_id

        # Boundary checks
        if s_frame < 0:
            raise SequenceValidationError(
                f"Invalid segment: start_frame ({s_frame}) must be >= 0 (segment: '{seg_id}')."
            )
        if e_frame < s_frame:
            raise SequenceValidationError(
                f"Invalid segment: end_frame ({e_frame}) < start_frame ({s_frame}) (segment: '{seg_id}')."
            )
        if e_frame >= total_frames:
            raise SequenceValidationError(
                f"Invalid segment: end_frame ({e_frame}) >= total_frames ({total_frames}) (segment: '{seg_id}')."
            )

        # Vocabulary verification
        if v_name not in VERB_TO_IDX:
            raise SequenceValidationError(f"Unknown action verb '{v_name}' not in VERB_VOCAB.")
        if o_name not in OBJECT_TO_IDX:
            raise SequenceValidationError(f"Unknown object '{o_name}' not in OBJECT_VOCAB.")
        if t_name not in TARGET_TO_IDX:
            raise SequenceValidationError(f"Unknown target '{t_name}' not in TARGET_VOCAB.")

        v_idx = VERB_TO_IDX[v_name]
        o_idx = OBJECT_TO_IDX[o_name]
        t_idx = TARGET_TO_IDX[t_name]
        vi_idx = VIOLATION_VOCAB.index(vi_name) if vi_name in VIOLATION_VOCAB else 0

        # Frame-level assignment with conflict detection
        for f in range(s_frame, e_frame + 1):
            if f in assigned_frames:
                prev_v, prev_o, prev_t, prev_seg_id = assigned_frames[f]
                if (prev_v, prev_o, prev_t) != (v_name, o_name, t_name):
                    raise AnnotationConflictError(
                        f"Conflicting overlapping annotations at frame {f}: "
                        f"'{prev_v}/{prev_o}/{prev_t}' (segment '{prev_seg_id}') vs "
                        f"'{v_name}/{o_name}/{t_name}' (segment '{seg_id}'). Fail-closed."
                    )
            assigned_frames[f] = (v_name, o_name, t_name, seg_id)
            verbs[f] = v_idx
            objects[f] = o_idx
            targets[f] = t_idx
            violations[f] = vi_idx

    return verbs, objects, targets, violations


class TemporalSequenceGenerator:
    """
    Phase 2.7 Temporal Sequence Generator.
    Slices run-level feature streams [T, 26] and frame labels into causal 30-frame windows.
    Zero future lookahead: X_i = F[i : i + 30], y_i = Y[i + 29].
    """

    def __init__(self, window_size: int = WINDOW_SIZE, stride: int = 1) -> None:
        if window_size < 1:
            raise SequenceValidationError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise SequenceValidationError(f"stride must be >= 1, got {stride}")
        self.window_size = window_size
        self.stride = stride

    def validate_run_stream(
        self,
        features: np.ndarray,
        verbs: np.ndarray,
        objects: np.ndarray,
        targets: np.ndarray,
    ) -> int:
        """
        Validates the run-level frame arrays against all contract invariants.
        Fails closed on any violation.
        """
        if not isinstance(features, np.ndarray):
            raise SequenceValidationError(f"features must be a numpy ndarray, got {type(features).__name__}")

        if features.ndim != 2:
            raise SequenceValidationError(
                f"features must be a 2D array of shape (T, 26), got ndim={features.ndim} (shape: {features.shape})"
            )

        if features.shape[1] != NUM_FEATURES:
            raise SequenceValidationError(
                f"features feature dimension must be {NUM_FEATURES}, got {features.shape[1]} (shape: {features.shape})"
            )

        if features.dtype != np.float32:
            raise SequenceValidationError(
                f"features dtype must be float32, got {features.dtype}"
            )

        T = len(features)
        if len(verbs) != T:
            raise SequenceValidationError(f"verbs length ({len(verbs)}) != features length ({T})")
        if len(objects) != T:
            raise SequenceValidationError(f"objects length ({len(objects)}) != features length ({T})")
        if len(targets) != T:
            raise SequenceValidationError(f"targets length ({len(targets)}) != features length ({T})")

        if T < self.window_size:
            raise SequenceValidationError(
                f"Recording length T={T} is shorter than minimum window_size={self.window_size}. Cannot generate sequences."
            )

        if np.isnan(features).any():
            nan_count = int(np.isnan(features).sum())
            raise SequenceValidationError(f"features matrix contains {nan_count} NaN values!")

        if np.isinf(features).any():
            inf_count = int(np.isinf(features).sum())
            raise SequenceValidationError(f"features matrix contains {inf_count} Inf values!")

        # Vocabulary range validation
        if np.any(verbs < 0) or np.any(verbs >= len(VERB_VOCAB)):
            raise SequenceValidationError(f"verbs contains IDs outside valid range [0, {len(VERB_VOCAB) - 1}]")
        if np.any(objects < 0) or np.any(objects >= len(OBJECT_VOCAB)):
            raise SequenceValidationError(f"objects contains IDs outside valid range [0, {len(OBJECT_VOCAB) - 1}]")
        if np.any(targets < 0) or np.any(targets >= len(TARGET_VOCAB)):
            raise SequenceValidationError(f"targets contains IDs outside valid range [0, {len(TARGET_VOCAB) - 1}]")

        return T

    def generate_sequences(
        self,
        features: np.ndarray,
        verbs: np.ndarray,
        objects: np.ndarray,
        targets: np.ndarray,
        recording_meta: RecordingMetadata | dict[str, Any] | None = None,
        run_id: str | None = None,
        video_id: str | None = None,
        subject_id: str | None = None,
        fps: float = 30.0,
    ) -> GeneratedSequences:
        """
        Slices validated run features and labels into causal 30-frame temporal samples.

        Output:
          X:       [N, 30, 26] float32
          verbs:   [N] int64
          objects: [N] int64
          targets: [N] int64
          where N = (T - window_size) // stride + 1.
        """
        T = self.validate_run_stream(features, verbs, objects, targets)

        # Extract provenance fields
        if recording_meta:
            if isinstance(recording_meta, dict):
                r_id = run_id or recording_meta.get("run_id") or "RUN-0001"
                v_id = video_id or recording_meta.get("video_id") or f"EXP001_{r_id}_CAM01"
                s_id = subject_id or recording_meta.get("subject_id") or "ASTRONAUT-01"
                fps_val = float(recording_meta.get("fps", fps))
            else:
                r_id = run_id or recording_meta.run_id
                v_id = video_id or recording_meta.video_id or f"EXP001_{r_id}_CAM01"
                s_id = subject_id or recording_meta.subject_id
                fps_val = float(recording_meta.fps)
        else:
            r_id = run_id or "RUN-0001"
            v_id = video_id or f"EXP001_{r_id}_CAM01"
            s_id = subject_id or "ASTRONAUT-01"
            fps_val = fps

        N = (T - self.window_size) // self.stride + 1

        X = np.empty((N, self.window_size, NUM_FEATURES), dtype=np.float32)
        y_verbs = np.empty(N, dtype=np.int64)
        y_objects = np.empty(N, dtype=np.int64)
        y_targets = np.empty(N, dtype=np.int64)
        metadata_list: list[dict[str, Any]] = []

        for idx, start_idx in enumerate(range(0, T - self.window_size + 1, self.stride)):
            end_idx = start_idx + self.window_size - 1
            label_frame = end_idx  # Strict causal endpoint: t_label = t_end

            # X_i = F[i : i + 30]
            X[idx] = features[start_idx : end_idx + 1]

            # y_i = Y[i + 29]
            y_verbs[idx] = verbs[label_frame]
            y_objects[idx] = objects[label_frame]
            y_targets[idx] = targets[label_frame]

            seq_id = f"{v_id}_{idx + 1:06d}"
            metadata_list.append({
                "sample_idx": idx,
                "sequence_id": seq_id,
                "run_id": r_id,
                "subject_id": s_id,
                "video_id": v_id,
                "start_frame": int(start_idx),
                "end_frame": int(end_idx),
                "label_frame": int(label_frame),
                "fps": float(fps_val),
                "verb": int(verbs[label_frame]),
                "object": int(objects[label_frame]),
                "target": int(targets[label_frame]),
            })

        return GeneratedSequences(
            X=X,
            verbs=y_verbs,
            objects=y_objects,
            targets=y_targets,
            sample_metadata=metadata_list,
            run_id=r_id,
            video_id=v_id,
            subject_id=s_id,
            total_frames=T,
            window_size=self.window_size,
            num_sequences=N,
        )

    def process_recording_to_npz(
        self,
        feature_npz_path: str | Path,
        annotation_json_path: str | Path | None,
        output_npz_path: str | Path,
    ) -> dict[str, Any]:
        """
        End-to-end processing of a single recording:
          1. Loads extracted continuous features [T, 26].
          2. Parses and validates segment annotations.
          3. Aligns segment intervals to frame-level labels.
          4. Slices into causal 30-frame temporal windows.
          5. Saves compressed [N, 30, 26] archive and releases memory.
        """
        f_path = Path(feature_npz_path)
        out_path = Path(output_npz_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not f_path.exists():
            raise FileNotFoundError(f"Feature archive does not exist: {f_path}")

        with np.load(f_path, allow_pickle=True) as data:
            if "features" not in data:
                raise SequenceValidationError(f"Missing 'features' in {f_path}")
            features = data["features"].astype(np.float32)
            timestamps = data["timestamps"] if "timestamps" in data else np.arange(len(features)) / 30.0

        T = len(features)

        # Parse annotations if provided
        segments: list[dict[str, Any]] = []
        rec_meta: dict[str, Any] = {}
        if annotation_json_path and Path(annotation_json_path).exists():
            with open(annotation_json_path, "r", encoding="utf-8") as f:
                rec_meta = json.load(f)
                segments = rec_meta.get("segments", [])

        verbs, objects, targets, violations = align_segments_to_frame_labels(
            segments=segments,
            total_frames=T,
        )

        gen_seqs = self.generate_sequences(
            features=features,
            verbs=verbs,
            objects=objects,
            targets=targets,
            recording_meta=rec_meta,
            run_id=f_path.stem,
        )

        np.savez_compressed(
            out_path,
            X=gen_seqs.X,
            verbs=gen_seqs.verbs,
            objects=gen_seqs.objects,
            targets=gen_seqs.targets,
            metadata=json.dumps(gen_seqs.sample_metadata),
            total_frames=T,
            window_size=self.window_size,
        )

        return {
            "run_id": gen_seqs.run_id,
            "total_frames": T,
            "num_sequences": gen_seqs.num_sequences,
            "shape": list(gen_seqs.X.shape),
            "output_file": str(out_path),
        }
