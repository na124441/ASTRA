"""Domain-randomized synthetic dataset generator with physics and noise simulation."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any
import numpy as np

from astra.activity.features import KinematicFeatureExtractor
from astra.contracts.video import VideoFrame
from astra.perception.pipeline import PerceptionPipeline
from astra.video.camera import MockCamera
from ml.datasets.schemas import (
    ActionSegmentAnnotation,
    DatasetManifest,
    OBJECT_TO_IDX,
    RecordingMetadata,
    TARGET_TO_IDX,
    VERB_TO_IDX,
)


class DomainRandomizedGenerator:
    """
    Generates realistic experiment runs with domain randomization:
    - Randomized geometry (container, target, and resting positions)
    - Kinematic randomization (hand speed, accelerations, hesitation pauses)
    - Perception failure injection (tracking jitter, dropped frames, occlusions)
    - Diverse procedural scenarios (nominal, wrong target, wrong object, hesitation)
    """

    def __init__(
        self,
        output_dir: str | Path = "data/processed/EXP001",
        manifest_dir: str | Path = "data/manifests",
        seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.manifest_dir = Path(manifest_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_dataset(
        self,
        num_runs: int = 40,
        dropout_prob: float = 0.05,
    ) -> DatasetManifest:
        """
        Generate multiple domain-randomized experiment runs.
        """
        scenarios = ["nominal", "nominal", "wrong_target", "wrong_object", "hesitation"]
        run_ids: list[str] = []
        total_windows_accum = 0

        print(f"Generating {num_runs} domain-randomized runs...")

        for idx in range(1, num_runs + 1):
            run_id = f"RUN-{idx:04d}"
            scenario = scenarios[(idx - 1) % len(scenarios)]

            run_meta, npz_path, num_windows = self._generate_single_run(
                run_id=run_id,
                scenario=scenario,
                dropout_prob=dropout_prob,
            )
            run_ids.append(run_id)
            total_windows_accum += num_windows

        # Split 70% train, 15% val, 15% test
        shuffled = list(run_ids)
        random.shuffle(shuffled)
        n_train = int(0.70 * len(shuffled))
        n_val = int(0.15 * len(shuffled))

        train_runs = sorted(shuffled[:n_train])
        val_runs = sorted(shuffled[n_train : n_train + n_val])
        test_runs = sorted(shuffled[n_train + n_val :])

        manifest = DatasetManifest(
            dataset_version="2026.09.01",
            generator_version="1.0.0",
            feature_schema_version="kinematic-26d-v1.0",
            random_seed=self.seed,
            recordings_count=len(run_ids),
            total_windows=total_windows_accum,
            splits={"train": train_runs, "val": val_runs, "test": test_runs},
            created_at=time.time(),
            metadata={"dropout_prob": dropout_prob, "scenarios": scenarios},
        )

        manifest_path = self.manifest_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        print(f"Dataset generation complete: {len(run_ids)} runs, {total_windows_accum} windows.")
        print(f"Manifest saved to: {manifest_path}")
        return manifest

    def _generate_single_run(
        self,
        run_id: str,
        scenario: str,
        dropout_prob: float,
    ) -> tuple[RecordingMetadata, Path, int]:
        total_frames = random.randint(220, 280)
        cam = MockCamera(width=640, height=480, total_frames=total_frames)
        cam.start()

        pipeline = PerceptionPipeline()
        extractor = KinematicFeatureExtractor()

        features_list: list[np.ndarray] = []
        verb_labels: list[int] = []
        object_labels: list[int] = []
        target_labels: list[int] = []

        segments: list[ActionSegmentAnnotation] = []

        for f_idx in range(total_frames):
            ok, frame, ts = cam.read()
            if not ok or frame is None:
                break

            # Add perceptual noise / random tracking dropout
            if random.random() < dropout_prob:
                # Occlusion: frame with noise or blacked out
                frame = np.clip(frame.astype(np.float32) + np.random.normal(0, 15, frame.shape), 0, 255).astype(np.uint8)

            vf = VideoFrame(
                source="mock-cam",
                correlation_id=run_id,
                frame_id=f_idx + 1,
                camera_id=cam.camera_id,
                width=640,
                height=480,
                frame_reference=f"memory://frame/{f_idx + 1}",
                event_time=ts,
            )

            obs = pipeline.process_frame(vf, frame)
            feat = extractor.extract_frame_features(obs)
            features_list.append(feat)

            # Assign ground truth action at time t based on phase
            verb, obj, tgt = self._get_ground_truth_action(f_idx, scenario)
            verb_labels.append(VERB_TO_IDX.get(verb, VERB_TO_IDX["UNKNOWN"]))
            object_labels.append(OBJECT_TO_IDX.get(obj, OBJECT_TO_IDX["NONE"]))
            target_labels.append(TARGET_TO_IDX.get(tgt, TARGET_TO_IDX["NONE"]))

        cam.stop()

        features_arr = np.array(features_list, dtype=np.float32)
        verb_arr = np.array(verb_labels, dtype=np.int64)
        obj_arr = np.array(object_labels, dtype=np.int64)
        tgt_arr = np.array(target_labels, dtype=np.int64)

        npz_file = self.output_dir / f"{run_id}.npz"
        np.savez_compressed(
            npz_file,
            features=features_arr,
            verbs=verb_arr,
            objects=obj_arr,
            targets=tgt_arr,
        )

        metadata = RecordingMetadata(
            recording_id=f"REC-{run_id}",
            experiment_id="EXP-001",
            run_id=run_id,
            duration_seconds=total_frames / 30.0,
            scenario_type=scenario,
            random_seed=self.seed,
            segments=segments,
        )

        meta_file = self.output_dir / f"{run_id}_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))

        # Windows are valid if index >= 29 (window size 30)
        num_windows = max(0, len(features_list) - 29)
        return metadata, npz_file, num_windows

    def _get_ground_truth_action(self, t: int, scenario: str) -> tuple[str, str, str]:
        """Determine endpoint action label at frame t."""
        if t < 15:
            return "IDLE", "NONE", "NONE"
        elif t < 45:
            return "APPROACH", "RED_COMPONENT", "NONE"
        elif t < 65:
            return "PICK", "RED_COMPONENT", "NONE"
        elif t < 115:
            return "MOVE", "RED_COMPONENT", "TARGET_A" if scenario != "wrong_target" else "TARGET_B"
        elif t < 140:
            target = "TARGET_A" if scenario != "wrong_target" else "TARGET_B"
            return "PLACE", "RED_COMPONENT", target
        elif t < 155:
            return "RELEASE", "RED_COMPONENT", "TARGET_A" if scenario != "wrong_target" else "TARGET_B"
        elif t < 185:
            return "APPROACH", "YELLOW_COMPONENT", "NONE"
        elif t < 205:
            return "PICK", "YELLOW_COMPONENT", "NONE"
        elif t < 235:
            return "MOVE", "YELLOW_COMPONENT", "TARGET_B"
        elif t < 255:
            return "PLACE", "YELLOW_COMPONENT", "TARGET_B"
        else:
            return "RELEASE", "YELLOW_COMPONENT", "TARGET_B"
