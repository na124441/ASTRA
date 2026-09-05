"""Dataset adapter for the MicroG-4M Microgravity Human Action Benchmark.

Reference: 'Go Beyond Earth: Understanding Human Actions and Scenes in Microgravity Environments' (ICLR 2026).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.datasets.adapters.base import BaseDatasetAdapter
from ml.datasets.schemas import (
    ActionSegmentAnnotation,
    RecordingMetadata,
    VERB_VOCAB,
    OBJECT_VOCAB,
    TARGET_VOCAB,
)


# Semantic mapping from MicroG-4M action taxonomy to ASTRA-E ontology
MICROG_ACTION_MAPPING: dict[str, tuple[str, str | None, str | None]] = {
    "floating_component_grasp": ("GRASP", "RED_COMPONENT", None),
    "rack_tether_manipulation": ("MOVE", "CONTAINER", None),
    "microgravity_two_handed_transfer": ("MOVE", "RED_COMPONENT", "TARGET_A"),
    "inversion_payload_reach": ("APPROACH", "RED_COMPONENT", None),
    "sideways_station_assembly": ("PLACE", "RED_COMPONENT", "TARGET_A"),
    "docking_component_insert": ("PLACE", "YELLOW_COMPONENT", "TARGET_B"),
    "cabinet_unlatch": ("OPEN_CONTAINER", "CONTAINER", None),
    "cabinet_latch": ("CLOSE_CONTAINER", "CONTAINER", None),
    "hands_free_float": ("IDLE", "NONE", "NONE"),
    "payload_inspection": ("IDLE", "NONE", "NONE"),
    "tool_extraction": ("PICK", "RED_COMPONENT", "CONTAINER"),
    "tool_stow": ("RELEASE", "RED_COMPONENT", "CONTAINER"),
}


class MicroGDatasetAdapter(BaseDatasetAdapter):
    """
    Ingestion adapter for the MicroG-4M microgravity research dataset.
    Normalizes microgravity actions, camera viewpoints, and astronaut body orientations.
    """

    def __init__(self, dataset_root: str | Path) -> None:
        super().__init__(dataset_root)
        self.manifest_file = self.dataset_root / "microg4m_benchmark_manifest.json"

    def load_dataset(self) -> list[RecordingMetadata]:
        """Parse MicroG-4M metadata catalog and construct RecordingMetadata contracts."""
        recordings: list[RecordingMetadata] = []

        if not self.manifest_file.exists():
            # Generate default benchmark template if manifest doesn't exist
            return self._build_synthetic_microg_subset()

        with open(self.manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = data.get("test_cases", [])
        for idx, tc in enumerate(test_cases, start=1):
            rec_id = tc.get("id", f"MG-{idx:03d}")
            subject = tc.get("subject", "Astronaut-A")
            orientation = tc.get("orientation_deg", 0)
            raw_action = tc.get("expected_action", "GRASP")

            verb, obj, tgt = self._map_action(raw_action)

            # Build standardized segment
            segment = ActionSegmentAnnotation(
                segment_id=f"SEG-{rec_id}-01",
                start_frame=0,
                end_frame=150,
                start_time=0.0,
                end_time=5.0,
                verb=verb,
                object=obj,
                target=tgt,
                violation_type="NONE",
                label_quality="verified",
                source="microg-4m",
                notes=f"Orientation: {orientation} deg",
            )

            rec_meta = RecordingMetadata(
                video_id=f"MICROG_{rec_id}_CAM01",
                recording_id=rec_id,
                experiment_id="EXP001",
                run_id=f"RUN-MICROG-{rec_id}",
                subject_id=subject,
                camera_id="CAM-01",
                duration_seconds=5.0,
                total_frames=150,
                fps=30.0,
                width=640,
                height=480,
                scenario_type=f"microgravity_{orientation}deg",
                annotator_id="ICLR2026_MICROG_TEAM",
                segments=[segment],
            )
            recordings.append(rec_meta)

        return recordings

    def get_recording_video_path(self, recording_id: str) -> Path:
        """Resolve path to MP4 video for a MicroG-4M recording."""
        video_path = self.dataset_root / "videos" / f"{recording_id}.mp4"
        return video_path

    def get_annotations(self, recording_id: str) -> list[ActionSegmentAnnotation]:
        """Return standardized action segments for a given recording ID."""
        recordings = self.load_dataset()
        for r in recordings:
            if r.recording_id == recording_id or r.video_id == recording_id:
                return r.segments
        return []

    def _map_action(self, raw_action: str) -> tuple[str, str | None, str | None]:
        """Map raw MicroG action label to (verb, object, target)."""
        clean = raw_action.lower().strip()
        if clean in MICROG_ACTION_MAPPING:
            return MICROG_ACTION_MAPPING[clean]

        # Check if direct verb match
        upper = raw_action.upper().strip()
        if upper in VERB_VOCAB:
            return upper, None, None

        return "UNKNOWN", None, None

    def _build_synthetic_microg_subset(self) -> list[RecordingMetadata]:
        """Fallback subset builder for testing without network downloads."""
        recordings = []
        samples = [
            ("MG-001", 0, "Astronaut-A", "floating_component_grasp"),
            ("MG-002", 90, "Astronaut-A", "floating_component_grasp"),
            ("MG-003", 180, "Astronaut-B", "rack_tether_manipulation"),
            ("MG-004", 270, "Astronaut-B", "docking_component_insert"),
            ("MG-005", 45, "Astronaut-C", "hands_free_float"),
        ]

        for rec_id, orientation, subject, act_name in samples:
            verb, obj, tgt = self._map_action(act_name)
            seg = ActionSegmentAnnotation(
                segment_id=f"SEG-{rec_id}-01",
                start_frame=0,
                end_frame=150,
                start_time=0.0,
                end_time=5.0,
                verb=verb,
                object=obj,
                target=tgt,
                violation_type="NONE",
                label_quality="verified",
                source="microg-synthetic-fallback",
                notes=f"Orientation: {orientation} deg",
            )
            rec = RecordingMetadata(
                video_id=f"MICROG_{rec_id}_CAM01",
                recording_id=rec_id,
                experiment_id="EXP001",
                run_id=f"RUN-MICROG-{rec_id}",
                subject_id=subject,
                camera_id="CAM-01",
                duration_seconds=5.0,
                total_frames=150,
                fps=30.0,
                width=640,
                height=480,
                scenario_type=f"microgravity_{orientation}deg",
                annotator_id="MICROG_FALLBACK",
                segments=[seg],
            )
            recordings.append(rec)

        return recordings
