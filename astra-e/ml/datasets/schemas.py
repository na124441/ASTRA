"""Dataset schemas, annotation contracts, and vocabulary definitions for Phase 3 ML."""

from __future__ import annotations

import time
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

# Multi-Head classification vocabularies including UNKNOWN/Abstention
VERB_VOCAB = [
    "IDLE",
    "APPROACH",
    "TOUCH",
    "GRASP",
    "PICK",
    "MOVE",
    "PLACE",
    "RELEASE",
    "OPEN_CONTAINER",
    "CLOSE_CONTAINER",
    "UNKNOWN",
]

OBJECT_VOCAB = [
    "NONE",
    "RED_COMPONENT",
    "YELLOW_COMPONENT",
    "CONTAINER",
    "UNKNOWN",
]

TARGET_VOCAB = [
    "NONE",
    "TARGET_A",
    "TARGET_B",
    "CONTAINER",
    "UNKNOWN",
]

VERB_TO_IDX = {v: i for i, v in enumerate(VERB_VOCAB)}
OBJECT_TO_IDX = {o: i for i, o in enumerate(OBJECT_VOCAB)}
TARGET_TO_IDX = {t: i for i, t in enumerate(TARGET_VOCAB)}

IDX_TO_VERB = {i: v for i, v in enumerate(VERB_VOCAB)}
IDX_TO_OBJECT = {i: o for i, o in enumerate(OBJECT_VOCAB)}
IDX_TO_TARGET = {i: t for i, t in enumerate(TARGET_VOCAB)}

# Ground-truth procedural violation categories (evaluation benchmark only, not LSTM heads)
VIOLATION_VOCAB = [
    "NONE",
    "WRONG_OBJECT",
    "WRONG_TARGET",
    "SKIPPED_STEP",
    "REPEATED_STEP",
    "PREMATURE_CLOSE",
    "OUT_OF_SEQUENCE",
    "AMBIGUOUS",
]

LABEL_QUALITY_VOCAB = [
    "verified",
    "reviewed",
    "synthetic",
    "ambiguous",
]


class ActionSegmentAnnotation(BaseModel):
    """
    Ground-truth temporal action interval and procedural evaluation metadata.
    Used for multi-head action supervision, temporal boundary detection, and confirmation benchmarking.
    """
    model_config = ConfigDict(frozen=True)

    segment_id: str | None = Field(default=None, description="Unique segment identifier within video")
    start_frame: int = Field(default=0, description="Inclusive start frame index (0-indexed)")
    end_frame: int = Field(default=0, description="Inclusive end frame index")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    verb: str = Field(description="Action verb from VERB_VOCAB")
    object: str | None = Field(default=None, description="Interacted object from OBJECT_VOCAB")
    target: str | None = Field(default=None, description="Target receptacle from TARGET_VOCAB")
    violation_type: str = Field(
        default="NONE",
        description="Ground-truth procedure violation category (for confirmation/procedure evaluation, NOT an LSTM class)",
    )
    label_quality: str = Field(
        default="verified",
        description="Annotation quality status: verified, reviewed, synthetic, ambiguous",
    )
    source: str = Field(default="human", description="Label source: human, model, synthetic")
    notes: str | None = Field(default=None, description="Optional annotator notes or rationale")


class RecordingMetadata(BaseModel):
    """
    Comprehensive provenance metadata for an individual experiment recording (video or synthetic).
    """
    model_config = ConfigDict(frozen=True)

    video_id: str = Field(default="", description="Unique video clip identifier, e.g. EXP001_RUN_001_CAM01")
    recording_id: str = Field(description="Recording identifier or run tag")
    experiment_id: str = Field(default="EXP001", description="Experiment procedure ID")
    run_id: str = Field(description="Associated run ID, e.g. RUN-0001")
    subject_id: str = Field(default="ASTRONAUT-01", description="Astronaut / human subject identifier")
    camera_id: str = Field(default="CAM-01", description="Camera identifier, e.g. CAM-01")
    duration_seconds: float = Field(description="Duration in seconds")
    total_frames: int = Field(default=0, description="Total frames in video clip")
    fps: float = Field(default=30.0, description="Capture frame rate")
    width: int = Field(default=640, description="Frame width in pixels")
    height: int = Field(default=480, description="Frame height in pixels")
    scenario_type: str = Field(default="nominal", description="nominal, wrong_object, wrong_target, skipped_step, repeated_step, hesitation, dropout")
    annotator_id: str | None = Field(default="SYSTEM", description="Annotator identifier or system tag")
    random_seed: int = Field(default=42, description="Random seed")
    created_at: float = Field(default_factory=time.time, description="Timestamp of recording/annotation creation")
    segments: list[ActionSegmentAnnotation] = Field(default_factory=list, description="Ordered ground-truth action segments")


class DatasetManifest(BaseModel):
    """Versioned dataset catalog ensuring strict reproducibility."""
    model_config = ConfigDict(frozen=True, extra="allow")

    dataset_version: str = "2026.09.01"
    generator_version: str = "1.0.0"
    feature_schema_version: str = "kinematic-26d-v1.0"
    random_seed: int = 42
    recordings_count: int
    total_windows: int
    splits: dict[str, list[str]]  # {"train": [...], "val": [...], "test": [...]}
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SplitPartition(BaseModel):
    """Container holding subject, run, and recording IDs assigned to a split."""
    model_config = ConfigDict(frozen=True)

    subjects: list[str] = Field(default_factory=list, description="Unique astronaut/subject IDs")
    runs: list[str] = Field(default_factory=list, description="Unique physical run IDs")
    recordings: list[str] = Field(default_factory=list, description="Unique video/recording IDs")


class SplitManifest(BaseModel):
    """Canonical Phase 2.8 Leakage-Safe Dataset Split Manifest."""
    model_config = ConfigDict(frozen=True, extra="allow")

    schema_version: str = "1.0"
    dataset_version: str = "2026.09.05"
    split_algorithm: str = "group_disjoint_v1.0"
    seed: int = 42
    group_by: str = "subject"  # "subject" | "run"
    ratios: dict[str, float] = Field(default_factory=lambda: {"train": 0.70, "validation": 0.15, "test": 0.15})
    splits: dict[str, list[str]] = Field(default_factory=dict)  # {"train": [...], "validation": [...], "test": [...]}
    train: SplitPartition = Field(default_factory=SplitPartition)
    validation: SplitPartition = Field(default_factory=SplitPartition)
    test: SplitPartition = Field(default_factory=SplitPartition)
    statistics: dict[str, Any] = Field(default_factory=dict)
    disjointness_audit: dict[str, Any] = Field(default_factory=dict)
    rare_classes: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)



# Feature Provenance Registry (verifies zero ground-truth leakage)
FEATURE_PROVENANCE = {
    0: ("hand_x", "Perception hand centroid x", "normalized [0, 1]"),
    1: ("hand_y", "Perception hand centroid y", "normalized [0, 1]"),
    2: ("hand_vx", "Hand velocity vx", "normalized velocity, 1/sec"),
    3: ("hand_vy", "Hand velocity vy", "normalized velocity, 1/sec"),
    4: ("red_x", "Red component centroid x", "normalized [0, 1]"),
    5: ("red_y", "Red component centroid y", "normalized [0, 1]"),
    6: ("red_vx", "Red component velocity vx", "normalized velocity, 1/sec"),
    7: ("red_vy", "Red component velocity vy", "normalized velocity, 1/sec"),
    8: ("yellow_x", "Yellow component centroid x", "normalized [0, 1]"),
    9: ("yellow_y", "Yellow component centroid y", "normalized [0, 1]"),
    10: ("yellow_vx", "Yellow component velocity vx", "normalized velocity, 1/sec"),
    11: ("yellow_vy", "Yellow component velocity vy", "normalized velocity, 1/sec"),
    12: ("dist_hand_red", "Euclidean distance hand-red", "normalized"),
    13: ("dist_hand_yellow", "Euclidean distance hand-yellow", "normalized"),
    14: ("dist_red_tgtA", "Euclidean distance red-TargetA", "normalized"),
    15: ("dist_yellow_tgtB", "Euclidean distance yellow-TargetB", "normalized"),
    16: ("dist_hand_container", "Euclidean distance hand-container", "normalized"),
    17: ("d_dot_hand_red", "Distance derivative d/dt(hand, red)", "approaching < 0, retreating > 0"),
    18: ("d_dot_hand_yellow", "Distance derivative d/dt(hand, yellow)", "approaching < 0, retreating > 0"),
    19: ("d_dot_red_tgtA", "Distance derivative d/dt(red, TargetA)", "placing < 0"),
    20: ("d_dot_yellow_tgtB", "Distance derivative d/dt(yellow, TargetB)", "placing < 0"),
    21: ("co_movement_red", "Relative velocity norm ||v_hand - v_red||", "carrying indicator (low when co-moving)"),
    22: ("co_movement_yellow", "Relative velocity norm ||v_hand - v_yellow||", "carrying indicator"),
    23: ("conf_hand", "Hand tracking confidence", "[0, 1], 0 if occluded/lost"),
    24: ("conf_red", "Red tracking confidence", "[0, 1], 0 if occluded/lost"),
    25: ("conf_yellow", "Yellow tracking confidence", "[0, 1], 0 if occluded/lost"),
}
NUM_FEATURES = len(FEATURE_PROVENANCE)  # 26 observable features
WINDOW_SIZE = 30  # Standard sequence duration (1 second at 30 FPS)


class SequenceSample(BaseModel):
    """
    Logical sample representation for sequence dataset.
    Represents a 30-frame historical window of 26-D features with endpoint multi-head targets.
    """
    model_config = ConfigDict(frozen=True)

    sequence_id: str = Field(description="Unique window sequence identifier, e.g. EXP001_RUN_001_CAM01_000001")
    run_id: str = Field(description="Associated run ID, e.g. RUN-0001")
    subject_id: str = Field(description="Subject / astronaut identifier, e.g. ASTRONAUT-01")
    video_id: str = Field(description="Source video identifier, e.g. EXP001_RUN_001_CAM01")
    start_frame: int = Field(description="Start frame index of the 30-frame window")
    end_frame: int = Field(description="End frame index of the 30-frame window (inclusive)")
    features: list[list[float]] = Field(description="Temporal feature window of shape [30, 26]")
    verb: int = Field(description="Integer index for action verb in VERB_VOCAB")
    object: int = Field(description="Integer index for interacted object in OBJECT_VOCAB")
    target: int = Field(description="Integer index for target receptacle in TARGET_VOCAB")


class SequenceLabel(BaseModel):
    """
    Metadata and multi-head target labels for an individual window index in labels.json.
    """
    model_config = ConfigDict(frozen=True)

    sample_idx: int = Field(description="0-indexed position within the split's features.npy")
    sequence_id: str = Field(description="Unique window sequence identifier, e.g. EXP001_RUN_001_CAM01_000001")
    run_id: str = Field(description="Associated run ID, e.g. RUN-0001")
    subject_id: str = Field(description="Subject identifier, e.g. ASTRONAUT-01")
    video_id: str = Field(description="Source video identifier, e.g. EXP001_RUN_001_CAM01")
    start_frame: int = Field(description="Start frame index")
    end_frame: int = Field(description="End frame index")
    verb: int = Field(description="Verb label index")
    object: int = Field(description="Object label index")
    target: int = Field(description="Target label index")


def export_feature_contract_dict() -> dict[str, Any]:
    """Generates the standardized feature contract metadata dictionary."""
    return {
        "feature_schema_version": "kinematic-26d-v1.0",
        "num_features": NUM_FEATURES,
        "window_size": WINDOW_SIZE,
        "features": [
            {
                "index": i,
                "name": name,
                "description": desc,
                "units": units,
            }
            for i, (name, desc, units) in sorted(FEATURE_PROVENANCE.items())
        ],
        "vocabularies": {
            "verb": VERB_VOCAB,
            "object": OBJECT_VOCAB,
            "target": TARGET_VOCAB,
            "violations": VIOLATION_VOCAB,
        },
    }

