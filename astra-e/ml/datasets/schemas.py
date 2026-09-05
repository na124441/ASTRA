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


class ActionSegmentAnnotation(BaseModel):
    """Ground-truth temporal action interval."""
    model_config = ConfigDict(frozen=True)

    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    verb: str = Field(description="Action verb")
    object: str | None = Field(default=None, description="Interacted object")
    target: str | None = Field(default=None, description="Target receptacle")
    quality: str = Field(default="verified", description="Annotation quality: verified, reviewed, synthetic")
    source: str = Field(default="synthetic", description="Label source: human, model, synthetic")


class RecordingMetadata(BaseModel):
    """Provenance metadata for an individual experiment recording."""
    model_config = ConfigDict(frozen=True)

    recording_id: str
    experiment_id: str
    run_id: str
    duration_seconds: float
    fps: float = 30.0
    width: int = 640
    height: int = 480
    scenario_type: str = "nominal"  # nominal, wrong_object, wrong_target, hesitation, dropout
    random_seed: int = 42
    segments: list[ActionSegmentAnnotation] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    """Versioned dataset catalog ensuring strict reproducibility."""
    model_config = ConfigDict(frozen=True)

    dataset_version: str = "2026.09.01"
    generator_version: str = "1.0.0"
    feature_schema_version: str = "kinematic-26d-v1.0"
    random_seed: int = 42
    recordings_count: int
    total_windows: int
    splits: dict[str, list[str]]  # {"train": [...], "val": [...], "test": [...]}
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Feature Provenance Registry (verifies zero ground-truth leakage)
FEATURE_PROVENANCE = {
    0: ("hand_x", "Perception hand centroid x", "normalized [0, 1]"),
    1: ("hand_y", "Perception hand centroid y", "normalized [0, 1]"),
    2: ("hand_vx", "Hand velocity vx", "px/sec normalized"),
    3: ("hand_vy", "Hand velocity vy", "px/sec normalized"),
    4: ("red_x", "Red component centroid x", "normalized [0, 1]"),
    5: ("red_y", "Red component centroid y", "normalized [0, 1]"),
    6: ("red_vx", "Red component velocity vx", "px/sec normalized"),
    7: ("red_vy", "Red component velocity vy", "px/sec normalized"),
    8: ("yellow_x", "Yellow component centroid x", "normalized [0, 1]"),
    9: ("yellow_y", "Yellow component centroid y", "normalized [0, 1]"),
    10: ("yellow_vx", "Yellow component velocity vx", "px/sec normalized"),
    11: ("yellow_vy", "Yellow component velocity vy", "px/sec normalized"),
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
