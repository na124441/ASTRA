"""Comprehensive unit tests for ASTRA-E Phase 2.7 TemporalSequenceGenerator.

Tests verify:
1. Window shape: (T=100 -> N=71, 30, 26)
2. Causal alignment: y_i = Y[i + 29]
3. Zero future frame leakage
4. Fail-closed on short recording (T=29)
5. Exact minimum recording boundary (T=30 -> N=1)
6. Rejection of NaN and Inf
7. Rejection of invalid feature dimensions (T, 25), (T, 27), ndim != 2, dtype != float32
8. Rejection of label length mismatches
9. Rejection of invalid vocabulary label IDs
10. Segment annotation to frame label conversion with IDLE defaults
11. Fail-closed on overlapping conflicting annotations (and pass on identical overlapping)
12. Recording boundary isolation (windows never cross runs)
13. Determinism across repeated executions
"""

import json
from pathlib import Path
import numpy as np
import pytest

from ml.datasets.schemas import (
    ActionSegmentAnnotation,
    NUM_FEATURES,
    OBJECT_TO_IDX,
    OBJECT_VOCAB,
    TARGET_TO_IDX,
    TARGET_VOCAB,
    VERB_TO_IDX,
    VERB_VOCAB,
    WINDOW_SIZE,
)
from ml.datasets.sequence_generator import (
    AnnotationConflictError,
    GeneratedSequences,
    SequenceGenerationError,
    SequenceValidationError,
    TemporalSequenceGenerator,
    align_segments_to_frame_labels,
)


def _make_dummy_stream(
    T: int,
    num_features: int = NUM_FEATURES,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper creating valid dummy continuous feature stream and labels."""
    features = np.zeros((T, num_features), dtype=dtype)
    # Give discernible pattern to features
    for t in range(T):
        features[t, 0] = (t % 100) / 100.0  # hand_x
        features[t, 1] = 0.5                 # hand_y
        features[t, 23] = 0.95               # conf_hand

    verbs = np.zeros(T, dtype=np.int64)
    objects = np.zeros(T, dtype=np.int64)
    targets = np.zeros(T, dtype=np.int64)
    return features, verbs, objects, targets


def test_correct_window_shape():
    """1. Verify T=100 generates exactly N=71 causal 30-frame windows of shape (71, 30, 26)."""
    T = 100
    features, verbs, objects, targets = _make_dummy_stream(T)
    generator = TemporalSequenceGenerator(window_size=30, stride=1)

    result = generator.generate_sequences(features, verbs, objects, targets)

    expected_N = T - 30 + 1  # 71
    assert result.num_sequences == expected_N
    assert result.X.shape == (expected_N, 30, 26)
    assert result.X.dtype == np.float32
    assert result.verbs.shape == (expected_N,)
    assert result.verbs.dtype == np.int64
    assert result.objects.shape == (expected_N,)
    assert result.targets.shape == (expected_N,)
    assert len(result.sample_metadata) == expected_N


def test_causal_alignment():
    """2. Verify supervision strictly aligns to the window endpoint: y_i = Y[i + 29]."""
    T = 60
    features, verbs, objects, targets = _make_dummy_stream(T)
    # Populate non-trivial changing labels
    for t in range(T):
        verbs[t] = t % len(VERB_VOCAB)
        objects[t] = t % len(OBJECT_VOCAB)
        targets[t] = t % len(TARGET_VOCAB)

    generator = TemporalSequenceGenerator(window_size=30, stride=1)
    result = generator.generate_sequences(features, verbs, objects, targets)

    for i in range(result.num_sequences):
        endpoint_frame = i + 29
        assert result.verbs[i] == verbs[endpoint_frame]
        assert result.objects[i] == objects[endpoint_frame]
        assert result.targets[i] == targets[endpoint_frame]

        meta = result.sample_metadata[i]
        assert meta["start_frame"] == i
        assert meta["end_frame"] == endpoint_frame
        assert meta["label_frame"] == endpoint_frame
        assert meta["verb"] == verbs[endpoint_frame]
        assert meta["object"] == objects[endpoint_frame]
        assert meta["target"] == targets[endpoint_frame]


def test_no_future_frame_leakage():
    """3. Verify zero future frame leakage: X_i only reflects frames i to i+29."""
    T = 50
    features, verbs, objects, targets = _make_dummy_stream(T)
    generator = TemporalSequenceGenerator(window_size=30, stride=1)

    # Modify future frame at index 35
    features_copy = features.copy()
    features_copy[35, :] = 999.0

    res_clean = generator.generate_sequences(features, verbs, objects, targets)
    res_modified = generator.generate_sequences(features_copy, verbs, objects, targets)

    # Windows 0..5 (ending at frames 29..34) MUST be identical and not contain 999.0
    for i in range(6):  # end_frame = i + 29 <= 34 < 35
        assert np.array_equal(res_clean.X[i], res_modified.X[i])
        assert not np.any(res_clean.X[i] == 999.0)
        assert not np.any(res_modified.X[i] == 999.0)

    # Window 6 (i=6, frames 6..35) contains frame 35, so it must differ
    assert not np.array_equal(res_clean.X[6], res_modified.X[6])
    assert np.any(res_modified.X[6] == 999.0)


def test_short_recording_fails():
    """4. Verify fail-closed behavior when recording is shorter than window_size (T=29 < 30)."""
    T = 29
    features, verbs, objects, targets = _make_dummy_stream(T)
    generator = TemporalSequenceGenerator(window_size=30)

    with pytest.raises(SequenceValidationError, match="shorter than minimum window_size=30"):
        generator.generate_sequences(features, verbs, objects, targets)


def test_exact_minimum_recording():
    """5. Verify exact minimum recording (T=30) generates exactly 1 causal sequence."""
    T = 30
    features, verbs, objects, targets = _make_dummy_stream(T)
    generator = TemporalSequenceGenerator(window_size=30)

    result = generator.generate_sequences(features, verbs, objects, targets)
    assert result.num_sequences == 1
    assert result.X.shape == (1, 30, 26)
    assert result.sample_metadata[0]["start_frame"] == 0
    assert result.sample_metadata[0]["end_frame"] == 29
    assert result.sample_metadata[0]["label_frame"] == 29


def test_nan_inf_rejection():
    """6. Verify rejection of NaN, +Inf, and -Inf in feature matrix."""
    T = 40
    generator = TemporalSequenceGenerator(window_size=30)

    # Test NaN
    features, verbs, objects, targets = _make_dummy_stream(T)
    features[15, 5] = np.nan
    with pytest.raises(SequenceValidationError, match="features matrix contains 1 NaN"):
        generator.generate_sequences(features, verbs, objects, targets)

    # Test +Inf
    features, verbs, objects, targets = _make_dummy_stream(T)
    features[20, 2] = np.inf
    with pytest.raises(SequenceValidationError, match="features matrix contains 1 Inf"):
        generator.generate_sequences(features, verbs, objects, targets)

    # Test -Inf
    features, verbs, objects, targets = _make_dummy_stream(T)
    features[10, 1] = -np.inf
    with pytest.raises(SequenceValidationError, match="features matrix contains 1 Inf"):
        generator.generate_sequences(features, verbs, objects, targets)


def test_invalid_feature_dimension():
    """7. Verify rejection of wrong dimensions: (T, 25), (T, 27), ndim != 2, and dtype != float32."""
    T = 40
    generator = TemporalSequenceGenerator(window_size=30)
    verbs = np.zeros(T, dtype=np.int64)
    objects = np.zeros(T, dtype=np.int64)
    targets = np.zeros(T, dtype=np.int64)

    # Shape (T, 25)
    f25 = np.zeros((T, 25), dtype=np.float32)
    with pytest.raises(SequenceValidationError, match="feature dimension must be 26, got 25"):
        generator.generate_sequences(f25, verbs, objects, targets)

    # Shape (T, 27)
    f27 = np.zeros((T, 27), dtype=np.float32)
    with pytest.raises(SequenceValidationError, match="feature dimension must be 26, got 27"):
        generator.generate_sequences(f27, verbs, objects, targets)

    # 1D array
    f_1d = np.zeros(T, dtype=np.float32)
    with pytest.raises(SequenceValidationError, match="must be a 2D array"):
        generator.generate_sequences(f_1d, verbs, objects, targets)

    # 3D array
    f_3d = np.zeros((T, 1, 26), dtype=np.float32)
    with pytest.raises(SequenceValidationError, match="must be a 2D array"):
        generator.generate_sequences(f_3d, verbs, objects, targets)

    # Wrong dtype (float64)
    f_f64 = np.zeros((T, 26), dtype=np.float64)
    with pytest.raises(SequenceValidationError, match="features dtype must be float32, got float64"):
        generator.generate_sequences(f_f64, verbs, objects, targets)


def test_label_length_mismatch():
    """8. Verify fail-closed behavior when label arrays do not match feature length T."""
    T = 40
    features, verbs, objects, targets = _make_dummy_stream(T)
    generator = TemporalSequenceGenerator(window_size=30)

    # verbs mismatch
    with pytest.raises(SequenceValidationError, match="verbs length.*!= features length"):
        generator.generate_sequences(features, verbs[:39], objects, targets)

    # objects mismatch
    with pytest.raises(SequenceValidationError, match="objects length.*!= features length"):
        generator.generate_sequences(features, verbs, objects[:35], targets)

    # targets mismatch
    with pytest.raises(SequenceValidationError, match="targets length.*!= features length"):
        generator.generate_sequences(features, verbs, objects, targets[:10])


def test_invalid_label_ids():
    """9. Verify rejection of out-of-range label vocabulary integers."""
    T = 40
    features, verbs, objects, targets = _make_dummy_stream(T)
    generator = TemporalSequenceGenerator(window_size=30)

    # Negative label
    bad_verbs = verbs.copy()
    bad_verbs[5] = -1
    with pytest.raises(SequenceValidationError, match="verbs contains IDs outside valid range"):
        generator.generate_sequences(features, bad_verbs, objects, targets)

    # Label exceeding vocab length
    bad_objects = objects.copy()
    bad_objects[12] = len(OBJECT_VOCAB)  # strictly out of bounds [0, len-1]
    with pytest.raises(SequenceValidationError, match="objects contains IDs outside valid range"):
        generator.generate_sequences(features, verbs, bad_objects, targets)

    bad_targets = targets.copy()
    bad_targets[20] = len(TARGET_VOCAB) + 10
    with pytest.raises(SequenceValidationError, match="targets contains IDs outside valid range"):
        generator.generate_sequences(features, verbs, objects, bad_targets)


def test_segment_to_frame_conversion():
    """10. Verify segment interval annotations accurately map to frame labels with IDLE defaults."""
    T = 60
    segments = [
        ActionSegmentAnnotation(
            segment_id="seg_001",
            start_frame=10,
            end_frame=25,
            start_time=10 / 30.0,
            end_time=25 / 30.0,
            verb="GRASP",
            object="RED_COMPONENT",
            target="TARGET_A",
        ),
        ActionSegmentAnnotation(
            segment_id="seg_002",
            start_frame=40,
            end_frame=55,
            start_time=40 / 30.0,
            end_time=55 / 30.0,
            verb="RELEASE",
            object="RED_COMPONENT",
            target="TARGET_A",
        ),
    ]

    verbs, objects, targets, violations = align_segments_to_frame_labels(segments, total_frames=T)

    assert len(verbs) == T
    assert len(objects) == T
    assert len(targets) == T
    assert len(violations) == T

    # Frames 0..9: Default IDLE / NONE / NONE
    for t in range(0, 10):
        assert verbs[t] == VERB_TO_IDX["IDLE"]
        assert objects[t] == OBJECT_TO_IDX["NONE"]
        assert targets[t] == TARGET_TO_IDX["NONE"]

    # Frames 10..25: GRASP / RED_COMPONENT / TARGET_A
    for t in range(10, 26):
        assert verbs[t] == VERB_TO_IDX["GRASP"]
        assert objects[t] == OBJECT_TO_IDX["RED_COMPONENT"]
        assert targets[t] == TARGET_TO_IDX["TARGET_A"]

    # Frames 26..39: Default IDLE / NONE / NONE
    for t in range(26, 40):
        assert verbs[t] == VERB_TO_IDX["IDLE"]
        assert objects[t] == OBJECT_TO_IDX["NONE"]
        assert targets[t] == TARGET_TO_IDX["NONE"]

    # Frames 40..55: RELEASE / RED_COMPONENT / TARGET_A
    for t in range(40, 56):
        assert verbs[t] == VERB_TO_IDX["RELEASE"]
        assert objects[t] == OBJECT_TO_IDX["RED_COMPONENT"]
        assert targets[t] == TARGET_TO_IDX["TARGET_A"]

    # Frames 56..59: Default IDLE / NONE / NONE
    for t in range(56, 60):
        assert verbs[t] == VERB_TO_IDX["IDLE"]


def test_overlapping_conflicting_segments_fails_closed():
    """11. Verify contradictory overlapping annotations fail closed, while identical overlapping passes."""
    T = 60

    # Contradictory segments overlapping at frames 20..25
    conflicting_segments = [
        ActionSegmentAnnotation(
            segment_id="seg_1",
            start_frame=10,
            end_frame=25,
            start_time=10 / 30.0,
            end_time=25 / 30.0,
            verb="GRASP",
            object="RED_COMPONENT",
            target="NONE",
        ),
        ActionSegmentAnnotation(
            segment_id="seg_2",
            start_frame=20,
            end_frame=35,
            start_time=20 / 30.0,
            end_time=35 / 30.0,
            verb="RELEASE",  # CONFLICT!
            object="RED_COMPONENT",
            target="NONE",
        ),
    ]

    with pytest.raises(AnnotationConflictError, match="Conflicting overlapping annotations at frame 20"):
        align_segments_to_frame_labels(conflicting_segments, total_frames=T)

    # Identical overlapping segments (consistent annotations) MUST pass cleanly
    identical_overlapping = [
        ActionSegmentAnnotation(
            segment_id="seg_a",
            start_frame=10,
            end_frame=25,
            start_time=10 / 30.0,
            end_time=25 / 30.0,
            verb="GRASP",
            object="RED_COMPONENT",
            target="NONE",
        ),
        ActionSegmentAnnotation(
            segment_id="seg_b",
            start_frame=20,
            end_frame=30,
            start_time=20 / 30.0,
            end_time=30 / 30.0,
            verb="GRASP",  # Identical
            object="RED_COMPONENT",
            target="NONE",
        ),
    ]
    verbs, _, _, _ = align_segments_to_frame_labels(identical_overlapping, total_frames=T)
    assert verbs[20] == VERB_TO_IDX["GRASP"]



def test_recording_boundary_isolation(tmp_path: Path):
    """12. Verify windows never cross distinct physical recording boundaries."""
    # Create two separate recordings: Run A (T=45) and Run B (T=35)
    fA, vA, oA, tA = _make_dummy_stream(45)
    fB, vB, oB, tB = _make_dummy_stream(35)

    npz_a = tmp_path / "RUN_001.npz"
    npz_b = tmp_path / "RUN_002.npz"
    np.savez_compressed(npz_a, features=fA)
    np.savez_compressed(npz_b, features=fB)

    out_a = tmp_path / "seq_RUN_001.npz"
    out_b = tmp_path / "seq_RUN_002.npz"

    generator = TemporalSequenceGenerator(window_size=30)
    res_a = generator.process_recording_to_npz(npz_a, None, out_a)
    res_b = generator.process_recording_to_npz(npz_b, None, out_b)

    # Run A: 45 - 30 + 1 = 16 sequences
    # Run B: 35 - 30 + 1 = 6 sequences
    assert res_a["num_sequences"] == 16
    assert res_b["num_sequences"] == 6

    # Load output NPZs and verify independence
    with np.load(out_a) as data_a:
        assert data_a["X"].shape == (16, 30, 26)
        meta_a = json.loads(str(data_a["metadata"]))
        assert meta_a[0]["video_id"] == "EXP001_RUN_001_CAM01"
        assert meta_a[-1]["end_frame"] == 44

    with np.load(out_b) as data_b:
        assert data_b["X"].shape == (6, 30, 26)
        meta_b = json.loads(str(data_b["metadata"]))
        assert meta_b[0]["video_id"] == "EXP001_RUN_002_CAM01"
        assert meta_b[-1]["end_frame"] == 34


def test_determinism():
    """13. Verify bit-level determinism across repeated executions with identical inputs."""
    T = 80
    features, verbs, objects, targets = _make_dummy_stream(T)
    for t in range(T):
        features[t, 5] = (t * 0.12345) % 1.0
        verbs[t] = t % len(VERB_VOCAB)
        objects[t] = t % len(OBJECT_VOCAB)
        targets[t] = t % len(TARGET_VOCAB)

    generator = TemporalSequenceGenerator(window_size=30, stride=1)
    run1 = generator.generate_sequences(features, verbs, objects, targets)
    run2 = generator.generate_sequences(features, verbs, objects, targets)

    assert np.array_equal(run1.X, run2.X)
    assert np.array_equal(run1.verbs, run2.verbs)
    assert np.array_equal(run1.objects, run2.objects)
    assert np.array_equal(run1.targets, run2.targets)
    assert run1.sample_metadata == run2.sample_metadata
