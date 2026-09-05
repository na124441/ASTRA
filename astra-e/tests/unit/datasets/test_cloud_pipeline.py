"""Unit tests for cloud pipeline components: adapters, FeatureSequenceDataset, and sequence builder."""

from pathlib import Path
import numpy as np
import pytest
import torch

from ml.datasets.adapters.microg import MicroGDatasetAdapter
from ml.datasets.feature_dataset import FeatureSequenceDataset
from scripts.cloud.build_sequences import build_aligned_sequence
from scripts.cloud.validate_features import validate_feature_file


def test_microg_adapter(tmp_path):
    """Verify MicroGDatasetAdapter correctly maps microgravity actions to ASTRA-E ontology."""
    adapter = MicroGDatasetAdapter(dataset_root=tmp_path)
    recordings = adapter.load_dataset()
    assert len(recordings) > 0

    first_rec = recordings[0]
    assert first_rec.experiment_id == "EXP001"
    assert first_rec.subject_id == "Astronaut-A"
    assert len(first_rec.segments) == 1
    assert first_rec.segments[0].verb == "GRASP"
    assert first_rec.segments[0].object == "RED_COMPONENT"


def test_build_aligned_sequence_and_validate(tmp_path):
    """Verify build_aligned_sequence generates contract-compliant sequences."""
    feat_npz = tmp_path / "raw_feat.npz"
    out_npz = tmp_path / "aligned_seq.npz"

    # Create dummy 26-D features (100 frames)
    n_frames = 100
    dummy_feats = np.zeros((n_frames, 26), dtype=np.float32)
    # Hand coordinates [0, 1]
    dummy_feats[:, 0] = 0.5
    dummy_feats[:, 1] = 0.5
    dummy_feats[:, 23] = 0.9  # conf

    np.savez_compressed(feat_npz, features=dummy_feats)

    # Build aligned sequence
    res = build_aligned_sequence(feat_npz, None, out_npz)
    assert res["total_frames"] == n_frames
    assert out_npz.exists()

    # Validate against contract
    ok, violations = validate_feature_file(out_npz)
    assert ok, f"Violations found: {violations}"


def test_feature_sequence_dataset(tmp_path):
    """Verify FeatureSequenceDataset slices causal sliding windows of shape (30, 26)."""
    seq_npz = tmp_path / "test_run.npz"
    n_frames = 60
    feats = np.ones((n_frames, 26), dtype=np.float32) * 0.5
    verbs = np.zeros(n_frames, dtype=np.int64)
    objects = np.zeros(n_frames, dtype=np.int64)
    targets = np.zeros(n_frames, dtype=np.int64)
    violations = np.zeros(n_frames, dtype=np.int64)

    np.savez_compressed(
        seq_npz,
        features=feats,
        verbs=verbs,
        objects=objects,
        targets=targets,
        violations=violations,
    )

    dataset = FeatureSequenceDataset([seq_npz], window_size=30)
    # Total windows should be: 60 - 30 + 1 = 31
    assert len(dataset) == 31

    item = dataset[0]
    assert "features" in item
    assert item["features"].shape == (30, 26)
    assert item["features"].dtype == torch.float32
    assert item["verb"].item() == 0


def test_video_feature_extractor_worker_compliance(tmp_path):
    """Verify VideoFeatureExtractorWorker satisfies all 10 cloud pipeline requirements."""
    import cv2
    from scripts.cloud.extract_features_cloud import VideoFeatureExtractorWorker

    video_path = tmp_path / "test_exp.mp4"
    out_npz = tmp_path / "test_exp.npz"

    # Write 15 mock frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    for i in range(15):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Red component rectangle
        cv2.rectangle(frame, (100, 100), (150, 150), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()

    worker = VideoFeatureExtractorWorker(frame_width=640.0, frame_height=480.0)
    res = worker.process_video(video_path, out_npz)

    assert out_npz.exists()
    assert res["frames"] == 15
    assert res["shape"] == (15, 26)

    # Load and inspect .npz
    with np.load(out_npz) as data:
        assert "features" in data
        assert "timestamps" in data
        assert "frame_ids" in data
        features = data["features"]
        timestamps = data["timestamps"]
        frame_ids = data["frame_ids"]

    assert features.shape == (15, 26)
    assert features.dtype == np.float32
    assert len(timestamps) == 15
    assert len(frame_ids) == 15
    assert frame_ids[0] == 0
    assert frame_ids[-1] == 14
    assert np.all(np.diff(timestamps) > 0)


def test_assert_extraction_invariants_fails_loudly():
    """Verify assert_extraction_invariants rejects corrupt, non-monotonic, or dimension-deviant features."""
    from scripts.cloud.extract_features_cloud import (
        ExtractionInvariantError,
        assert_extraction_invariants,
    )

    T = 10
    valid_feats = np.zeros((T, 26), dtype=np.float32)
    valid_ts = np.arange(T, dtype=np.float32) * (1.0 / 30.0)
    valid_ids = np.arange(T, dtype=np.int32)

    # 1. Valid features pass cleanly
    assert_extraction_invariants(valid_feats, valid_ts, valid_ids, "test_valid")

    # 2. Reject empty recordings (T = 0)
    with pytest.raises(ExtractionInvariantError, match="Rejected empty recording"):
        assert_extraction_invariants(
            np.empty((0, 26), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )

    # 3. Reject wrong feature dimension (e.g. 25 instead of 26)
    bad_dim = np.zeros((T, 25), dtype=np.float32)
    with pytest.raises(ExtractionInvariantError, match="features.shape\\[1\\] must be exactly 26"):
        assert_extraction_invariants(bad_dim, valid_ts, valid_ids)

    # 4. Reject wrong ndim (e.g. 1D or 3D)
    with pytest.raises(ExtractionInvariantError, match="features.ndim must be 2"):
        assert_extraction_invariants(valid_feats[0], valid_ts[:1], valid_ids[:1])

    # 5. Reject wrong dtype (e.g. float64)
    bad_dtype = valid_feats.astype(np.float64)
    with pytest.raises(ExtractionInvariantError, match="features.dtype must be float32"):
        assert_extraction_invariants(bad_dtype, valid_ts, valid_ids)

    # 6. Reject NaN
    nan_feats = valid_feats.copy()
    nan_feats[2, 5] = np.nan
    with pytest.raises(ExtractionInvariantError, match="detected 1 NaN values"):
        assert_extraction_invariants(nan_feats, valid_ts, valid_ids)

    # 7. Reject Inf
    inf_feats = valid_feats.copy()
    inf_feats[4, 2] = np.inf
    with pytest.raises(ExtractionInvariantError, match="detected 1 Inf values"):
        assert_extraction_invariants(inf_feats, valid_ts, valid_ids)

    # 8. Reject non-monotonic timestamps (e.g. dt <= 0)
    bad_ts = valid_ts.copy()
    bad_ts[3] = bad_ts[2]  # dt = 0
    with pytest.raises(ExtractionInvariantError, match="non-monotonic timestamps detected"):
        assert_extraction_invariants(valid_feats, bad_ts, valid_ids)

    bad_ts_reverse = valid_ts.copy()
    bad_ts_reverse[4] = bad_ts_reverse[3] - 0.01  # time goes backwards
    with pytest.raises(ExtractionInvariantError, match="non-monotonic timestamps detected"):
        assert_extraction_invariants(valid_feats, bad_ts_reverse, valid_ids)


