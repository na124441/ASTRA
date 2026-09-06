"""Unit tests for ASTRA-E Real-Video Temporal Action Recognition Inference.

Verifies:
Test 1 - Class mapping (0->brush_hair, 1->drink, 2->eat, 3->pour, 4->clap, 5->wave)
Test 2 - Frame sampling (exactly 16 uniform indices across [0, N-1])
Test 3 - Short video handling (videos shorter than 16 frames resample without crashing)
Test 4 - Tensor shapes (CNN sequence -> [1, 16, 576], LSTM output -> [1, 6])
Test 5 - Invalid checkpoint handling (mismatched classes, hidden size, or dimensions fail closed)
Test 6 - Invalid video handling (missing or empty/corrupt video fails with clear error)
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure astra package is resolvable whether run from root or astra-e
for candidate in [
    Path(__file__).resolve().parent.parent.parent,
    Path(__file__).resolve().parent.parent.parent / "astra-e",
    Path(__file__).resolve().parent.parent.parent.parent,
    Path(__file__).resolve().parent.parent.parent.parent / "astra-e",
]:
    if (candidate / "astra").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from astra.inference.realvideo import (
    ACTIONS,
    ASTRARealVideoModel,
    ASTRARealVideoNet,
    sample_frame_indices,
)


class TestRealVideoInference:
    def test_class_mapping(self):
        """Test 1: Verify exact 6-class HMDB51 subset mapping and ordering."""
        expected = [
            "brush_hair",
            "drink",
            "eat",
            "pour",
            "clap",
            "wave",
        ]
        assert len(ACTIONS) == 6
        assert ACTIONS == expected
        assert ACTIONS[0] == "brush_hair"
        assert ACTIONS[1] == "drink"
        assert ACTIONS[2] == "eat"
        assert ACTIONS[3] == "pour"
        assert ACTIONS[4] == "clap"
        assert ACTIONS[5] == "wave"

    def test_frame_sampling(self):
        """Test 2: Given arbitrary frame counts, verify exactly 16 uniform indices across [0, N-1]."""
        # Case A: 100 frames
        indices_100 = sample_frame_indices(num_frames=100, target_frames=16)
        assert len(indices_100) == 16
        assert indices_100[0] == 0
        assert indices_100[-1] == 99
        # Monotonically non-decreasing
        assert all(x <= y for x, y in zip(indices_100, indices_100[1:]))

        # Case B: Exactly 16 frames
        indices_16 = sample_frame_indices(num_frames=16, target_frames=16)
        assert indices_16 == list(range(16))

        # Case C: 31 frames (steps of 2)
        indices_31 = sample_frame_indices(num_frames=31, target_frames=16)
        assert indices_31 == [i * 2 for i in range(16)]

        # Error cases
        with pytest.raises(ValueError):
            sample_frame_indices(0, 16)
        with pytest.raises(ValueError):
            sample_frame_indices(10, 0)

    def test_short_video_resampling(self):
        """Test 3: Verify videos shorter than 16 frames still produce a 16-frame tensor."""
        # 5 frames
        indices_5 = sample_frame_indices(num_frames=5, target_frames=16)
        assert len(indices_5) == 16
        assert min(indices_5) == 0
        assert max(indices_5) == 4
        assert indices_5[0] == 0
        assert indices_5[-1] == 4

        # 1 frame
        indices_1 = sample_frame_indices(num_frames=1, target_frames=16)
        assert len(indices_1) == 16
        assert indices_1 == [0] * 16

        # Test predict_frames on short sequence of dummy frames
        net = ASTRARealVideoNet()
        model = ASTRARealVideoModel.__new__(ASTRARealVideoModel)
        model.net = net
        model.device = torch.device("cpu")
        model.num_classes = 6
        model.feature_dim = 576
        model.hidden_dim = 128
        model.num_layers = 2
        net.eval()

        dummy_frames = [np.zeros((160, 160, 3), dtype=np.uint8) for _ in range(5)]
        result = model.predict_frames(dummy_frames, top_k=3)
        assert "action" in result
        assert result["action"] in ACTIONS
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["top_k"]) == 3
        assert result["latency_ms"] > 0

    def test_tensor_shapes(self):
        """Test 4: Verify CNN sequence -> [1, 16, 576] and LSTM output -> [1, 6]."""
        net = ASTRARealVideoNet(num_classes=6, feature_dim=576, hidden_dim=128, num_layers=2)
        net.eval()

        # Single batch: [1, 16, 3, 160, 160]
        x_single = torch.randn(1, 16, 3, 160, 160)
        with torch.no_grad():
            cnn_features = net.extract_visual_features(x_single)
            logits = net(x_single)

        assert cnn_features.shape == (1, 16, 576)
        assert logits.shape == (1, 6)

        # Multi-batch: [2, 16, 3, 160, 160]
        x_batch = torch.randn(2, 16, 3, 160, 160)
        with torch.no_grad():
            cnn_batch = net.extract_visual_features(x_batch)
            logits_batch = net(x_batch)

        assert cnn_batch.shape == (2, 16, 576)
        assert logits_batch.shape == (2, 6)

    def test_invalid_checkpoint_fails_closed(self, tmp_path):
        """Test 5: Verify an incompatible checkpoint fails with a useful, actionable error."""
        # Checkpoint with wrong number of classes (10 instead of 6)
        bad_net = ASTRARealVideoNet(num_classes=10)
        bad_ckpt = tmp_path / "bad_classes.pt"
        torch.save({"model_state_dict": bad_net.state_dict()}, bad_ckpt)

        with pytest.raises(ValueError) as exc_info:
            ASTRARealVideoModel(checkpoint_path=bad_ckpt)
        assert "ASTRA-E real-video checkpoint incompatible" in str(exc_info.value)
        assert "expected 6 output classes, found 10" in str(exc_info.value)

        # Checkpoint with wrong hidden size (64 instead of 128)
        bad_hidden = ASTRARealVideoNet(hidden_dim=64)
        bad_hidden_ckpt = tmp_path / "bad_hidden.pt"
        torch.save({"model_state_dict": bad_hidden.state_dict()}, bad_hidden_ckpt)

        with pytest.raises(ValueError) as exc_hidden:
            ASTRARealVideoModel(checkpoint_path=bad_hidden_ckpt)
        assert "expected LSTM hidden size 128" in str(exc_hidden.value)

        # Non-existent checkpoint path
        with pytest.raises(FileNotFoundError) as exc_nf:
            ASTRARealVideoModel(checkpoint_path="non_existent/path/model.pt")
        assert "ASTRA-E real-video checkpoint not found" in str(exc_nf.value)

    def test_invalid_video_fails(self, tmp_path):
        """Test 6: Verify missing or corrupt video produces a clear error."""
        # Find or create valid model
        ckpt_path = Path("models/realvideo/astra_realvideo_lstm_best.pt")
        if not ckpt_path.exists():
            ckpt_path = Path("astra-e/models/realvideo/astra_realvideo_lstm_best.pt")

        model = ASTRARealVideoModel(checkpoint_path=ckpt_path)

        # Missing video
        with pytest.raises(FileNotFoundError) as exc_missing:
            model.predict("totally_missing_video.mp4")
        assert "Video file does not exist" in str(exc_missing.value)

        # Empty/corrupt video file
        corrupt_file = tmp_path / "corrupt.mp4"
        corrupt_file.write_bytes(b"")

        with pytest.raises(ValueError) as exc_corrupt:
            model.predict(corrupt_file)
        assert "corrupt" in str(exc_corrupt.value).lower()
