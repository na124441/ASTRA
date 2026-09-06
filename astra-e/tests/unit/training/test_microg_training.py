"""Unit tests for MicroG-4M temporal action recognition baseline.

Verifies:
1. Sparse MicroG action IDs map correctly to contiguous IDs (0..49)
2. Label mapping is deterministic
3. Same seed produces identical grouped split partitions
4. Video-level groups never overlap across partitions (zero temporal leakage)
5. Temporal window has exactly 30 frames (X_i = F[i:i+30])
6. Zero future lookahead (endpoint frame is i+29)
7. Model forward pass produces tensor of shape [B, 50]
8. Checkpoint saving and loading (best.pt) preserves state_dict and configs
9. Missing video data triggers MicroGVideoUnavailableError and fails closed
"""

import json
import sys
from pathlib import Path
import pytest
import torch

# Ensure ml package is resolvable whether pytest is run from root or astra-e
for candidate in [
    Path(__file__).resolve().parent.parent.parent,
    Path(__file__).resolve().parent.parent.parent / "astra-e",
    Path(__file__).resolve().parent.parent.parent.parent,
    Path(__file__).resolve().parent.parent.parent.parent / "astra-e",
]:
    if (candidate / "ml").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from ml.datasets.adapters.microg_taxonomy import MicroGTaxonomy
from ml.datasets.adapters.microg_splitter import MicroGGroupedSplitter, GroupedSplitError
from ml.datasets.adapters.microg_temporal_dataset import (
    MicroGTemporalDataset,
    MicroGVideoUnavailableError,
)
from ml.activity.models.microg_lstm import CausalMicroGLSTM


class TestMicroGTaxonomy:
    def test_sparse_to_contiguous_mapping(self):
        """Test 1: Sparse MicroG action IDs map correctly to contiguous IDs (0..49)."""
        taxonomy = MicroGTaxonomy()
        assert taxonomy.num_classes == 50

        # Verify all contiguous IDs are in [0, 49]
        contiguous_indices = set(taxonomy.sparse_to_contiguous.values())
        assert contiguous_indices == set(range(50))

        # Check known sparse IDs
        # Sparse ID 1 -> "bend/bow (at the waist)"
        assert 1 in taxonomy.sparse_to_contiguous
        assert 80 in taxonomy.sparse_to_contiguous
        assert 79 in taxonomy.sparse_to_contiguous

        # Bidirectional consistency
        for sparse_id, contig_idx in taxonomy.sparse_to_contiguous.items():
            assert taxonomy.to_contiguous(sparse_id) == contig_idx
            assert taxonomy.to_sparse(contig_idx) == sparse_id
            assert 0 <= contig_idx < 50

        # Invalid ID should raise KeyError
        with pytest.raises(KeyError):
            taxonomy.to_contiguous(9999)
        with pytest.raises(KeyError):
            taxonomy.to_sparse(100)

    def test_label_mapping_determinism(self, tmp_path):
        """Test 2: Label mapping is deterministic across multiple instances and runs."""
        tax1 = MicroGTaxonomy()
        tax2 = MicroGTaxonomy()

        assert tax1.sorted_action_ids == tax2.sorted_action_ids
        assert tax1.sparse_to_contiguous == tax2.sparse_to_contiguous
        assert tax1.contiguous_to_sparse == tax2.contiguous_to_sparse

        # Verify export format is deterministic and valid JSON
        out1 = tmp_path / "label_map_1.json"
        out2 = tmp_path / "label_map_2.json"
        tax1.export_label_map(out1)
        tax2.export_label_map(out2)

        with open(out1, "r", encoding="utf-8") as f1, open(out2, "r", encoding="utf-8") as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)

        assert data1 == data2
        assert len(data1) == 50
        # Check structure of entries
        sample_entry = data1["1"]
        assert "class_index" in sample_entry
        assert "name" in sample_entry
        assert "label_type" in sample_entry


class TestMicroGGroupedSplitter:
    @pytest.fixture
    def mock_annotations(self):
        """Create mock annotations across 20 distinct videos."""
        annotations = []
        for v in range(20):
            vid = f"clip_{v:03d}"
            # Add 2-4 actions per video
            for a in range(3):
                annotations.append({
                    "video_id": vid,
                    "movie_or_real": "movie",
                    "person_id": "1",
                    "action": (v * 3 + a) % 50 + 1,  # valid sparse action
                })
        return annotations

    def test_reproducible_split_partitions(self, mock_annotations):
        """Test 3: Same seed produces identical grouped split partitions."""
        splitter_a = MicroGGroupedSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
        splitter_b = MicroGGroupedSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

        split_a = splitter_a.split_annotations(mock_annotations, group_by="video_id")
        split_b = splitter_b.split_annotations(mock_annotations, group_by="video_id")

        for key in ["train", "val", "test"]:
            vids_a = [r["video_id"] for r in split_a[key]]
            vids_b = [r["video_id"] for r in split_b[key]]
            assert vids_a == vids_b

    def test_zero_temporal_data_leakage(self, mock_annotations):
        """Test 4: Video-level groups never overlap across partitions (zero temporal leakage)."""
        splitter = MicroGGroupedSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=123)
        split = splitter.split_annotations(mock_annotations, group_by="video_id")

        train_vids = set(r["video_id"] for r in split["train"])
        val_vids = set(r["video_id"] for r in split["val"])
        test_vids = set(r["video_id"] for r in split["test"])

        assert len(train_vids) > 0
        assert len(val_vids) > 0
        assert len(test_vids) > 0

        # Strict disjoint assertions
        assert train_vids.isdisjoint(val_vids), "Temporal leakage detected between Train and Val!"
        assert train_vids.isdisjoint(test_vids), "Temporal leakage detected between Train and Test!"
        assert val_vids.isdisjoint(test_vids), "Temporal leakage detected between Val and Test!"

        # Ensure all videos are preserved
        all_vids = set(r["video_id"] for r in mock_annotations)
        assert (train_vids | val_vids | test_vids) == all_vids


class TestMicroGTemporalDataset:
    @pytest.fixture
    def sample_data(self):
        taxonomy = MicroGTaxonomy()
        annotations = [
            {"video_id": "test_clip_01", "action": 1, "person_id": "1"},
            {"video_id": "test_clip_02", "action": 3, "person_id": "1"},
        ]
        return annotations, taxonomy

    def test_temporal_window_size(self, sample_data):
        """Test 5: Temporal window has exactly 30 frames (X_i = F[i:i+30])."""
        annotations, taxonomy = sample_data
        dataset = MicroGTemporalDataset(
            annotations=annotations,
            taxonomy=taxonomy,
            window_size=30,
            feature_dim=128,
            allow_synthetic_test=True,
        )

        assert len(dataset) > 0
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        # Verify exactly 30 temporal steps and 128 feature dimension
        assert x.shape == (30, 128)
        assert y.dtype == torch.long
        assert 0 <= y.item() < 50

    def test_zero_future_lookahead(self, sample_data):
        """Test 6: Zero future lookahead (endpoint frame is i+29)."""
        annotations, taxonomy = sample_data
        dataset = MicroGTemporalDataset(
            annotations=annotations,
            taxonomy=taxonomy,
            window_size=30,
            allow_synthetic_test=True,
        )

        for sample in dataset.samples:
            start_f = sample["start_frame"]
            endpoint_f = sample["endpoint_frame"]
            # Window encompasses start_f to start_f + 30 - 1
            assert endpoint_f == start_f + 29
            # Verify the causal endpoint precedes or equals endpoint_f (never beyond)
            assert endpoint_f < start_f + 30

    def test_missing_video_data_fails_closed(self, sample_data):
        """Test 9: Missing video data triggers MicroGVideoUnavailableError and fails closed."""
        annotations, taxonomy = sample_data

        # Explicitly without allow_synthetic_test and without valid video/feature dir
        with pytest.raises(MicroGVideoUnavailableError) as exc_info:
            MicroGTemporalDataset(
                annotations=annotations,
                taxonomy=taxonomy,
                video_dir=None,
                feature_dir=None,
                allow_synthetic_test=False,
            )

        assert "MicroG-4M Video Source Not Found" in str(exc_info.value)
        assert "Synthetic / random video frames will NOT be fabricated" in str(exc_info.value)


class TestCausalMicroGLSTM:
    def test_model_forward_shape(self):
        """Test 7: Model forward pass produces tensor of shape [B, 50]."""
        model = CausalMicroGLSTM(
            input_dim=128,
            hidden_dim=256,
            num_layers=2,
            num_classes=50,
            dropout=0.2,
        )
        model.eval()

        batch_sizes = [1, 4, 16]
        for b in batch_sizes:
            x = torch.randn(b, 30, 128)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (b, 50)
            assert not torch.isnan(out).any()

    def test_checkpoint_saving_and_loading(self, tmp_path):
        """Test 8: Checkpoint saving and loading (best.pt) preserves state_dict and configs."""
        model = CausalMicroGLSTM(
            input_dim=128,
            hidden_dim=256,
            num_layers=2,
            num_classes=50,
            dropout=0.2,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_path = tmp_path / "best.pt"

        checkpoint_data = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": 0.845,
            "best_val_accuracy": 0.890,
            "class_mapping": {1: 0, 3: 1},
            "num_classes": 50,
            "model_config": {
                "input_dim": 128,
                "hidden_dim": 256,
                "num_layers": 2,
                "num_classes": 50,
                "dropout": 0.2,
            },
            "random_seed": 42,
        }
        torch.save(checkpoint_data, ckpt_path)
        assert ckpt_path.exists()

        # Load back
        loaded = torch.load(ckpt_path, map_location="cpu")
        assert loaded["epoch"] == 5
        assert loaded["best_val_macro_f1"] == 0.845
        assert loaded["num_classes"] == 50
        assert loaded["model_config"]["hidden_dim"] == 256

        # Re-instantiate model with saved config and load weights
        new_model = CausalMicroGLSTM(
            input_dim=loaded["model_config"]["input_dim"],
            hidden_dim=loaded["model_config"]["hidden_dim"],
            num_layers=loaded["model_config"]["num_layers"],
            num_classes=loaded["model_config"]["num_classes"],
            dropout=loaded["model_config"]["dropout"],
        )
        new_model.load_state_dict(loaded["model_state_dict"])
        new_model.eval()
        model.eval()

        # Check numerical equivalence of predictions
        dummy_x = torch.randn(2, 30, 128)
        with torch.no_grad():
            out_orig = model(dummy_x)
            out_loaded = new_model(dummy_x)
        assert torch.allclose(out_orig, out_loaded, atol=1e-6)
