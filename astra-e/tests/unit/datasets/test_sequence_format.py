"""Unit tests for the frozen sequence dataset format (astra-e-features layout)."""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from ml.datasets.feature_dataset import MmapFeatureDataset, create_mmap_dataloaders
from ml.datasets.schemas import SequenceSample, SequenceLabel, NUM_FEATURES, WINDOW_SIZE
from scripts.cloud.build_packed_dataset import compile_packed_dataset


def test_sequence_sample_pydantic_schema():
    """Verify SequenceSample Pydantic validation."""
    sample_data = {
        "sequence_id": "EXP001_RUN_001_CAM01_000001",
        "run_id": "RUN-0001",
        "subject_id": "ASTRONAUT-01",
        "video_id": "EXP001_RUN_001_CAM01",
        "start_frame": 0,
        "end_frame": 29,
        "features": [[0.1] * 26 for _ in range(30)],
        "verb": 1,
        "object": 1,
        "target": 0,
    }
    sample = SequenceSample(**sample_data)
    assert sample.sequence_id == "EXP001_RUN_001_CAM01_000001"
    assert len(sample.features) == 30
    assert len(sample.features[0]) == 26
    assert sample.verb == 1
    assert sample.object == 1
    assert sample.target == 0


def test_compile_packed_dataset_and_mmap_loading(tmp_path: Path):
    """Verify compiling raw sequences into astra-e-features layout using an explicit manifest."""
    seq_dir = tmp_path / "raw_sequences"
    seq_dir.mkdir(parents=True)

    # Create 3 synthetic sequences
    for i in range(3):
        t_frames = 60
        features = np.random.randn(t_frames, 26).astype(np.float32)
        verbs = np.random.randint(0, 10, size=t_frames)
        objects = np.random.randint(0, 4, size=t_frames)
        targets = np.random.randint(0, 4, size=t_frames)
        np.savez_compressed(
            seq_dir / f"RUN-000{i+1}.npz",
            features=features,
            verbs=verbs,
            objects=objects,
            targets=targets,
        )

    # Create explicit, leakage-free manifest partitioned by run
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_dict = {
        "splits": {
            "train": ["RUN-0001"],
            "validation": ["RUN-0002"],
            "test": ["RUN-0003"],
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dict, f)

    out_dataset = tmp_path / "astra-e-features"

    manifest_data = compile_packed_dataset(
        sequences_dir=seq_dir,
        manifest_path=manifest_path,
        output_dir=out_dataset,
        window_size=30,
    )

    assert (out_dataset / "train" / "features.npy").exists()
    assert (out_dataset / "train" / "labels.json").exists()
    assert (out_dataset / "validation" / "features.npy").exists()
    assert (out_dataset / "validation" / "labels.json").exists()
    assert (out_dataset / "test" / "features.npy").exists()
    assert (out_dataset / "test" / "labels.json").exists()
    assert (out_dataset / "metadata" / "dataset_manifest.json").exists()
    assert (out_dataset / "metadata" / "feature_contract.json").exists()

    # Verify tensor shapes
    train_feats = np.load(out_dataset / "train" / "features.npy")
    assert train_feats.ndim == 3
    assert train_feats.shape[1] == 30
    assert train_feats.shape[2] == 26
    assert train_feats.dtype == np.float32

    # Verify MmapFeatureDataset loading
    train_ds = MmapFeatureDataset(out_dataset / "train")
    assert len(train_ds) == len(train_feats)

    item = train_ds[0]
    assert item["features"].shape == (30, 26)
    assert isinstance(item["verb"], torch.Tensor)
    assert isinstance(item["object"], torch.Tensor)
    assert isinstance(item["target"], torch.Tensor)

    # Verify to_logical_sample matches exact user schema
    logical_sample = train_ds.to_logical_sample(0)
    expected_keys = {
        "sequence_id", "run_id", "subject_id", "video_id",
        "start_frame", "end_frame", "features", "verb", "object", "target"
    }
    assert set(logical_sample.keys()) == expected_keys
    assert len(logical_sample["features"]) == 30
    assert len(logical_sample["features"][0]) == 26
    assert logical_sample["start_frame"] == 0
    assert logical_sample["end_frame"] == 29

    # Verify DataLoaders
    train_loader, val_loader, test_loader = create_mmap_dataloaders(out_dataset, batch_size=4)
    batch = next(iter(train_loader))
    assert batch["features"].shape == (min(4, len(train_ds)), 30, 26)
    assert batch["verb"].shape == (min(4, len(train_ds)),)


def test_compile_fails_closed_on_missing_or_leaking_manifest(tmp_path: Path):
    """Verify compiler fails closed when manifest is missing, invalid, or contains data leakage."""
    seq_dir = tmp_path / "raw_sequences"
    seq_dir.mkdir(parents=True)
    np.savez_compressed(
        seq_dir / "RUN-0001.npz",
        features=np.zeros((60, 26), dtype=np.float32),
        verbs=np.zeros(60, dtype=np.int64),
        objects=np.zeros(60, dtype=np.int64),
        targets=np.zeros(60, dtype=np.int64),
    )

    out_dataset = tmp_path / "astra-e-features-fail"

    # 1. Fail closed when manifest_path is None
    with pytest.raises(ValueError, match="leakage-safe split manifest is strictly required"):
        compile_packed_dataset(sequences_dir=seq_dir, manifest_path=None, output_dir=out_dataset)

    # 2. Fail closed when manifest file does not exist
    with pytest.raises(FileNotFoundError, match="Specified split manifest does not exist"):
        compile_packed_dataset(sequences_dir=seq_dir, manifest_path=tmp_path / "ghost.json", output_dir=out_dataset)

    # 3. Fail closed on data leakage (e.g. RUN-0001 in both train and validation)
    leak_manifest_path = tmp_path / "leaking_manifest.json"
    with open(leak_manifest_path, "w", encoding="utf-8") as f:
        json.dump({"splits": {"train": ["RUN-0001"], "validation": ["RUN-0001"], "test": []}}, f)

    with pytest.raises(ValueError, match="CRITICAL: Data leakage detected"):
        compile_packed_dataset(sequences_dir=seq_dir, manifest_path=leak_manifest_path, output_dir=out_dataset)

    # 4. Fail closed when manifest references runs not in sequences_dir
    missing_run_manifest = tmp_path / "missing_run_manifest.json"
    with open(missing_run_manifest, "w", encoding="utf-8") as f:
        json.dump({"splits": {"train": ["RUN-9999"], "validation": [], "test": []}}, f)

    with pytest.raises(FileNotFoundError, match="references 1 run.*not found"):
        compile_packed_dataset(sequences_dir=seq_dir, manifest_path=missing_run_manifest, output_dir=out_dataset)

