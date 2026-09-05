"""Unit tests for Phase 2.8: Leakage-Safe Run/Subject-Level Split Generation."""

import json
from pathlib import Path
import numpy as np
import pytest

from ml.datasets.schemas import (
    OBJECT_VOCAB,
    SplitManifest,
    TARGET_VOCAB,
    VERB_VOCAB,
)
from ml.datasets.splits import (
    DataLeakageError,
    InsufficientGroupsError,
    SplitManager,
    SplitValidationError,
    generate_leakage_safe_splits,
    validate_split_manifest,
)
from scripts.cloud.build_packed_dataset import compile_packed_dataset


def _create_mock_dataset(
    root: Path,
    subjects: list[str],
    runs_per_subject: int = 2,
    num_cameras: int = 1,
    T: int = 60,
    special_classes: dict[str, dict[str, str]] | None = None,
) -> Path:
    """Helper creating mock synthetic dataset with sequence NPZs and metadata JSONs."""
    seq_dir = root / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    run_counter = 1
    for s_id in subjects:
        for _ in range(runs_per_subject):
            run_id = f"RUN-{run_counter:04d}"
            run_counter += 1

            for c_idx in range(1, num_cameras + 1):
                cam_id = f"CAM{c_idx:02d}"
                stem = f"EXP001_{run_id}_{cam_id}"

                features = np.zeros((T, 26), dtype=np.float32)
                features[:, 0] = 0.5
                features[:, 23] = 0.95

                verbs = np.zeros(T, dtype=np.int64)
                objects = np.zeros(T, dtype=np.int64)
                targets = np.zeros(T, dtype=np.int64)

                # If special classes configured for this run
                if special_classes and run_id in special_classes:
                    sc = special_classes[run_id]
                    if "verb" in sc:
                        verbs[:] = VERB_VOCAB.index(sc["verb"])
                    if "object" in sc:
                        objects[:] = OBJECT_VOCAB.index(sc["object"])
                    if "target" in sc:
                        targets[:] = TARGET_VOCAB.index(sc["target"])

                npz_path = seq_dir / f"{stem}.npz"
                np.savez_compressed(
                    npz_path,
                    features=features,
                    verbs=verbs,
                    objects=objects,
                    targets=targets,
                )

                meta = {
                    "recording_id": stem,
                    "video_id": stem,
                    "experiment_id": "EXP001",
                    "run_id": run_id,
                    "subject_id": s_id,
                    "camera_id": cam_id,
                    "duration_seconds": float(T / 30.0),
                    "fps": 30.0,
                    "width": 640,
                    "height": 480,
                    "scenario_type": "nominal",
                    "random_seed": 42,
                    "segments": [],
                }
                with open(seq_dir / f"{stem}_meta.json", "w", encoding="utf-8") as mf:
                    json.dump(meta, mf, indent=2)

    return seq_dir


def test_basic_split_assignment(tmp_path: Path):
    """1. Verify 10 subjects are assigned across train (7), val, and test with 70/15/15 ratios."""
    subjects = [f"ASTRO_{chr(65+i)}" for i in range(10)]  # ASTRO_A to ASTRO_J
    seq_dir = _create_mock_dataset(tmp_path, subjects=subjects, runs_per_subject=2)

    manifest = generate_leakage_safe_splits(
        sequences_dir=seq_dir,
        group_by="subject",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    assert len(manifest.train.subjects) == 7
    assert len(manifest.validation.subjects) > 0
    assert len(manifest.test.subjects) > 0
    assert len(manifest.validation.subjects) + len(manifest.test.subjects) == 3

    # All 10 subjects must be uniquely assigned
    all_assigned_subjects = set(manifest.train.subjects) | set(manifest.validation.subjects) | set(manifest.test.subjects)
    assert all_assigned_subjects == set(subjects)



def test_mutual_exclusion_runs(tmp_path: Path):
    """2. Verify mutual exclusion: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅."""
    subjects = [f"ASTRO_{i:02d}" for i in range(6)]
    seq_dir = _create_mock_dataset(tmp_path, subjects=subjects, runs_per_subject=3)

    manifest = generate_leakage_safe_splits(
        sequences_dir=seq_dir,
        group_by="run",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_runs = set(manifest.train.runs)
    val_runs = set(manifest.validation.runs)
    test_runs = set(manifest.test.runs)

    assert len(train_runs & val_runs) == 0
    assert len(train_runs & test_runs) == 0
    assert len(val_runs & test_runs) == 0


def test_subject_isolation(tmp_path: Path):
    """3. Verify all runs belonging to a subject remain together in subject-disjoint mode."""
    subjects = ["ASTRO_ALPHA", "ASTRO_BETA", "ASTRO_GAMMA", "ASTRO_DELTA"]
    seq_dir = _create_mock_dataset(tmp_path, subjects=subjects, runs_per_subject=3)

    manifest = generate_leakage_safe_splits(
        sequences_dir=seq_dir,
        group_by="subject",
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )

    # In subject mode, subject sets must be mutually exclusive
    train_subjs = set(manifest.train.subjects)
    val_subjs = set(manifest.validation.subjects)
    test_subjs = set(manifest.test.subjects)

    assert len(train_subjs & val_subjs) == 0
    assert len(train_subjs & test_subjs) == 0
    assert len(val_subjs & test_subjs) == 0

    # Verify no run from a subject appears in another partition
    for s_name, part in (("train", manifest.train), ("validation", manifest.validation), ("test", manifest.test)):
        for s_id in part.subjects:
            # All runs of s_id should only exist in this partition
            for other_name, other_part in (("train", manifest.train), ("validation", manifest.validation), ("test", manifest.test)):
                if other_name != s_name:
                    assert not (set(part.runs) & set(other_part.runs))


def test_multi_camera_isolation(tmp_path: Path):
    """4. Verify multi-camera views (CAM01, CAM02) of the same physical run remain in the same partition."""
    subjects = ["ASTRO_1", "ASTRO_2", "ASTRO_3"]
    # 2 cameras per run
    seq_dir = _create_mock_dataset(tmp_path, subjects=subjects, runs_per_subject=2, num_cameras=2)

    manifest = generate_leakage_safe_splits(
        sequences_dir=seq_dir,
        group_by="run",
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        seed=42,
    )

    # For every run in train, all its camera recordings must be in train
    for s_name, part in (("train", manifest.train), ("validation", manifest.validation), ("test", manifest.test)):
        for r_id in part.runs:
            cam1_rec = f"EXP001_{r_id}_CAM01"
            cam2_rec = f"EXP001_{r_id}_CAM02"
            assert cam1_rec in part.recordings
            assert cam2_rec in part.recordings

            # Must NOT be in any other partition
            for other_name, other_part in (("train", manifest.train), ("validation", manifest.validation), ("test", manifest.test)):
                if other_name != s_name:
                    assert cam1_rec not in other_part.recordings
                    assert cam2_rec not in other_part.recordings


def test_determinism(tmp_path: Path):
    """5. Verify identical metadata and configuration produce bit-for-bit identical manifests."""
    subjects = [f"ASTRO_{i}" for i in range(8)]
    seq_dir = _create_mock_dataset(tmp_path, subjects=subjects, runs_per_subject=2)

    manifest1 = generate_leakage_safe_splits(sequences_dir=seq_dir, seed=42)
    manifest2 = generate_leakage_safe_splits(sequences_dir=seq_dir, seed=42)

    assert manifest1.splits == manifest2.splits
    assert manifest1.train.runs == manifest2.train.runs
    assert manifest1.validation.runs == manifest2.validation.runs
    assert manifest1.test.runs == manifest2.test.runs
    assert manifest1.statistics == manifest2.statistics


def test_different_seed(tmp_path: Path):
    """6. Verify different random seeds produce different valid permutations."""
    subjects = [f"ASTRO_{i}" for i in range(12)]
    seq_dir = _create_mock_dataset(tmp_path, subjects=subjects, runs_per_subject=2)

    m1 = generate_leakage_safe_splits(sequences_dir=seq_dir, seed=42)
    m2 = generate_leakage_safe_splits(sequences_dir=seq_dir, seed=999)

    # Both must be valid and disjoint
    assert set(m1.train.subjects) & set(m1.validation.subjects) == set()
    assert set(m2.train.subjects) & set(m2.validation.subjects) == set()
    # But permutations should differ
    assert m1.train.subjects != m2.train.subjects


def test_missing_metadata_fails_closed(tmp_path: Path):
    """7. Verify file with completely unresolvable identity raises SplitValidationError."""
    seq_dir = tmp_path / "sequences"
    seq_dir.mkdir()

    # Create empty directory
    with pytest.raises((SplitValidationError, FileNotFoundError)):
        generate_leakage_safe_splits(sequences_dir=seq_dir)



def test_duplicate_run_fails_closed():
    """8. Verify validator fails closed when duplicate runs exist within a partition."""
    bad_manifest = {
        "schema_version": "1.0",
        "group_by": "run",
        "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "splits": {
            "train": ["RUN-0001", "RUN-0001"],  # DUPLICATE!
            "validation": ["RUN-0002"],
            "test": ["RUN-0003"],
        },
    }
    is_valid, violations = validate_split_manifest(bad_manifest)
    assert not is_valid
    assert any("Duplicate run IDs" in v for v in violations)


def test_overlap_fails_closed():
    """9. Verify validator fails closed when overlap is detected across splits."""
    leaking_manifest = {
        "schema_version": "1.0",
        "group_by": "subject",
        "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "splits": {
            "train": ["RUN-0001", "RUN-0002"],
            "validation": ["RUN-0002"],  # RUN LEAKAGE!
            "test": ["RUN-0003"],
        },
        "train": {"subjects": ["ASTRO_1", "ASTRO_2"]},
        "validation": {"subjects": ["ASTRO_2"]},  # SUBJECT LEAKAGE!
        "test": {"subjects": ["ASTRO_3"]},
    }
    is_valid, violations = validate_split_manifest(leaking_manifest)
    assert not is_valid
    assert any("Run leakage" in v for v in violations)
    assert any("Subject leakage" in v for v in violations)


def test_unknown_run_fails_closed():
    """10. Verify validator fails closed when manifest references unknown runs."""
    manifest = {
        "schema_version": "1.0",
        "group_by": "run",
        "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "splits": {
            "train": ["RUN-0001", "RUN-9999"],  # UNKNOWN RUN!
            "validation": ["RUN-0002"],
            "test": ["RUN-0003"],
        },
    }
    is_valid, violations = validate_split_manifest(
        manifest, known_runs=["RUN-0001", "RUN-0002", "RUN-0003"]
    )
    assert not is_valid
    assert any("references unknown runs" in v for v in violations)


def test_invalid_ratios_fails_closed(tmp_path: Path):
    """11. Verify ratios summing != 1.0 or out of bounds fail closed."""
    subjects = ["ASTRO_A", "ASTRO_B", "ASTRO_C"]
    seq_dir = _create_mock_dataset(tmp_path, subjects=subjects)

    # Sum > 1.0
    with pytest.raises(SplitValidationError, match="must sum to 1.0"):
        generate_leakage_safe_splits(seq_dir, train_ratio=0.8, val_ratio=0.2, test_ratio=0.2)

    # Negative ratio
    with pytest.raises(SplitValidationError, match="out of bounds"):
        generate_leakage_safe_splits(seq_dir, train_ratio=-0.1, val_ratio=0.5, test_ratio=0.6)


def test_empty_partition_fails_closed(tmp_path: Path):
    """12. Verify splitting fails closed when active partitions cannot be filled."""
    # 2 runs, but 3 non-empty partitions requested
    subjects = ["ASTRO_A"]
    seq_dir = _create_mock_dataset(tmp_path, subjects=subjects, runs_per_subject=2)

    with pytest.raises(InsufficientGroupsError, match="Minimum required is 3"):
        generate_leakage_safe_splits(
            seq_dir, group_by="run", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )


def test_insufficient_subjects_fails_closed(tmp_path: Path):
    """13. Verify subject-disjoint mode fails closed on < 3 subjects and refuses silent downgrade."""
    # Only 2 subjects, but 3 splits requested
    subjects = ["ASTRO_ALPHA", "ASTRO_BETA"]
    seq_dir = _create_mock_dataset(tmp_path, subjects=subjects, runs_per_subject=5)

    with pytest.raises(InsufficientGroupsError, match="Subject-disjoint splitting requested.*only 2 distinct subject"):
        generate_leakage_safe_splits(
            seq_dir, group_by="subject", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )


def test_rare_classes_reported(tmp_path: Path):
    """14. Verify rare classes present in only 1 group are detected and flagged in manifest."""
    subjects = ["ASTRO_1", "ASTRO_2", "ASTRO_3", "ASTRO_4"]
    special = {
        "RUN-0001": {"verb": "OPEN_CONTAINER", "object": "CONTAINER"},  # only in ASTRO_1
    }
    seq_dir = _create_mock_dataset(tmp_path, subjects=subjects, runs_per_subject=2, special_classes=special)

    manifest = generate_leakage_safe_splits(
        sequences_dir=seq_dir,
        group_by="subject",
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )

    # OPEN_CONTAINER appears only in ASTRO_1, so it cannot appear in all 3 partitions
    rare_names = [rc["class_name"] for rc in manifest.rare_classes]
    assert "OPEN_CONTAINER" in rare_names


def test_integration_split_to_packed_dataset(tmp_path: Path):
    """15. End-to-end integration: metadata -> create_splits -> build_packed_dataset -> verify partitions."""
    subjects = ["ASTRO_1", "ASTRO_2", "ASTRO_3", "ASTRO_4"]
    seq_dir = _create_mock_dataset(tmp_path, subjects=subjects, runs_per_subject=2, T=40)

    manifest_path = tmp_path / "splits.json"
    manifest = generate_leakage_safe_splits(
        sequences_dir=seq_dir,
        output_manifest=manifest_path,
        group_by="subject",
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )
    assert manifest_path.exists()

    # Compile packed dataset using generated manifest
    out_packed = tmp_path / "packed_dataset"
    packed_meta = compile_packed_dataset(
        sequences_dir=seq_dir,
        manifest_path=manifest_path,
        output_dir=out_packed,
        window_size=30,
    )

    # Verify packed files exist
    assert (out_packed / "train" / "features.npy").exists()
    assert (out_packed / "validation" / "features.npy").exists()
    assert (out_packed / "test" / "features.npy").exists()

    # Verify train labels only contain train runs
    with open(out_packed / "train" / "labels.json", "r", encoding="utf-8") as f:
        train_labels = json.load(f)
    train_label_runs = {item["run_id"] for item in train_labels}
    assert train_label_runs.issubset(set(manifest.train.runs))
    assert not (train_label_runs & set(manifest.validation.runs))
    assert not (train_label_runs & set(manifest.test.runs))
