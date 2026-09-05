"""Phase 2.8: Recording- and subject-level split management, leakage protection, and split generation."""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from ml.datasets.schemas import (
    OBJECT_VOCAB,
    SplitManifest,
    SplitPartition,
    TARGET_VOCAB,
    VERB_VOCAB,
    WINDOW_SIZE,
)


class SplitGenerationError(ValueError):
    """Base exception for dataset splitting errors."""
    pass


class DataLeakageError(SplitGenerationError):
    """Raised when data leakage (overlap across partitions) is detected."""
    pass


class SplitValidationError(SplitGenerationError):
    """Raised when split configuration, ratios, or manifest contents violate invariants."""
    pass


class InsufficientGroupsError(SplitGenerationError):
    """Raised when there are fewer groups (subjects/runs) than required non-empty partitions."""
    pass


@dataclass(frozen=True)
class RecordingEntity:
    """Metadata representation of an individual recording/video for grouping."""
    recording_id: str
    run_id: str
    subject_id: str
    video_id: str
    total_windows: int
    verb_counts: dict[str, int]
    object_counts: dict[str, int]
    target_counts: dict[str, int]
    source_file: str


def resolve_entity_identity(
    file_path: Path,
    meta_dict: dict[str, Any] | None = None,
) -> tuple[str, str, str, str]:
    """
    Resolves canonical (run_id, subject_id, recording_id, video_id) from file path and metadata.
    Fails closed if run_id cannot be determined.
    """
    stem = file_path.stem
    meta = meta_dict or {}

    # 1. Canonical run_id: metadata primary, then parse prefix before camera tag
    run_id = meta.get("run_id")
    if not run_id:
        # Standard ASTRA-E convention: EXP001_RUN_001_CAM01 -> RUN-001 or RUN_001
        parts = stem.split("_CAM")[0].split("-CAM")[0]
        if "RUN" in parts:
            run_id = parts
        else:
            run_id = stem

    run_id = str(run_id).strip()
    if not run_id:
        raise SplitValidationError(f"Could not resolve canonical run_id for file: {file_path}")

    # 2. Canonical subject_id: metadata primary, default to ASTRONAUT-01
    subject_id = meta.get("subject_id") or meta.get("subject") or "ASTRONAUT-01"
    subject_id = str(subject_id).strip()

    # 3. Canonical recording_id and video_id
    recording_id = meta.get("recording_id") or stem
    video_id = meta.get("video_id") or f"EXP001_{run_id}_CAM01"

    return run_id, subject_id, str(recording_id), str(video_id)


def inspect_recording_statistics(
    npz_path: Path,
    meta_dict: dict[str, Any] | None = None,
    window_size: int = WINDOW_SIZE,
) -> tuple[int, dict[str, int], dict[str, int], dict[str, int]]:
    """
    Extracts total windows and class label occurrences from .npz or metadata.
    """
    total_windows = 0
    verb_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    # 1. Check if .npz exists and has label arrays
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "X" in data:
                    total_windows = len(data["X"])
                elif "features" in data:
                    n_frames = len(data["features"])
                    total_windows = max(0, n_frames - window_size + 1)

                if "verbs" in data:
                    for v in data["verbs"]:
                        idx = int(v)
                        name = VERB_VOCAB[idx] if 0 <= idx < len(VERB_VOCAB) else "UNKNOWN"
                        verb_counts[name] += 1
                if "objects" in data:
                    for o in data["objects"]:
                        idx = int(o)
                        name = OBJECT_VOCAB[idx] if 0 <= idx < len(OBJECT_VOCAB) else "UNKNOWN"
                        object_counts[name] += 1
                if "targets" in data:
                    for t in data["targets"]:
                        idx = int(t)
                        name = TARGET_VOCAB[idx] if 0 <= idx < len(TARGET_VOCAB) else "UNKNOWN"
                        target_counts[name] += 1
        except Exception:
            pass

    # 2. If no labels from .npz, inspect metadata segments
    if not verb_counts and meta_dict and "segments" in meta_dict:
        for seg in meta_dict.get("segments", []):
            v = seg.get("verb", "IDLE")
            o = seg.get("object") or "NONE"
            t = seg.get("target") or "NONE"
            verb_counts[v] += 1
            object_counts[o] += 1
            target_counts[t] += 1

    return total_windows, dict(verb_counts), dict(object_counts), dict(target_counts)


def gather_dataset_entities(
    sequences_dir: str | Path,
    metadata_dir: str | Path | None = None,
    window_size: int = WINDOW_SIZE,
) -> list[RecordingEntity]:
    """
    Scans sequences directory and compiles RecordingEntity objects for all available runs.
    """
    seq_dir = Path(sequences_dir)
    meta_dir = Path(metadata_dir) if metadata_dir else seq_dir

    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequences directory does not exist: {seq_dir}")

    # Gather .npz sequence files
    npz_files = sorted(list(seq_dir.glob("*.npz")))
    # If no npz files, search for metadata JSONs (e.g. pre-extraction stage)
    if not npz_files:
        json_files = sorted(list(meta_dir.glob("*_meta.json")) + list(meta_dir.glob("*.json")))
        if not json_files:
            raise FileNotFoundError(f"No .npz sequences or .json metadata found in {seq_dir} or {meta_dir}")
        stems = [f.stem.replace("_meta", "") for f in json_files]
        candidate_files = [(meta_dir / f"{s}_meta.json", s) for s in stems]
    else:
        candidate_files = [(f, f.stem) for f in npz_files]

    entities: list[RecordingEntity] = []
    seen_recordings: set[str] = set()

    for f_path, stem in candidate_files:
        # Resolve metadata JSON path
        meta_file = meta_dir / f"{stem}_meta.json"
        if not meta_file.exists():
            meta_file = meta_dir / f"{stem}.json"
        if not meta_file.exists():
            meta_file = seq_dir / f"{stem}_meta.json"
        if not meta_file.exists():
            meta_file = seq_dir / f"{stem}.json"

        meta_dict: dict[str, Any] = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as mf:
                    meta_dict = json.load(mf)
            except Exception as e:
                raise SplitValidationError(f"Corrupt metadata JSON at {meta_file}: {e}")

        npz_target = f_path if f_path.suffix == ".npz" else seq_dir / f"{stem}.npz"

        run_id, subject_id, recording_id, video_id = resolve_entity_identity(f_path, meta_dict)

        if recording_id in seen_recordings:
            raise SplitValidationError(f"Duplicate recording identifier found: '{recording_id}'")
        seen_recordings.add(recording_id)

        tot_windows, v_counts, o_counts, t_counts = inspect_recording_statistics(
            npz_target, meta_dict, window_size=window_size
        )

        entities.append(
            RecordingEntity(
                recording_id=recording_id,
                run_id=run_id,
                subject_id=subject_id,
                video_id=video_id,
                total_windows=tot_windows,
                verb_counts=v_counts,
                object_counts=o_counts,
                target_counts=t_counts,
                source_file=str(f_path),
            )
        )

    return entities


def generate_leakage_safe_splits(
    sequences_dir: str | Path,
    output_manifest: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    group_by: str = "subject",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    window_size: int = WINDOW_SIZE,
    dataset_version: str = "2026.09.05",
) -> SplitManifest:
    """
    Deterministic, leakage-safe dataset partitioner implementing Phase 2.8 requirements:
      - Atomic grouping: subject (preferred) or run. Never individual temporal windows.
      - Multi-camera binding: cameras of the same run are bound together.
      - Disjointness guarantees: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅.
      - Fail closed: fails loudly on ratio deviations, insufficient subjects, or overlap.
      - Auditing: computes per-partition distributions and flags rare classes.
    """
    # 1. Validate Ratios
    for name, r in (("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)):
        if r < 0.0 or r > 1.0:
            raise SplitValidationError(f"Ratio {name}={r} is out of bounds [0.0, 1.0]")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-5:
        raise SplitValidationError(f"Ratios must sum to 1.0 (got {ratio_sum:.6f})")

    if group_by not in ("subject", "run"):
        raise SplitValidationError(f"Invalid group_by='{group_by}'. Supported: 'subject', 'run'.")

    # 2. Gather Entities
    entities = gather_dataset_entities(sequences_dir, metadata_dir, window_size=window_size)
    if not entities:
        raise SplitValidationError(f"No recording entities found in {sequences_dir}")

    # 3. Group Entities
    group_to_entities: dict[str, list[RecordingEntity]] = defaultdict(list)
    for ent in entities:
        key = ent.subject_id if group_by == "subject" else ent.run_id
        group_to_entities[key].append(ent)

    sorted_groups = sorted(list(group_to_entities.keys()))
    num_groups = len(sorted_groups)

    # Determine required non-empty partitions
    active_splits = [k for k, r in (("train", train_ratio), ("validation", val_ratio), ("test", test_ratio)) if r > 0.0]
    min_required = len(active_splits)

    if num_groups < min_required:
        if group_by == "subject":
            raise InsufficientGroupsError(
                f"Subject-disjoint splitting requested with group_by='subject', but only {num_groups} "
                f"distinct subject(s) were found: {sorted_groups}. Minimum required for disjoint "
                f"partitions ({', '.join(active_splits)}) is {min_required}. "
                "Refusing to silently downgrade to run-level splitting. Explicitly configure "
                "--group-by run if run-level splitting is intended."
            )
        else:
            raise InsufficientGroupsError(
                f"Run-level splitting requested with group_by='run', but only {num_groups} "
                f"distinct run(s) were found: {sorted_groups}. Minimum required is {min_required}."
            )

    # 4. Deterministic Partitioning (explicit seed, deterministic sorting first)
    rng = random.Random(seed)
    shuffled_groups = list(sorted_groups)
    rng.shuffle(shuffled_groups)

    # Calculate partition group counts
    n_train = int(round(train_ratio * num_groups)) if train_ratio > 0 else 0
    n_val = int(round(val_ratio * num_groups)) if val_ratio > 0 else 0
    n_test = num_groups - n_train - n_val

    # Ensure all active partitions have at least 1 group
    if "train" in active_splits and n_train < 1:
        n_train = 1
    if "validation" in active_splits and n_val < 1:
        n_val = 1
    if "test" in active_splits and n_test < 1:
        # Borrow from largest partition
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        n_test = 1

    # Re-verify non-negative and exact sum
    total_assigned = n_train + n_val + n_test
    if total_assigned != num_groups:
        diff = num_groups - total_assigned
        n_train += diff

    train_keys = sorted(shuffled_groups[:n_train])
    val_keys = sorted(shuffled_groups[n_train : n_train + n_val])
    test_keys = sorted(shuffled_groups[n_train + n_val:])

    # Re-verify no empty partitions if requested
    for s_name, s_keys in (("train", train_keys), ("validation", val_keys), ("test", test_keys)):
        if s_name in active_splits and not s_keys:
            raise SplitValidationError(f"Partition '{s_name}' ended up empty with {num_groups} groups. Rejecting empty partition.")

    # 5. Build Split Containers and Collect Statistics
    split_keys_map = {
        "train": train_keys,
        "validation": val_keys,
        "test": test_keys,
    }

    partitions: dict[str, SplitPartition] = {}
    splits_dict: dict[str, list[str]] = {}
    stats_dict: dict[str, Any] = {}
    class_presence: dict[str, dict[str, set[str]]] = {
        "verb": defaultdict(set),
        "object": defaultdict(set),
        "target": defaultdict(set),
    }

    for s_name in ("train", "validation", "test"):
        g_keys = split_keys_map[s_name]
        p_entities = [ent for k in g_keys for ent in group_to_entities[k]]

        p_subjs = sorted(list({ent.subject_id for ent in p_entities}))
        p_runs = sorted(list({ent.run_id for ent in p_entities}))
        p_recs = sorted(list({ent.recording_id for ent in p_entities}))
        tot_windows = sum(ent.total_windows for ent in p_entities)

        verb_dist: Counter[str] = Counter()
        obj_dist: Counter[str] = Counter()
        tgt_dist: Counter[str] = Counter()

        for ent in p_entities:
            for v, c in ent.verb_counts.items():
                verb_dist[v] += c
                class_presence["verb"][v].add(s_name)
            for o, c in ent.object_counts.items():
                obj_dist[o] += c
                class_presence["object"][o].add(s_name)
            for t, c in ent.target_counts.items():
                tgt_dist[t] += c
                class_presence["target"][t].add(s_name)

        partitions[s_name] = SplitPartition(
            subjects=p_subjs,
            runs=p_runs,
            recordings=p_recs,
        )
        splits_dict[s_name] = p_runs
        stats_dict[s_name] = {
            "num_subjects": len(p_subjs),
            "num_runs": len(p_runs),
            "num_recordings": len(p_recs),
            "num_windows": tot_windows,
            "verb_distribution": dict(sorted(verb_dist.items())),
            "object_distribution": dict(sorted(obj_dist.items())),
            "target_distribution": dict(sorted(tgt_dist.items())),
        }

    # 6. Disjointness Audit
    train_runs = set(partitions["train"].runs)
    val_runs = set(partitions["validation"].runs)
    test_runs = set(partitions["test"].runs)

    train_subjs = set(partitions["train"].subjects)
    val_subjs = set(partitions["validation"].subjects)
    test_subjs = set(partitions["test"].subjects)

    train_recs = set(partitions["train"].recordings)
    val_recs = set(partitions["validation"].recordings)
    test_recs = set(partitions["test"].recordings)

    # Run overlap check
    if (train_runs & val_runs) or (train_runs & test_runs) or (val_runs & test_runs):
        raise DataLeakageError(
            f"CRITICAL: Run overlap detected across splits!\n"
            f"  Train & Val: {train_runs & val_runs}\n"
            f"  Train & Test: {train_runs & test_runs}\n"
            f"  Val & Test: {val_runs & test_runs}"
        )

    # Recording overlap check
    if (train_recs & val_recs) or (train_recs & test_recs) or (val_recs & test_recs):
        raise DataLeakageError(
            f"CRITICAL: Recording overlap detected across splits!\n"
            f"  Train & Val: {train_recs & val_recs}\n"
            f"  Train & Test: {train_recs & test_recs}\n"
            f"  Val & Test: {val_recs & test_recs}"
        )

    # Subject overlap check (if subject-disjoint mode)
    subject_disjoint_enforced = (group_by == "subject")
    if subject_disjoint_enforced:
        if (train_subjs & val_subjs) or (train_subjs & test_subjs) or (val_subjs & test_subjs):
            raise DataLeakageError(
                f"CRITICAL: Subject overlap detected under group_by='subject'!\n"
                f"  Train & Val: {train_subjs & val_subjs}\n"
                f"  Train & Test: {train_subjs & test_subjs}\n"
                f"  Val & Test: {val_subjs & test_subjs}"
            )

    disjointness_audit = {
        "mutually_disjoint_runs": True,
        "mutually_disjoint_recordings": True,
        "mutually_disjoint_subjects": bool(not (train_subjs & val_subjs or train_subjs & test_subjs or val_subjs & test_subjs)),
        "subject_disjoint_enforced": subject_disjoint_enforced,
    }

    # 7. Rare Class Detection
    rare_classes: list[dict[str, Any]] = []
    for cat in ("verb", "object", "target"):
        vocab = VERB_VOCAB if cat == "verb" else OBJECT_VOCAB if cat == "object" else TARGET_VOCAB
        for c_name in vocab:
            present_in = sorted(list(class_presence[cat].get(c_name, set())))
            if present_in and len(present_in) < len(active_splits):
                missing = [s for s in active_splits if s not in present_in]
                warning_msg = (
                    f"WARNING: {cat}='{c_name}' appears only in {len(present_in)} partition(s) "
                    f"({present_in}). Missing from: {missing}. It cannot be evaluated in all splits."
                )
                rare_classes.append({
                    "category": cat,
                    "class_name": c_name,
                    "present_in_splits": present_in,
                    "missing_splits": missing,
                    "warning": warning_msg,
                })

    manifest = SplitManifest(
        schema_version="1.0",
        dataset_version=dataset_version,
        split_algorithm="group_disjoint_v1.0",
        seed=seed,
        group_by=group_by,
        ratios={"train": train_ratio, "validation": val_ratio, "test": test_ratio},
        splits=splits_dict,
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        statistics=stats_dict,
        disjointness_audit=disjointness_audit,
        rare_classes=rare_classes,
        metadata={
            "created_at": time.time(),
            "sequences_dir": str(sequences_dir),
            "metadata_dir": str(metadata_dir) if metadata_dir else str(sequences_dir),
            "total_entities": len(entities),
            "total_groups": num_groups,
            "window_size": window_size,
        },
    )

    # Save to disk if requested
    if output_manifest:
        out_p = Path(output_manifest)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

    return manifest


def validate_split_manifest(
    manifest_input: str | Path | dict[str, Any],
    known_runs: Sequence[str] | None = None,
    known_recordings: Sequence[str] | None = None,
    known_subjects: Sequence[str] | None = None,
) -> tuple[bool, list[str]]:
    """
    Validates an existing split manifest against all Phase 2.8 invariants.
    Returns (is_valid, list_of_violations).
    """
    violations: list[str] = []

    if isinstance(manifest_input, (str, Path)):
        p = Path(manifest_input)
        if not p.exists():
            return False, [f"Manifest file does not exist: {p}"]
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return False, [f"Corrupt manifest JSON: {e}"]
    else:
        data = dict(manifest_input)

    # Extract splits dict
    raw_splits = data.get("splits")
    if not isinstance(raw_splits, dict):
        # Fallback to train/validation/test partition objects
        if "train" in data and isinstance(data["train"], dict) and "runs" in data["train"]:
            raw_splits = {
                "train": data["train"].get("runs", []),
                "validation": data.get("validation", {}).get("runs", []),
                "test": data.get("test", {}).get("runs", []),
            }
        else:
            violations.append("Manifest missing 'splits' dict or 'train/validation/test' partitions.")
            return False, violations

    train_runs = list(raw_splits.get("train", []))
    val_runs = list(raw_splits.get("validation", raw_splits.get("val", [])))
    test_runs = list(raw_splits.get("test", []))

    # 1. Ratios validation
    ratios = data.get("ratios")
    if ratios and isinstance(ratios, dict):
        for rk in ("train", "validation", "test"):
            r_val = ratios.get(rk, ratios.get("val" if rk == "validation" else rk))
            if r_val is not None:
                if r_val < 0.0 or r_val > 1.0:
                    violations.append(f"Invalid ratio {rk}={r_val} outside [0.0, 1.0].")
        r_sum = sum(float(v) for v in ratios.values())
        if abs(r_sum - 1.0) > 1e-5:
            violations.append(f"Ratios do not sum to 1.0 (sum={r_sum:.6f}).")

    # 2. Duplicate runs within partition
    for s_name, runs_list in (("train", train_runs), ("validation", val_runs), ("test", test_runs)):
        if len(runs_list) != len(set(runs_list)):
            dups = [r for r, c in Counter(runs_list).items() if c > 1]
            violations.append(f"Duplicate run IDs found within partition '{s_name}': {dups}")

    # 3. Disjointness across partitions
    train_set = set(train_runs)
    val_set = set(val_runs)
    test_set = set(test_runs)

    if train_set & val_set:
        violations.append(f"Run leakage between train and validation: {train_set & val_set}")
    if train_set & test_set:
        violations.append(f"Run leakage between train and test: {train_set & test_set}")
    if val_set & test_set:
        violations.append(f"Run leakage between validation and test: {val_set & test_set}")

    # 4. Subject Disjointness (if subject mode)
    group_by = data.get("group_by", "run")
    if group_by == "subject":
        train_subjs = set(data.get("train", {}).get("subjects", []))
        val_subjs = set(data.get("validation", {}).get("subjects", []))
        test_subjs = set(data.get("test", {}).get("subjects", []))

        if train_subjs & val_subjs:
            violations.append(f"Subject leakage between train and validation: {train_subjs & val_subjs}")
        if train_subjs & test_subjs:
            violations.append(f"Subject leakage between train and test: {train_subjs & test_subjs}")
        if val_subjs & test_subjs:
            violations.append(f"Subject leakage between validation and test: {val_subjs & test_subjs}")

    # 5. Empty partition checks (if expected non-empty)
    if not train_runs and (not ratios or ratios.get("train", 1.0) > 0.0):
        violations.append("Partition 'train' is unexpectedly empty.")

    # 6. Known runs verification (if provided)
    if known_runs is not None:
        known_set = set(known_runs)
        assigned_runs = train_set | val_set | test_set

        unknown = assigned_runs - known_set
        if unknown:
            violations.append(f"Manifest references unknown runs not present in dataset: {sorted(list(unknown))}")

        missing = known_set - assigned_runs
        if missing:
            violations.append(f"Known runs missing from manifest split assignment: {sorted(list(missing))}")

    return len(violations) == 0, violations


class SplitManager:
    """Manages recording-level splits strictly by run/subject to avoid temporal frame leakage."""

    def __init__(self, manifest_path: str | Path = "data/manifests/dataset_manifest.json") -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest: dict[str, Any] | None = None
        self.raw_data: dict[str, Any] = {}
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.raw_data = json.load(f)
                self.manifest = self.raw_data

    def get_split_runs(self, split_name: str) -> list[str]:
        """Return list of run IDs assigned to split ('train', 'val'/'validation', 'test')."""
        if not self.manifest:
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        # Normalize split name
        s_key = "validation" if split_name in ("val", "validation") else split_name

        # Check 'splits' dict first
        splits = self.manifest.get("splits")
        if isinstance(splits, dict):
            if s_key in splits:
                return list(splits[s_key])
            if split_name in splits:
                return list(splits[split_name])

        # Check structured partition object
        part = self.manifest.get(s_key, self.manifest.get(split_name))
        if isinstance(part, dict) and "runs" in part:
            return list(part["runs"])

        return []

    def get_split_subjects(self, split_name: str) -> list[str]:
        """Return list of subject IDs assigned to split."""
        if not self.manifest:
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
        s_key = "validation" if split_name in ("val", "validation") else split_name
        part = self.manifest.get(s_key, self.manifest.get(split_name))
        if isinstance(part, dict) and "subjects" in part:
            return list(part["subjects"])
        return []

    def get_split_recordings(self, split_name: str) -> list[str]:
        """Return list of recording IDs assigned to split."""
        if not self.manifest:
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
        s_key = "validation" if split_name in ("val", "validation") else split_name
        part = self.manifest.get(s_key, self.manifest.get(split_name))
        if isinstance(part, dict) and "recordings" in part:
            return list(part["recordings"])
        return []

    def verify_no_leakage(self, subject_disjoint: bool | None = None) -> bool:
        """Verify that train, val, and test partitions are completely disjoint."""
        if not self.manifest:
            return True

        is_valid, violations = validate_split_manifest(self.manifest)
        if not is_valid:
            raise DataLeakageError("Leakage verification failed:\n" + "\n".join(f"  - {v}" for v in violations))
        return True

