"""Offline Quality Gate & Feature Contract Validator: Verifies 26-D feature integrity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
import numpy as np
import torch

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.activity.models.lstm import CausalTemporalActionLSTM
from ml.datasets.schemas import FEATURE_PROVENANCE, NUM_FEATURES


def validate_feature_file(npz_file: str | Path) -> tuple[bool, list[str]]:
    """
    Validates an individual .npz sequence file against the frozen 26-D feature contract.
    Returns (is_valid, list_of_violations).
    """
    p = Path(npz_file)
    violations: list[str] = []

    try:
        with np.load(p, allow_pickle=True) as data:
            if "features" not in data:
                return False, ["Missing 'features' array in NPZ."]
            features = data["features"]
    except Exception as e:
        return False, [f"Corrupt or unreadable NPZ archive: {e}"]

    # 1. Shape check
    if features.ndim != 2:
        violations.append(f"Features must be 2D array (T, 26), got shape {features.shape}.")
    elif features.shape[1] != NUM_FEATURES:
        violations.append(f"Feature dimension must be {NUM_FEATURES}, got {features.shape[1]}.")

    # 2. NaN / Inf check
    if np.isnan(features).any():
        nan_count = np.isnan(features).sum()
        violations.append(f"Detected {nan_count} NaN values in feature matrix.")
    if np.isinf(features).any():
        inf_count = np.isinf(features).sum()
        violations.append(f"Detected {inf_count} Inf values in feature matrix.")

    if violations:
        return False, violations

    # 3. Contract Bounds & Semantics
    # Coordinates [0, 1]
    coord_indices = [0, 1, 4, 5, 8, 9]  # hand_x, hand_y, red_x, red_y, yellow_x, yellow_y
    for idx in coord_indices:
        vals = features[:, idx]
        if (vals < -0.05).any() or (vals > 1.05).any():
            fname = FEATURE_PROVENANCE[idx][0]
            violations.append(f"Coordinate '{fname}' out of bounds [0, 1]: min={vals.min():.3f}, max={vals.max():.3f}")

    # Distances [0, 1]
    dist_indices = [12, 13, 14, 15, 16]
    for idx in dist_indices:
        vals = features[:, idx]
        if (vals < -0.01).any() or (vals > 1.05).any():
            fname = FEATURE_PROVENANCE[idx][0]
            violations.append(f"Distance '{fname}' out of bounds [0, 1]: min={vals.min():.3f}, max={vals.max():.3f}")

    # First-order Derivatives [-3.0, 3.0]
    deriv_indices = [17, 18, 19, 20]
    for idx in deriv_indices:
        vals = features[:, idx]
        if (vals < -3.05).any() or (vals > 3.05).any():
            fname = FEATURE_PROVENANCE[idx][0]
            violations.append(f"Derivative '{fname}' exceeded clamped range [-3.0, 3.0]: min={vals.min():.3f}, max={vals.max():.3f}")

    # Tracking Confidences [0, 1]
    conf_indices = [23, 24, 25]
    for idx in conf_indices:
        vals = features[:, idx]
        if (vals < 0.0).any() or (vals > 1.0).any():
            fname = FEATURE_PROVENANCE[idx][0]
            violations.append(f"Confidence '{fname}' out of bounds [0, 1]: min={vals.min():.3f}, max={vals.max():.3f}")

    is_valid = len(violations) == 0
    return is_valid, violations


def validate_feature_directory(data_dir: str | Path) -> bool:
    """Validate all .npz feature files in directory."""
    d = Path(data_dir)
    print("\n" + "=" * 70)
    print("║" + "ASTRA-E 26-D FEATURE CONTRACT AUDITOR".center(68) + "║")
    print("=" * 70)
    print(f"  Target: {d}")

    npz_files = sorted(list(d.glob("*.npz")) + list(d.glob("**/*.npz")))
    if not npz_files:
        print(f"  [ERROR] No .npz files found in {d}")
        return False

    total_valid = 0
    total_invalid = 0

    for idx, f in enumerate(npz_files, start=1):
        ok, viols = validate_feature_file(f)
        if ok:
            total_valid += 1
        else:
            total_invalid += 1
            print(f"  ❌ FAIL: {f.name}")
            for v in viols:
                print(f"      - {v}")

    # 4. Neural Network Forward-Pass Verification
    print("\n  Running Causal LSTM Forward-Pass Sanity Test...")
    try:
        model = CausalTemporalActionLSTM(input_dim=NUM_FEATURES, hidden_dim=64)
        model.eval()
        dummy_input = torch.randn(2, 30, NUM_FEATURES)
        with torch.no_grad():
            v, o, t = model(dummy_input)
            assert v.shape == (2, 11)
            assert o.shape == (2, 5)
            assert t.shape == (2, 5)
        print("  ✓ Causal Temporal Action LSTM (30, 26) -> (Verb, Object, Target) pass PASSED.")
    except Exception as e:
        print(f"  ❌ Neural network forward pass error: {e}")
        return False

    print("-" * 70)
    print(f"AUDIT SUMMARY: {total_valid} Passed | {total_invalid} Failed (Total: {len(npz_files)})")
    if total_invalid == 0:
        print("✓ ALL 26-D FEATURE FILES COMPLIANT WITH ASTRA-E CONTRACT v1.0")
        print("=" * 70 + "\n")
        return True
    else:
        print("❌ CONTRACT VIOLATIONS FOUND. REVIEW ABOVE LOGS.")
        print("=" * 70 + "\n")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA-E Feature Contract Validator")
    parser.add_argument("--data-dir", default="data/processed/EXP001", help="Path to sequence directory")
    args = parser.parse_args()

    success = validate_feature_directory(args.data_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
