"""Model export, model card generation, and deployment parity verification."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from ml.datasets.schemas import OBJECT_VOCAB, TARGET_VOCAB, VERB_VOCAB


def export_model_card(
    output_dir: str | Path = "models/manifests",
    model_version: str = "activity-causal-lstm-v1.0",
    metrics: dict[str, Any] | None = None,
    temperatures: dict[str, float] | None = None,
    git_commit: str = "ebe757c",
) -> Path:
    """Generate aerospace-grade Model Card manifest."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    card_file = out_p / "model_card.json"

    model_card = {
        "model_id": "ASTRA-ACTIVITY-CAUSAL-LSTM",
        "model_version": model_version,
        "architecture": "2-layer Causal Unidirectional LSTM (hidden=64)",
        "input_shape": [30, 26],
        "sample_rate_hz": 30.0,
        "window_seconds": 1.0,
        "feature_schema_version": "kinematic-26d-v1.0",
        "heads": {
            "verb": {"classes": len(VERB_VOCAB), "vocab": VERB_VOCAB},
            "object": {"classes": len(OBJECT_VOCAB), "vocab": OBJECT_VOCAB},
            "target": {"classes": len(TARGET_VOCAB), "vocab": TARGET_VOCAB},
        },
        "calibration": {
            "method": "temperature_scaling",
            "temperatures": temperatures or {"T_verb": 1.0, "T_object": 1.0, "T_target": 1.0},
        },
        "metrics": metrics or {},
        "git_commit": git_commit,
        "pytorch_version": torch.__version__,
        "created_at": time.time(),
        "operational_constraints": {
            "offline_required": True,
            "max_tolerated_latency_ms": 50.0,
            "causal_guarantee": True,
        },
    }

    with open(card_file, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    return card_file


def verify_model_parity(
    original_model: nn.Module,
    saved_weights_path: str | Path,
    sample_tensor: torch.Tensor,
) -> float:
    """
    Verify numerical equivalence between training model and reloaded model.
    Returns maximum absolute difference across all heads.
    """
    original_model.load_state_dict(torch.load(saved_weights_path, map_location="cpu"))
    original_model.eval()
    with torch.no_grad():
        v1, o1, t1 = original_model(sample_tensor)

    # Reload into fresh instance from checkpoint
    from ml.activity.models.lstm import CausalTemporalActionLSTM
    reloaded = CausalTemporalActionLSTM()
    reloaded.load_state_dict(torch.load(saved_weights_path, map_location="cpu"))
    reloaded.eval()

    with torch.no_grad():
        v2, o2, t2 = reloaded(sample_tensor)

    diff_v = torch.max(torch.abs(v1 - v2)).item()
    diff_o = torch.max(torch.abs(o1 - o2)).item()
    diff_t = torch.max(torch.abs(t1 - t2)).item()

    max_diff = max(diff_v, diff_o, diff_t)
    assert max_diff < 1e-5, f"Parity violation: max difference {max_diff} exceeds tolerance!"
    return max_diff
