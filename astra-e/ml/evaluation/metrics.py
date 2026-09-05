"""Comprehensive evaluation metrics: Classification, Event-Level IoU, and Latency Profiling."""

from __future__ import annotations

import time
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> dict[str, float]:
    """Calculate accuracy, macro precision, macro recall, and macro F1."""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    acc = float(np.mean(y_t == y_p))

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = np.sum((y_p == c) & (y_t == c))
        fp = np.sum((y_p == c) & (y_t != c))
        fn = np.sum((y_p != c) & (y_t == c))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
    }


def compute_temporal_iou(
    pred_interval: tuple[float, float],
    true_interval: tuple[float, float],
) -> float:
    """
    Calculate Temporal Intersection-over-Union (tIoU):
      tIoU = |P n G| / |P u G|
    """
    p_start, p_end = pred_interval
    t_start, t_end = true_interval

    inter_start = max(p_start, t_start)
    inter_end = min(p_end, t_end)
    intersection = max(0.0, inter_end - inter_start)

    union_start = min(p_start, t_start)
    union_end = max(p_end, t_end)
    union = max(1e-6, union_end - union_start)

    return float(intersection / union)


def profile_runtime_latency(
    model: nn.Module,
    input_dim: int = 26,
    window_size: int = 30,
    n_warmup: int = 20,
    n_runs: int = 200,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Empirically measure inference latency percentiles (p50, p95, p99) on the host machine.
    """
    model.eval()
    dummy_input = torch.randn(1, window_size, input_dim).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms.sort()
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))

    return {
        "latency_p50_ms": round(p50, 3),
        "latency_p95_ms": round(p95, 3),
        "latency_p99_ms": round(p99, 3),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
    }
