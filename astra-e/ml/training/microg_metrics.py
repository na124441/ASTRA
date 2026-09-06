"""Metrics Tracker & Evaluation Engine for MicroG-4M.

Computes accuracy, macro precision, macro recall, macro F1, and weighted F1
with support for severe class imbalance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np


class MicroGMetricsTracker:
    """Computes full classification metrics over continuous evaluation batches."""

    def __init__(self, num_classes: int = 50, taxonomy: Any = None) -> None:
        self.num_classes = num_classes
        self.taxonomy = taxonomy
        self.all_preds: list[int] = []
        self.all_targets: list[int] = []

    def update(self, preds: Sequence[int] | np.ndarray, targets: Sequence[int] | np.ndarray) -> None:
        """Accumulate batch predictions and targets."""
        p_list = [int(p) for p in preds]
        t_list = [int(t) for t in targets]
        self.all_preds.extend(p_list)
        self.all_targets.extend(t_list)

    def reset(self) -> None:
        self.all_preds.clear()
        self.all_targets.clear()

    def compute(self) -> dict[str, Any]:
        """Compute all metrics over accumulated predictions."""
        if not self.all_targets:
            return {
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "total_samples": 0,
                "per_class": {},
            }

        y_true = np.array(self.all_targets, dtype=np.int64)
        y_pred = np.array(self.all_preds, dtype=np.int64)
        n = len(y_true)

        accuracy = float(np.mean(y_true == y_pred))

        per_class: dict[str, dict[str, Any]] = {}
        precisions = []
        recalls = []
        f1s = []
        supports = []

        for c in range(self.num_classes):
            tp = int(np.sum((y_true == c) & (y_pred == c)))
            fp = int(np.sum((y_true != c) & (y_pred == c)))
            fn = int(np.sum((y_true == c) & (y_pred != c)))
            support = int(np.sum(y_true == c))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            class_name = self.taxonomy.get_class_name(c) if self.taxonomy else f"Class_{c}"
            per_class[str(c)] = {
                "name": class_name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }

            if support > 0:
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                supports.append(support)

        macro_precision = float(np.mean(precisions)) if precisions else 0.0
        macro_recall = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

        # Weighted F1
        total_supp = sum(supports)
        weighted_f1 = (
            float(sum(f * s for f, s in zip(f1s, supports)) / total_supp)
            if total_supp > 0
            else 0.0
        )

        return {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "total_samples": n,
            "per_class": per_class,
        }

    def format_final_report(self, metrics: dict[str, Any], best_epoch: int = 1) -> str:
        """
        Produces the exact formatted banner required in Section 11.
        """
        banner = [
            "=" * 50,
            "MicroG-4M Temporal Baseline",
            f"Classes: {self.num_classes}",
            f"Best epoch: {best_epoch}",
            f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%",
            f"Test Macro-F1: {metrics['macro_f1']:.4f}",
            f"Test Weighted-F1: {metrics['weighted_f1']:.4f}",
            "=" * 50,
        ]
        return "\n".join(banner)

    def export_reports(
        self,
        metrics: dict[str, Any],
        output_dir: str | Path,
        best_epoch: int = 1,
    ) -> tuple[Path, Path]:
        """Saves metrics.json and classification_report.json."""
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        metrics_file = out_d / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({
                "best_epoch": best_epoch,
                "test_accuracy": metrics["accuracy"],
                "test_macro_precision": metrics["macro_precision"],
                "test_macro_recall": metrics["macro_recall"],
                "test_macro_f1": metrics["macro_f1"],
                "test_weighted_f1": metrics["weighted_f1"],
                "total_samples": metrics["total_samples"],
            }, f, indent=2)

        report_file = out_d / "classification_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(metrics["per_class"], f, indent=2)

        return metrics_file, report_file
