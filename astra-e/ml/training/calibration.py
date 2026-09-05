"""Temperature Scaling Calibration and Expected Calibration Error (ECE) for multi-head logits."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """
    Learns positive scalar temperatures (T_verb, T_object, T_target) on held-out validation data
    to calibrate softmax confidence scores.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temp_verb = nn.Parameter(torch.ones(1) * 1.2)
        self.temp_object = nn.Parameter(torch.ones(1) * 1.2)
        self.temp_target = nn.Parameter(torch.ones(1) * 1.2)

    def forward(
        self,
        verb_logits: torch.Tensor,
        object_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale logits by learned temperatures."""
        # Clamp temperature to avoid numerical instability
        t_v = torch.clamp(self.temp_verb, min=0.1, max=10.0)
        t_o = torch.clamp(self.temp_object, min=0.1, max=10.0)
        t_t = torch.clamp(self.temp_target, min=0.1, max=10.0)

        return (
            verb_logits / t_v,
            object_logits / t_o,
            target_logits / t_t,
        )

    def fit(
        self,
        val_logits_v: torch.Tensor,
        val_logits_o: torch.Tensor,
        val_logits_t: torch.Tensor,
        y_v: torch.Tensor,
        y_o: torch.Tensor,
        y_t: torch.Tensor,
        max_iter: int = 50,
    ) -> dict[str, float]:
        """Optimize temperature parameters using L-BFGS on validation data."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            sv, so, st = self.forward(val_logits_v, val_logits_o, val_logits_t)
            loss = criterion(sv, y_v) + criterion(so, y_o) + criterion(st, y_t)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        return {
            "T_verb": float(self.temp_verb.item()),
            "T_object": float(self.temp_object.item()),
            "T_target": float(self.temp_target.item()),
        }

    @staticmethod
    def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        """
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(targets)

        ece = torch.zeros(1, device=probs.device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece.item())
