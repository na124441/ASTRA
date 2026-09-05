"""Multi-task loss function for Multi-Head Action classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskActionLoss(nn.Module):
    """
    Multi-Task loss combining:
      L = lambda_v * L_verb + lambda_o * L_object + lambda_t * L_target
    """

    def __init__(
        self,
        weight_verb: float = 1.0,
        weight_object: float = 0.8,
        weight_target: float = 0.8,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.w_v = weight_verb
        self.w_o = weight_object
        self.w_t = weight_target

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        verb_logits: torch.Tensor,
        object_logits: torch.Tensor,
        target_logits: torch.Tensor,
        verb_targets: torch.Tensor,
        object_targets: torch.Tensor,
        target_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_v = self.loss_fn(verb_logits, verb_targets)
        loss_o = self.loss_fn(object_logits, object_targets)
        loss_t = self.loss_fn(target_logits, target_targets)

        total_loss = (self.w_v * loss_v) + (self.w_o * loss_o) + (self.w_t * loss_t)

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_verb": float(loss_v.item()),
            "loss_object": float(loss_o.item()),
            "loss_target": float(loss_t.item()),
        }
        return total_loss, metrics
