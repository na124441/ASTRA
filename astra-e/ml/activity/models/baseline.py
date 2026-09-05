"""Scientific baseline models (Majority class & Static single-frame MLP)."""

from __future__ import annotations

import torch
import torch.nn as nn
from ml.datasets.schemas import OBJECT_VOCAB, TARGET_VOCAB, VERB_VOCAB


class StaticFrameMLP(nn.Module):
    """
    Static Frame Baseline.
    Consumes only the single instantaneous frame at time t (features[:, -1, :])
    without any temporal history. Used to prove that temporal modeling (LSTM)
    provides statistically significant improvement over static classification.
    """

    def __init__(
        self,
        input_dim: int = 26,
        hidden_dim: int = 64,
        num_verbs: int = len(VERB_VOCAB),
        num_objects: int = len(OBJECT_VOCAB),
        num_targets: int = len(TARGET_VOCAB),
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.verb_head = nn.Linear(hidden_dim, num_verbs)
        self.object_head = nn.Linear(hidden_dim, num_objects)
        self.target_head = nn.Linear(hidden_dim, num_targets)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # If x has shape (B, T, D), take only the last instantaneous frame t
        if x.dim() == 3:
            x_t = x[:, -1, :]
        else:
            x_t = x

        h = self.encoder(x_t)
        return self.verb_head(h), self.object_head(h), self.target_head(h)
