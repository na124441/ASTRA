"""Causal Multi-Head Temporal LSTM action classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
from ml.datasets.schemas import OBJECT_VOCAB, TARGET_VOCAB, VERB_VOCAB


class CausalTemporalActionLSTM(nn.Module):
    """
    Causal Multi-Head Temporal Action Model.
    Architecture:
      Input (B, T=30, D=26)
        -> Linear Projection + LayerNorm (26 -> 64)
        -> 2-Layer Causal Unidirectional LSTM (hidden_size=64, dropout=0.2)
        -> Last Hidden State at time t=29
        -> 3 Independent Heads: Verb Head, Object Head, Target Head.
    Strictly causal (no future lookahead).
    """

    def __init__(
        self,
        input_dim: int = 26,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_verbs: int = len(VERB_VOCAB),
        num_objects: int = len(OBJECT_VOCAB),
        num_targets: int = len(TARGET_VOCAB),
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # Strictly Causal!
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Multi-Head Prediction Layers
        self.verb_head = nn.Linear(hidden_dim, num_verbs)
        self.object_head = nn.Linear(hidden_dim, num_objects)
        self.target_head = nn.Linear(hidden_dim, num_targets)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, T=30, D=26)
        Returns:
            verb_logits: (B, num_verbs)
            object_logits: (B, num_objects)
            target_logits: (B, num_targets)
        """
        # Linear projection per timestep: (B, T, hidden_dim)
        h_in = self.input_proj(x)

        # Causal LSTM forward pass
        lstm_out, _ = self.lstm(h_in)

        # Extract last hidden state at causal window endpoint t = 29
        endpoint_feature = self.norm(lstm_out[:, -1, :])

        # Multi-head logits
        verb_logits = self.verb_head(endpoint_feature)
        object_logits = self.object_head(endpoint_feature)
        target_logits = self.target_head(endpoint_feature)

        return verb_logits, object_logits, target_logits
