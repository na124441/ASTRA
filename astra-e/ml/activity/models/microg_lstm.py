"""Causal Temporal LSTM Action Recognition Baseline for MicroG-4M.

Architecture:
  Frame Representation [B, T=30, D]
    -> Linear Projection + LayerNorm (D -> hidden_size)
    -> 2-Layer Causal Unidirectional LSTM (hidden_size, dropout=0.2)
    -> Endpoint LayerNorm at time t=29
    -> Linear Classifier (hidden_size -> num_classes)
Output: [B, num_classes] logits.
Strictly causal: zero future-frame leakage.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalMicroGLSTM(nn.Module):
    """
    Causal Temporal LSTM for 50-class MicroG-4M human action recognition.
    Enforces unidirectional computation across the 30-frame temporal window.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 50,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        # Frame projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Strictly causal unidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # NO future lookahead!
        )

        # Endpoint feature normalization
        self.endpoint_norm = nn.LayerNorm(hidden_dim)

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, T=30, D)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        # Shape: (B, T, hidden_dim)
        h = self.input_proj(x)

        # LSTM output: (B, T, hidden_dim)
        lstm_out, _ = self.lstm(h)

        # Extract last hidden state at causal endpoint t = 29
        endpoint_feature = self.endpoint_norm(lstm_out[:, -1, :])

        # Logits: (B, num_classes)
        logits = self.classifier(endpoint_feature)
        return logits
