"""PyTorch training and validation runner with autograd, early stopping, and checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ml.training.losses import MultiTaskActionLoss

logger = logging.getLogger("astra.ml.trainer")


class ActionModelTrainer:
    """Trains and validates multi-head temporal models."""

    def __init__(
        self,
        model: nn.Module,
        criterion: MultiTaskActionLoss | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or MultiTaskActionLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=20,
            eta_min=1e-5,
        )

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Execute one training epoch with gradient backward pass and clipping."""
        self.model.train()
        total_loss = 0.0
        batches = 0

        for x, y_v, y_o, y_t in dataloader:
            x = x.to(self.device)
            y_v = y_v.to(self.device)
            y_o = y_o.to(self.device)
            y_t = y_t.to(self.device)

            self.optimizer.zero_grad()

            logits_v, logits_o, logits_t = self.model(x)
            loss, _ = self.criterion(logits_v, logits_o, logits_t, y_v, y_o, y_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        self.scheduler.step()
        avg_loss = total_loss / max(1, batches)
        return {"train_loss": avg_loss}

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate model on validation/test dataloader without gradients."""
        self.model.eval()
        total_loss = 0.0
        correct_v, correct_o, correct_t = 0, 0, 0
        total_samples = 0

        with torch.no_grad():
            for x, y_v, y_o, y_t in dataloader:
                x = x.to(self.device)
                y_v = y_v.to(self.device)
                y_o = y_o.to(self.device)
                y_t = y_t.to(self.device)

                logits_v, logits_o, logits_t = self.model(x)
                loss, _ = self.criterion(logits_v, logits_o, logits_t, y_v, y_o, y_t)

                total_loss += loss.item() * len(x)
                total_samples += len(x)

                pred_v = torch.argmax(logits_v, dim=-1)
                pred_o = torch.argmax(logits_o, dim=-1)
                pred_t = torch.argmax(logits_t, dim=-1)

                correct_v += (pred_v == y_v).sum().item()
                correct_o += (pred_o == y_o).sum().item()
                correct_t += (pred_t == y_t).sum().item()

        n = max(1, total_samples)
        acc_v = correct_v / n
        acc_o = correct_o / n
        acc_t = correct_t / n
        joint_acc = (acc_v + acc_o + acc_t) / 3.0

        return {
            "eval_loss": total_loss / n,
            "acc_verb": acc_v,
            "acc_object": acc_o,
            "acc_target": acc_t,
            "joint_accuracy": joint_acc,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        save_path: str | Path = "models/activity/temporal_model.pt",
    ) -> dict[str, Any]:
        """Full training loop with model checkpointing."""
        best_val_loss = float("inf")
        history: list[dict[str, float]] = []

        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)

        for ep in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            metrics = {**train_metrics, **val_metrics, "epoch": float(ep)}
            history.append(metrics)

            if val_metrics["eval_loss"] < best_val_loss:
                best_val_loss = val_metrics["eval_loss"]
                torch.save(self.model.state_dict(), save_p)

        # Restore best weights
        if save_p.exists():
            self.model.load_state_dict(torch.load(save_p, map_location=self.device))

        return {
            "best_val_loss": best_val_loss,
            "history": history,
            "saved_checkpoint": str(save_p),
        }
