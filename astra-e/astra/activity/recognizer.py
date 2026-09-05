"""ActivityRecognizer executing local causal temporal inference on sliding feature windows."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from astra.contracts.activity import ActionObservation, TemporalWindow
from astra.contracts.base import default_uuid
from ml.activity.models.lstm import CausalTemporalActionLSTM
from ml.datasets.schemas import IDX_TO_OBJECT, IDX_TO_TARGET, IDX_TO_VERB
from ml.training.calibration import TemperatureScaler


class ActivityRecognizer:
    """
    Local runtime neural inference wrapper.
    Processes rolling window of 30 frames (1 second at 30 fps) and outputs ActionObservation.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        window_size: int = 30,
        device: str = "cpu",
    ) -> None:
        self.window_size = window_size
        self.device = device
        self.model = CausalTemporalActionLSTM(input_dim=26, hidden_dim=64).to(device)
        self.calibrator = TemperatureScaler().to(device)
        self._window_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=window_size)
        self._timestamps_buffer: collections.deque[float] = collections.deque(maxlen=window_size)

        if model_path is not None and Path(model_path).exists():
            self.load_weights(model_path)

        self.model.eval()

    def load_weights(self, path: str | Path) -> None:
        """Load trained model state dict."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def set_temperatures(self, t_v: float, t_o: float, t_t: float) -> None:
        """Set learned calibration temperatures."""
        with torch.no_grad():
            self.calibrator.temp_verb.copy_(torch.tensor([t_v]))
            self.calibrator.temp_object.copy_(torch.tensor([t_o]))
            self.calibrator.temp_target.copy_(torch.tensor([t_t]))

    def reset(self) -> None:
        """Reset rolling buffer."""
        self._window_buffer.clear()
        self._timestamps_buffer.clear()

    def push_frame_features(
        self,
        features: np.ndarray,
        timestamp: float,
        correlation_id: str = "RUN-DEFAULT",
    ) -> ActionObservation | None:
        """
        Append single-frame 26-D feature vector and evaluate temporal window.
        Returns ActionObservation once window is filled (T >= 30).
        """
        self._window_buffer.append(features)
        self._timestamps_buffer.append(timestamp)

        if len(self._window_buffer) < self.window_size:
            # Insufficient temporal context
            return None

        # Assemble causal window tensor of shape (1, 30, 26)
        window_arr = np.array(self._window_buffer, dtype=np.float32)
        x_tensor = torch.from_numpy(window_arr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_v, raw_o, raw_t = self.model(x_tensor)
            scaled_v, scaled_o, scaled_t = self.calibrator(raw_v, raw_o, raw_t)

            probs_v = F.softmax(scaled_v, dim=-1)[0]
            probs_o = F.softmax(scaled_o, dim=-1)[0]
            probs_t = F.softmax(scaled_t, dim=-1)[0]

            conf_v, pred_v = torch.max(probs_v, dim=-1)
            conf_o, pred_o = torch.max(probs_o, dim=-1)
            conf_t, pred_t = torch.max(probs_t, dim=-1)

        verb = IDX_TO_VERB.get(int(pred_v.item()), "UNKNOWN")
        obj = IDX_TO_OBJECT.get(int(pred_o.item()), "NONE")
        tgt = IDX_TO_TARGET.get(int(pred_t.item()), "NONE")

        # Joint calibrated confidence score
        joint_conf = float((conf_v.item() * 0.5) + (conf_o.item() * 0.25) + (conf_t.item() * 0.25))

        t_start = self._timestamps_buffer[0]
        t_end = self._timestamps_buffer[-1]

        return ActionObservation(
            message_id=f"act-obs-{default_uuid()[:8]}",
            source="activity-recognizer",
            correlation_id=correlation_id,
            action=verb,
            object_id=obj if obj != "NONE" else None,
            target_id=tgt if tgt != "NONE" else None,
            confidence=round(joint_conf, 4),
            event_time=t_end,
            temporal_window=TemporalWindow(start=t_start, end=t_end),
            metadata={
                "verb_confidence": round(float(conf_v.item()), 4),
                "object_confidence": round(float(conf_o.item()), 4),
                "target_confidence": round(float(conf_t.item()), 4),
            },
        )
