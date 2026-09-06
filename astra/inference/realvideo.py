"""Real-Video Temporal Action Recognition Inference Module for ASTRA-E.

Auxiliary prototype backend using:
  Video -> 16 uniformly sampled frames
        -> MobileNetV3-Small (576-D per frame)
        -> 2-layer unidirectional LSTM (hidden 128)
        -> 6-class action classifier (HMDB51 subset)
        -> Prediction, confidence, top-k ranking, and execution latency.
"""

from __future__ import annotations

import argparse
from collections import deque
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

# Ensure astra package is resolvable whether invoked from root or astra-e subdir
current_dir = Path(__file__).resolve().parent
for candidate in [
    current_dir.parent.parent,
    current_dir.parent.parent / "astra-e",
    current_dir.parent.parent.parent,
    current_dir.parent.parent.parent / "astra-e",
]:
    if (candidate / "astra").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger("astra.inference.realvideo")

# Canonical 6-class order strictly evaluated during Colab training
ACTIONS: list[str] = [
    "brush_hair",
    "drink",
    "eat",
    "pour",
    "clap",
    "wave",
]

SUPPORTED_EXTENSIONS: set[str] = {".avi", ".mp4", ".mov", ".mkv", ".webm"}

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sample_frame_indices(num_frames: int, target_frames: int = 16) -> list[int]:
    """
    Deterministic uniform temporal sampling across [0, num_frames - 1].
    Handles videos with >= target_frames and videos shorter than target_frames.
    Always returns exactly target_frames indices.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")

    if num_frames == 1:
        return [0] * target_frames

    # Uniform linear distribution across [0, num_frames - 1]
    indices = np.linspace(0, num_frames - 1, num=target_frames)
    return [int(round(x)) for x in indices]


def preprocess_frame(bgr_frame: np.ndarray, target_size: tuple[int, int] = (160, 160)) -> np.ndarray:
    """
    Preprocess single OpenCV BGR frame:
      BGR -> RGB -> Resize to (160, 160) -> Scale [0, 1] -> ImageNet Normalization.
    Returns:
      Normalized float32 array with shape (3, H, W)
    """
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    if (rgb.shape[1], rgb.shape[0]) != target_size:
        rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)

    # Scale to [0.0, 1.0]
    img = rgb.astype(np.float32) / 255.0
    # Normalize: (img - mean) / std
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    # Transpose HWC -> CHW
    return np.transpose(img, (2, 0, 1))


class ASTRARealVideoNet(nn.Module):
    """
    Neural architecture reconstructing the trained Colab prototype:
      - Backbone: MobileNetV3-Small with classifier removed (output 576-D)
      - Recurrent Core: 2-layer unidirectional LSTM (input 576, hidden 128)
      - Classifier Head: Linear(128, 6)
    """

    def __init__(
        self,
        num_classes: int = 6,
        feature_dim: int = 576,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Visual Backbone
        self.backbone = models.mobilenet_v3_small(weights=None)
        self.backbone.classifier = nn.Identity()

        # Temporal Model
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # Unidirectional causal recurrence
        )

        # Classification Head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frame features through MobileNetV3-Small.
        Args:
            x: Tensor of shape (B, T, 3, 160, 160)
        Returns:
            Tensor of shape (B, T, 576)
        """
        b, t, c, h, w = x.shape
        x_flat = x.view(b * t, c, h, w)
        feats = self.backbone(x_flat)  # (B * T, 576)
        return feats.view(b, t, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, T=16, C=3, H=160, W=160)
        Returns:
            logits: Tensor of shape (B, 6)
        """
        seq_features = self.extract_visual_features(x)  # (B, T, 576)
        lstm_out, _ = self.lstm(seq_features)           # (B, T, 128)
        final_temporal_state = lstm_out[:, -1, :]       # (B, 128)
        logits = self.classifier(final_temporal_state)   # (B, num_classes)
        return logits


class ASTRARealVideoModel:
    """
    Production-quality inference engine for real-video temporal action recognition.
    Loads trained Colab checkpoint, validates architectures, executes inference,
    computes top-k predictions, and measures execution latency.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "models/realvideo/astra_realvideo_lstm_best.pt",
        device: str | torch.device | None = None,
        num_classes: int = 6,
        feature_dim: int = 576,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Build network architecture
        self.net = ASTRARealVideoNet(
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Load and validate checkpoint
        self._load_and_validate_checkpoint()
        self.net.to(self.device)
        self.net.eval()

        # Streaming buffer for future live/webcam streaming
        self._stream_buffer: deque[np.ndarray] = deque(maxlen=16)

    def _load_and_validate_checkpoint(self) -> None:
        """
        Loads and validates checkpoint structure and tensor shapes.
        Fails clearly if checkpoint does not exist or is incompatible.
        """
        if not self.checkpoint_path.exists():
            # Only search alternative repo locations if default filename was requested
            found = None
            if self.checkpoint_path.name == "astra_realvideo_lstm_best.pt":
                alt_candidates = [
                    Path("models/realvideo/astra_realvideo_lstm_best.pt"),
                    Path("astra-e/models/realvideo/astra_realvideo_lstm_best.pt"),
                    current_dir.parent.parent / "models/realvideo/astra_realvideo_lstm_best.pt",
                    current_dir.parent.parent.parent / "models/realvideo/astra_realvideo_lstm_best.pt",
                ]
                for cand in alt_candidates:
                    if cand.resolve().exists():
                        found = cand.resolve()
                        break

            if found:
                self.checkpoint_path = found
            else:
                raise FileNotFoundError(
                    f"ASTRA-E real-video checkpoint not found at: '{self.checkpoint_path}'.\n"
                    f"Please copy the trained checkpoint from Google Colab into:\n"
                    f"  models/realvideo/astra_realvideo_lstm_best.pt"
                )

        # Load weights on CPU first
        try:
            raw_ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint file '{self.checkpoint_path}': {e}") from e

        # Extract state dict
        if isinstance(raw_ckpt, dict):
            if "model_state_dict" in raw_ckpt:
                state_dict = raw_ckpt["model_state_dict"]
            elif "state_dict" in raw_ckpt:
                state_dict = raw_ckpt["state_dict"]
            else:
                # Check if it looks like a state dict directly
                state_dict = raw_ckpt
        elif isinstance(raw_ckpt, nn.Module):
            state_dict = raw_ckpt.state_dict()
        else:
            raise ValueError(f"Unexpected checkpoint format in '{self.checkpoint_path}': {type(raw_ckpt)}")

        # Validate weights exist
        if not state_dict:
            raise ValueError(f"Checkpoint '{self.checkpoint_path}' contains empty state_dict.")

        # Validate classifier output dimension
        classifier_weight_key = next((k for k in state_dict if "classifier.weight" in k or "fc.weight" in k), None)
        if classifier_weight_key:
            out_classes = state_dict[classifier_weight_key].shape[0]
            if out_classes != self.num_classes:
                raise ValueError(
                    f"ASTRA-E real-video checkpoint incompatible:\n"
                    f"expected {self.num_classes} output classes, found {out_classes} in {classifier_weight_key}"
                )

        # Validate LSTM dimensions
        lstm_ih_key = next((k for k in state_dict if "lstm.weight_ih_l0" in k), None)
        if lstm_ih_key:
            # Shape of weight_ih_l0 is (4 * hidden_dim, feature_dim)
            shape = state_dict[lstm_ih_key].shape
            expected_ih_shape = (4 * self.hidden_dim, self.feature_dim)
            if shape[1] != self.feature_dim:
                raise ValueError(
                    f"ASTRA-E real-video checkpoint incompatible:\n"
                    f"expected feature dimension {self.feature_dim}, found {shape[1]} in {lstm_ih_key}"
                )
            if shape[0] != 4 * self.hidden_dim:
                raise ValueError(
                    f"ASTRA-E real-video checkpoint incompatible:\n"
                    f"expected LSTM hidden size {self.hidden_dim}, found {shape[0] // 4} in {lstm_ih_key}"
                )

        # Harmonize keys if prefix differs (e.g. module. or cnn. vs backbone.)
        clean_state_dict = {}
        for k, v in state_dict.items():
            key = k.removeprefix("module.")
            clean_state_dict[key] = v

        # Load weights into network
        self.net.load_state_dict(clean_state_dict, strict=True)
        logger.info("Successfully loaded and validated checkpoint: %s", self.checkpoint_path)

    def load_video_frames(self, video_path: str | Path, target_frames: int = 16) -> list[np.ndarray]:
        """
        Decodes video file and returns exactly target_frames uniformly sampled raw BGR frames.
        Supports .avi, .mp4, .mov, .mkv, .webm.
        """
        vid_p = Path(video_path)
        if not vid_p.exists():
            raise FileNotFoundError(f"Video file does not exist: {vid_p}")

        if vid_p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(
                "Video extension '%s' not in tested set %s, attempting decode anyway.",
                vid_p.suffix,
                SUPPORTED_EXTENSIONS,
            )

        cap = cv2.VideoCapture(str(vid_p))
        if not cap.isOpened():
            raise ValueError(f"OpenCV failed to open video file: '{vid_p}'. File may be corrupt or missing codec.")

        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames.append(frame)
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"Video file '{vid_p}' contains 0 readable frames or is corrupt.")

        # Uniform temporal sampling
        sample_indices = sample_frame_indices(len(frames), target_frames=target_frames)
        selected_frames = [frames[i] for i in sample_indices]
        return selected_frames

    def predict(
        self,
        video_path: str | Path,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Performs end-to-end temporal action recognition on a video file.
        Args:
            video_path: Path to .avi, .mp4, .mov, .mkv, or .webm video.
            top_k: Number of top predictions to include in result (default: 3).
        Returns:
            Dictionary containing:
                - action: Top predicted action name (str)
                - confidence: Softmax probability in [0.0, 1.0] (float)
                - top_k: List of top-k dictionaries [{"action": str, "confidence": float}, ...]
                - latency_ms: Execution latency of the neural pipeline in milliseconds (float)
        """
        raw_frames = self.load_video_frames(video_path, target_frames=16)
        return self.predict_frames(raw_frames, top_k=top_k)

    def predict_frames(
        self,
        raw_frames: Sequence[np.ndarray],
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Runs action prediction on a sequence of raw BGR frames.
        Args:
            raw_frames: Sequence of BGR image frames (length must be 16).
            top_k: Top-k predictions to return.
        """
        if len(raw_frames) != 16:
            # Resample if not exactly 16
            indices = sample_frame_indices(len(raw_frames), target_frames=16)
            raw_frames = [raw_frames[i] for i in indices]

        # Preprocess each frame
        preprocessed = [preprocess_frame(f, target_size=(160, 160)) for f in raw_frames]
        # Shape: (16, 3, 160, 160)
        temporal_array = np.stack(preprocessed, axis=0)
        # Shape: (1, 16, 3, 160, 160)
        input_tensor = torch.from_numpy(temporal_array).unsqueeze(0).to(self.device)

        return self.predict_tensor(input_tensor, top_k=top_k)

    def predict_tensor(
        self,
        input_tensor: torch.Tensor,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Executes model inference and measures execution latency.
        Args:
            input_tensor: Tensor of shape (1, 16, 3, 160, 160) on self.device
        """
        # Timing setup
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.inference_mode():
            logits = self.net(input_tensor) # (1, 6)
            probs = F.softmax(logits, dim=-1).squeeze(0) # (6,)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        probs_np = probs.cpu().numpy()
        top_k_count = min(top_k, self.num_classes)
        top_indices = np.argsort(probs_np)[::-1][:top_k_count]

        top_predictions = []
        for idx in top_indices:
            action_name = ACTIONS[idx] if idx < len(ACTIONS) else f"action_{idx}"
            conf = float(probs_np[idx])
            top_predictions.append({
                "action": action_name,
                "confidence": round(conf, 4),
            })

        best_action = top_predictions[0]["action"]
        best_conf = top_predictions[0]["confidence"]

        return {
            "action": best_action,
            "confidence": best_conf,
            "top_k": top_predictions,
            "latency_ms": round(latency_ms, 2),
        }

    def process_frame(self, frame: np.ndarray, top_k: int = 3) -> dict[str, Any] | None:
        """
        Streaming hook for live camera / webcam input (Phase 2 readiness).
        Appends frame to internal 16-frame sliding deque and triggers prediction
        once the buffer is filled.
        """
        self._stream_buffer.append(frame)
        if len(self._stream_buffer) == 16:
            return self.predict_frames(list(self._stream_buffer), top_k=top_k)
        return None


def format_cli_banner(
    checkpoint_name: str,
    device_name: str,
    video_name: str,
    result: dict[str, Any],
) -> str:
    """Formats inference result matching Section 13 specification."""
    lines = [
        "=" * 60,
        "ASTRA-E REAL-VIDEO TEMPORAL INFERENCE",
        "=" * 60,
        "",
        "Model:",
        f"  {checkpoint_name}",
        "",
        "Device:",
        f"  {device_name}",
        "",
        "Video:",
        f"  {video_name}",
        "",
        "Frames:",
        "  16",
        "",
        "Prediction:",
        f"  {result['action'].upper()}",
        "",
        "Confidence:",
        f"  {result['confidence'] * 100.0:.2f}%",
        "",
        "Top Predictions:",
    ]
    for rank, item in enumerate(result.get("top_k", []), start=1):
        lines.append(f"  {rank}. {item['action'].upper():<12} {item['confidence'] * 100.0:.2f}%")

    lines.extend([
        "",
        "Inference latency:",
        f"  {result['latency_ms']:.2f} ms",
        "",
        "=" * 60,
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="ASTRA-E Real-Video Temporal Action Recognition Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="models/realvideo/astra_realvideo_lstm_best.pt",
        help="Path to trained real-video temporal checkpoint",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file (.avi, .mp4, .mov, .mkv, .webm)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Inference compute device (default: auto)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3)",
    )

    args = parser.parse_args()

    try:
        model = ASTRARealVideoModel(checkpoint_path=args.model, device=args.device)
        result = model.predict(args.video, top_k=args.top_k)
        banner = format_cli_banner(
            checkpoint_name=Path(args.model).name,
            device_name=str(model.device),
            video_name=Path(args.video).name,
            result=result,
        )
        print(banner)
        return 0
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
