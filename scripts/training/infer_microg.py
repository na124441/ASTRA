"""Inference Entrypoint for MicroG-4M Causal Temporal Baseline.

Executes causal temporal inference on a video file or feature sequence.
Loads best.pt checkpoint and outputs predicted action name, confidence, and class ID.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path (supports running from repo root or astra-e subdir)
current_dir = Path(__file__).resolve().parent
for candidate in [
    current_dir.parent.parent,
    current_dir.parent.parent / "astra-e",
    current_dir.parent.parent.parent,
    current_dir.parent.parent.parent / "astra-e",
]:
    if (candidate / "ml").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import torch
import torch.nn.functional as F

from ml.activity.models.microg_lstm import CausalMicroGLSTM
from ml.datasets.adapters.microg_taxonomy import MicroGTaxonomy


def load_inference_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[CausalMicroGLSTM, dict[str, Any], MicroGTaxonomy]:
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    ckpt = torch.load(ckpt_p, map_location=device)
    config = ckpt.get("model_config", {
        "input_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "num_classes": ckpt.get("num_classes", 50),
        "dropout": 0.2,
        "window_size": 30,
    })

    model = CausalMicroGLSTM(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        dropout=config.get("dropout", 0.0),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    taxonomy = MicroGTaxonomy()
    return model, config, taxonomy


def run_inference_on_window(
    model: CausalMicroGLSTM,
    window_tensor: torch.Tensor,
    taxonomy: MicroGTaxonomy,
    device: torch.device,
) -> tuple[str, float, int]:
    """
    Run causal inference on a [30, D] or [1, 30, D] temporal tensor.
    Returns (predicted_action_name, confidence, class_id).
    """
    if window_tensor.ndim == 2:
        window_tensor = window_tensor.unsqueeze(0)  # [1, 30, D]

    window_tensor = window_tensor.to(device)

    with torch.no_grad():
        logits = model(window_tensor)
        probs = F.softmax(logits, dim=-1)[0]
        class_id = int(torch.argmax(probs).item())
        confidence = float(probs[class_id].item())
        action_name = taxonomy.get_class_name(class_id)

    return action_name, confidence, class_id


def main():
    parser = argparse.ArgumentParser(description="Run ASTRA-E MicroG-4M Inference")
    parser.add_argument("--checkpoint", default="outputs/microg_baseline/best.pt", help="Path to best.pt checkpoint")
    parser.add_argument("--video", default=None, help="Path to MP4 video or NPY feature tensor")
    parser.add_argument("--window-size", type=int, default=30, help="Causal window size")
    parser.add_argument("--synthetic-test", action="store_true", help="Run test inference on random tensor")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, config, taxonomy = load_inference_checkpoint(args.checkpoint, device)
    except Exception as e:
        print(f"[ERROR] Could not load checkpoint: {e}", file=sys.stderr)
        sys.exit(1)

    dim = config.get("input_dim", 128)
    w_size = config.get("window_size", args.window_size)

    if args.synthetic_test or not args.video:
        # Synthetic tensor demonstration
        sample_x = torch.randn(1, w_size, dim)
        action_name, confidence, class_id = run_inference_on_window(model, sample_x, taxonomy, device)
    else:
        v_path = Path(args.video)
        if not v_path.exists():
            print(f"[ERROR] Video file not found: {v_path}", file=sys.stderr)
            sys.exit(1)

        if v_path.suffix in (".npy", ".npz"):
            arr = np.load(v_path)
            if v_path.suffix == ".npz":
                arr = arr["features"]
            if len(arr) < w_size:
                print(f"[ERROR] Video feature length ({len(arr)}) < window size ({w_size})", file=sys.stderr)
                sys.exit(1)
            sample_x = torch.from_numpy(arr[:w_size].astype(np.float32)).unsqueeze(0)
            action_name, confidence, class_id = run_inference_on_window(model, sample_x, taxonomy, device)
        else:
            # Video file with OpenCV
            import cv2
            cap = cv2.VideoCapture(str(v_path))
            frames = []
            while len(frames) < w_size:
                ret, frame = cap.read()
                if not ret:
                    break
                resized = cv2.resize(frame, (112, 112))
                norm = resized.astype(np.float32) / 255.0
                frames.append(norm.mean(axis=-1).flatten()[:dim])
            cap.release()

            if len(frames) < w_size:
                print(f"[ERROR] Extracted {len(frames)} frames, need at least {w_size}", file=sys.stderr)
                sys.exit(1)

            sample_x = torch.from_numpy(np.array(frames, dtype=np.float32)).unsqueeze(0)
            action_name, confidence, class_id = run_inference_on_window(model, sample_x, taxonomy, device)

    # Output formatted as requested in Section 12
    print("Predicted action:")
    print(f"  {action_name}")
    print("Confidence:")
    print(f"  {confidence:.2f}")
    print("Class ID:")
    print(f"  {class_id}")


if __name__ == "__main__":
    main()
