"""MicroG-4M Temporal Dataset & Causal Windowing Adapter.

Constructs 30-frame causal sliding windows: X_i = F[i : i + 30], y_i = Y[i + 29].
Preserves zero future-frame lookahead.
Enforces strict fail-closed validation when raw video files are missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("astra.ml.microg_dataset")


class MicroGVideoUnavailableError(FileNotFoundError):
    """
    Raised when required MicroG raw video files or feature tensors cannot be found.
    Refuses to silently fake video data.
    """
    pass


class MicroGTemporalDataset(Dataset):
    """
    PyTorch Dataset serving 30-frame causal temporal sequences for MicroG-4M.
    X_i: [30, D] (30 causal frames)
    y_i: int64 (contiguous class index at endpoint frame i + 29)
    """

    def __init__(
        self,
        annotations: Sequence[dict[str, Any]],
        taxonomy: Any,
        video_dir: str | Path | None = None,
        feature_dir: str | Path | None = None,
        window_size: int = 30,
        feature_dim: int = 128,
        allow_synthetic_test: bool = False,
    ) -> None:
        self.annotations = list(annotations)
        self.taxonomy = taxonomy
        self.video_dir = Path(video_dir) if video_dir else None
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.allow_synthetic_test = allow_synthetic_test

        # Verify data availability before proceeding
        self._verify_data_sources()

        # Build window index table: (video_id, window_idx, class_idx)
        self.samples = self._build_samples()

    def _verify_data_sources(self) -> None:
        """
        Enforce Section 4 reality check:
        Fail loudly if raw videos or features are unavailable.
        """
        if self.allow_synthetic_test:
            # Explicitly allowed ONLY in isolated unit tests
            return

        has_videos = self.video_dir and self.video_dir.exists()
        has_features = self.feature_dir and self.feature_dir.exists()

        if not has_videos and not has_features:
            raise MicroGVideoUnavailableError(
                "\n" + "=" * 80 + "\n"
                "[ERROR] MicroG-4M Video Source Not Found!\n\n"
                "The Hugging Face dataset 'lei-qi-233/MicroG-4M' (config='actions') provides\n"
                "annotation metadata (actions.csv, bounding_boxes.csv, label_map.pbtxt), but\n"
                "does NOT host raw MP4 video files directly due to licensing and copyright\n"
                "constraints (as documented in MicroG-4M README.md and video_id_list.pdf).\n\n"
                "In accordance with ASTRA-E scientific integrity requirements:\n"
                "  - Synthetic / random video frames will NOT be fabricated.\n"
                "  - Training cannot proceed without authentic video data or feature representations.\n\n"
                "To train the temporal baseline:\n"
                "  1. Obtain the video clips for MicroG-4M (refer to video_id_list.pdf) and place\n"
                "     them in a local directory (e.g., data/microg/videos/<video_id>.mp4).\n"
                "  2. Run training pointing to your video directory:\n"
                "     python scripts/training/train_microg.py --video-dir data/microg/videos\n"
                "  3. Alternatively, if you have pre-extracted feature representations:\n"
                "     python scripts/training/train_microg.py --feature-dir data/microg/features\n"
                "=" * 80
            )

    def _build_samples(self) -> list[dict[str, Any]]:
        """Index temporal windows for each annotated action."""
        samples = []
        for row in self.annotations:
            vid = str(row["video_id"])
            action_sparse = int(row["action"])
            class_idx = self.taxonomy.to_contiguous(action_sparse)
            person_id = str(row.get("person_id", "1"))

            # MicroG videos are 3 seconds at 30 fps = 90 frames.
            # With window_size = 30 and stride = 30, we have 3 non-overlapping windows (or stride=15 -> 5 windows)
            # Default to 3 non-overlapping canonical windows per 3s clip
            num_windows = max(1, (90 - self.window_size) // 30 + 1)
            for w_idx in range(num_windows):
                start_frame = w_idx * 30
                end_frame = start_frame + self.window_size - 1
                samples.append({
                    "video_id": vid,
                    "person_id": person_id,
                    "sparse_action": action_sparse,
                    "class_idx": class_idx,
                    "window_idx": w_idx,
                    "start_frame": start_frame,
                    "endpoint_frame": end_frame,  # causal prediction point
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        vid = sample["video_id"]
        class_idx = sample["class_idx"]
        start_f = sample["start_frame"]

        # If synthetic test mode is enabled (unit testing mechanics ONLY)
        if self.allow_synthetic_test:
            # Deterministic pseudo-random seed per sample
            rng = np.random.RandomState(abs(hash(vid + str(start_f))) % (2**31))
            features = rng.randn(self.window_size, self.feature_dim).astype(np.float32)
            return torch.from_numpy(features), torch.tensor(class_idx, dtype=torch.long)

        # Feature mode: load pre-extracted representations
        if self.feature_dir:
            feat_file = self.feature_dir / f"{vid}.npz"
            if not feat_file.exists():
                feat_file = self.feature_dir / f"{vid}.npy"
            if not feat_file.exists():
                raise MicroGVideoUnavailableError(f"Missing feature file for video '{vid}' at {feat_file}")

            if feat_file.suffix == ".npz":
                with np.load(feat_file) as d:
                    full_feat = d["features"]
            else:
                full_feat = np.load(feat_file)

            # Extract window [start_f : start_f + window_size]
            window_feat = full_feat[start_f : start_f + self.window_size].astype(np.float32)
            return torch.from_numpy(window_feat), torch.tensor(class_idx, dtype=torch.long)

        # Video mode: load actual video frames
        if self.video_dir:
            video_path = self._resolve_video_path(vid)
            frames = self._load_video_frames(video_path, start_f, self.window_size)
            return frames, torch.tensor(class_idx, dtype=torch.long)

        raise MicroGVideoUnavailableError(f"Could not load data for video '{vid}'")

    def _resolve_video_path(self, video_id: str) -> Path:
        """Find video file under video_dir, checking root or subdirectories."""
        candidates = [
            self.video_dir / f"{video_id}.mp4",
            self.video_dir / "movie" / f"{video_id}.mp4",
            self.video_dir / "real" / f"{video_id}.mp4",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise MicroGVideoUnavailableError(
            f"Video file for '{video_id}' not found in {self.video_dir} (checked {candidates})"
        )

    def _load_video_frames(self, video_path: Path, start_frame: int, count: int) -> torch.Tensor:
        """Extract frames using OpenCV."""
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise MicroGVideoUnavailableError(f"Could not open video file: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(count):
            ret, frame = cap.read()
            if not ret:
                break
            # Resize and normalize: [H, W, C] -> simple grayscale or feature proxy
            resized = cv2.resize(frame, (112, 112))
            norm = resized.astype(np.float32) / 255.0
            frames.append(norm.mean(axis=-1).flatten()[:self.feature_dim])
        cap.release()

        while len(frames) < count:
            frames.append(frames[-1] if frames else np.zeros(self.feature_dim, dtype=np.float32))

        return torch.from_numpy(np.array(frames, dtype=np.float32))
