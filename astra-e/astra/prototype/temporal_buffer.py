"""Temporal Sliding-Window FrameBuffer for ASTRA-E Prototype (Section 4)."""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any
import numpy as np

logger = logging.getLogger("astra.prototype.buffer")


class TemporalFrame:
    """Timestamped frame entry stored in buffer."""
    __slots__ = ("frame_id", "timestamp", "image")

    def __init__(self, frame_id: int, timestamp: float, image: np.ndarray) -> None:
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.image = image


class TemporalSlidingWindowBuffer:
    """
    Sliding window buffer for temporal action recognition.
    Avoids sending 60 individual frames per second over the network.
    Instead aggregates frames into overlapping windows (e.g. 16 frames with stride).
    """

    def __init__(
        self,
        window_size: int = 16,
        stride: int = 6,
        max_buffer_size: int = 120,
    ) -> None:
        self.window_size = max(4, window_size)
        self.stride = max(1, stride)
        self.max_buffer_size = max_buffer_size

        self._lock = threading.Lock()
        self._frames: list[TemporalFrame] = []
        self._frame_counter = 0
        self._frames_since_last_window = 0

    def push(self, frame: np.ndarray, timestamp: float | None = None) -> None:
        """Add a frame to the temporal buffer."""
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            self._frame_counter += 1
            tf = TemporalFrame(
                frame_id=self._frame_counter,
                timestamp=timestamp,
                image=frame,
            )
            self._frames.append(tf)
            self._frames_since_last_window += 1

            # Prune buffer to avoid memory leak
            if len(self._frames) > self.max_buffer_size:
                excess = len(self._frames) - self.max_buffer_size
                self._frames = self._frames[excess:]

    def is_window_ready(self) -> bool:
        """Check if enough new frames have accumulated according to stride."""
        with self._lock:
            return (
                len(self._frames) >= self.window_size
                and self._frames_since_last_window >= self.stride
            )

    def get_window(self) -> tuple[float, float, list[np.ndarray], list[int]] | None:
        """
        Extract the latest window of frames with start/end timestamps.
        Resets the stride accumulator.
        """
        with self._lock:
            if len(self._frames) < self.window_size:
                return None

            window_slice = self._frames[-self.window_size :]
            self._frames_since_last_window = 0

            start_time = window_slice[0].timestamp
            end_time = window_slice[-1].timestamp
            images = [f.image for f in window_slice]
            frame_ids = [f.frame_id for f in window_slice]

            return start_time, end_time, images, frame_ids

    def get_latest_frame(self) -> np.ndarray | None:
        """Get the single most recent frame for real-time UI rendering."""
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1].image

    def clear(self) -> None:
        """Reset buffer state."""
        with self._lock:
            self._frames.clear()
            self._frame_counter = 0
            self._frames_since_last_window = 0


def encode_window_to_payload(
    window_start: float,
    window_end: float,
    images: list[np.ndarray],
    subsample: int = 2,
    jpeg_quality: int = 70,
) -> dict[str, Any]:
    """
    Encodes a temporal window of frames into a network-efficient JSON payload.
    Compresses selected frames to JPEG base64 strings.
    """
    import cv2  # Lazy import or fallback

    encoded_frames: list[str] = []
    # Subsample to keep network payload compact (e.g. 8 frames out of 16)
    sampled = images[::subsample] if subsample > 1 else images

    for img in sampled:
        # Resize frame if large to 320x240 for rapid network transmission
        h, w = img.shape[:2]
        if w > 320:
            scale = 320.0 / w
            img = cv2.resize(img, (320, int(h * scale)), interpolation=cv2.INTER_AREA)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        success, buffer = cv2.imencode(".jpg", img, encode_param)
        if success:
            b64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
            encoded_frames.append(b64_str)

    return {
        "window_start": window_start,
        "window_end": window_end,
        "num_frames": len(images),
        "encoded_frames": encoded_frames,
    }
