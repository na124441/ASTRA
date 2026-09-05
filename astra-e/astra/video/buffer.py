"""High-performance in-memory ring FrameBuffer for zero-copy frame management."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any
import numpy as np
from astra.contracts.base import default_uuid
from astra.contracts.video import VideoFrame


class FrameBuffer:
    """
    Thread-safe circular ring buffer for video frames.
    Stores raw numpy arrays in memory and issues immutable VideoFrame contracts
    carrying zero-copy memory references (e.g. 'memory://frame/1024').
    Prevents large image serialization across IPC/events.
    """

    def __init__(self, capacity: int = 120) -> None:
        self.capacity = max(1, capacity)
        self._lock = threading.RLock()
        self._frames: OrderedDict[int, tuple[VideoFrame, np.ndarray]] = OrderedDict()
        self._next_frame_id = 1

    def push(
        self,
        camera_id: str,
        frame: np.ndarray,
        event_time: float | None = None,
        correlation_id: str = "RUN-DEFAULT",
    ) -> VideoFrame:
        """
        Store a new frame in the buffer and return its VideoFrame contract.
        Evicts oldest frame if capacity is exceeded.
        """
        if event_time is None:
            event_time = time.time()

        height, width = frame.shape[:2]

        with self._lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            frame_ref = f"memory://frame/{frame_id}"

            vf = VideoFrame(
                message_id=f"vf-{default_uuid()[:8]}",
                source="video-ingestion",
                correlation_id=correlation_id,
                frame_id=frame_id,
                camera_id=camera_id,
                width=width,
                height=height,
                format="BGR",
                frame_reference=frame_ref,
                event_time=event_time,
            )

            # Evict oldest if full
            if len(self._frames) >= self.capacity:
                self._frames.popitem(last=False)

            self._frames[frame_id] = (vf, frame)
            return vf

    def get_frame(self, frame_reference: str) -> np.ndarray | None:
        """Retrieve raw frame matrix by memory URI reference."""
        try:
            # Parse memory://frame/{frame_id}
            fid_str = frame_reference.split("/")[-1]
            frame_id = int(fid_str)
            return self.get_frame_by_id(frame_id)
        except (ValueError, IndexError):
            return None

    def get_frame_by_id(self, frame_id: int) -> np.ndarray | None:
        """Retrieve raw frame matrix by integer frame_id."""
        with self._lock:
            item = self._frames.get(frame_id)
            if item is not None:
                return item[1]
            return None

    def get_contract_by_id(self, frame_id: int) -> VideoFrame | None:
        """Retrieve VideoFrame contract by frame_id."""
        with self._lock:
            item = self._frames.get(frame_id)
            if item is not None:
                return item[0]
            return None

    def get_recent_frames(self, count: int = 30) -> list[tuple[VideoFrame, np.ndarray]]:
        """
        Retrieve a temporal window of the most recent frames (ordered oldest -> newest).
        Essential for sliding window activity inference.
        """
        with self._lock:
            items = list(self._frames.values())
            return items[-count:]

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def clear(self) -> None:
        """Purge all stored frames."""
        with self._lock:
            self._frames.clear()
