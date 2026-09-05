"""CapturePipeline orchestrating camera acquisition and buffer ingestion."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable
import numpy as np
from astra.contracts.video import VideoFrame
from astra.video.buffer import FrameBuffer
from astra.video.camera import Camera

logger = logging.getLogger("astra.video.capture")


class CapturePipeline:
    """
    Asynchronous or synchronous capture pipeline.
    Reads frames from Camera, pushes them into FrameBuffer, and invokes on_frame callbacks.
    """

    def __init__(
        self,
        camera: Camera,
        buffer: FrameBuffer | None = None,
        correlation_id: str = "RUN-DEFAULT",
    ) -> None:
        self.camera = camera
        self.buffer = buffer or FrameBuffer()
        self.correlation_id = correlation_id
        self._callbacks: list[Callable[[VideoFrame, np.ndarray], None]] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def add_callback(self, callback: Callable[[VideoFrame, np.ndarray], None]) -> None:
        """Register a callback to process every captured frame."""
        self._callbacks.append(callback)

    def read_single_frame(self) -> tuple[VideoFrame, np.ndarray] | None:
        """Read a single frame synchronously."""
        success, frame, capture_time = self.camera.read()
        if not success or frame is None:
            return None

        vf = self.buffer.push(
            camera_id=self.camera.camera_id,
            frame=frame,
            event_time=capture_time,
            correlation_id=self.correlation_id,
        )

        for cb in self._callbacks:
            try:
                cb(vf, frame)
            except Exception as e:
                logger.error(f"Error in capture callback: {e}", exc_info=True)

        return vf, frame

    def start_background(self) -> None:
        """Start non-blocking background capture loop."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self.camera.start()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"CapturePipeline started for camera '{self.camera.camera_id}'.")

    def _run_loop(self) -> None:
        interval = 1.0 / self.camera.fps
        while not self._stop_event.is_set():
            t0 = time.time()
            res = self.read_single_frame()
            if res is None:
                # End of stream or temporary failure
                time.sleep(0.01)
                continue

            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        """Stop background capture and release camera."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.camera.stop()
        logger.info("CapturePipeline stopped.")
