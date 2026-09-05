"""Video recording utility for offline archiving of experiment camera feeds."""

from __future__ import annotations

import logging
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger("astra.video.recorder")


class VideoRecorder:
    """
    Records video frames to local filesystem storage.
    Satisfies FR-025 (Local Recording) and FR-026 (Event Association).
    """

    def __init__(
        self,
        output_dir: str | Path = "data/recordings",
        fourcc: str = "mp4v",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        self._current_file: Path | None = None
        self._is_recording = False

    def start_recording(
        self,
        run_id: str,
        camera_id: str,
        width: int,
        height: int,
        fps: float = 30.0,
    ) -> Path:
        """Initialize video file and start writer."""
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self._current_file = run_dir / f"{camera_id}.mp4"
        codec = cv2.VideoWriter_fourcc(*self.fourcc)

        self._writer = cv2.VideoWriter(
            str(self._current_file),
            codec,
            fps,
            (width, height),
        )

        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to initialize VideoWriter at {self._current_file}")

        self._is_recording = True
        logger.info(f"Started recording to {self._current_file}")
        return self._current_file

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame to the recording."""
        if self._is_recording and self._writer is not None:
            self._writer.write(frame)

    def stop_recording(self) -> Path | None:
        """Finalize video file and release writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        self._is_recording = False
        saved_path = self._current_file
        logger.info(f"Finished recording: {saved_path}")
        return saved_path
