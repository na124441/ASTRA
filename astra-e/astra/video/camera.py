"""Camera protocols and concrete implementations for ASTRA-E video acquisition."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any
import cv2
import numpy as np


class Camera(ABC):
    """Abstract Base Class defining the unified camera interface."""

    def __init__(self, camera_id: str = "CAM-01") -> None:
        self.camera_id = camera_id
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @abstractmethod
    def start(self) -> None:
        """Initialize and open the video acquisition source."""
        pass

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """
        Capture the next frame.
        Returns:
            (success, frame_ndarray_bgr, ingestion_timestamp_seconds)
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Release camera resources."""
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        pass


class MockCamera(Camera):
    """
    Synthetic camera generating simulated video frames of an astronaut payload experiment.
    Renders the experimental container, colored components, target receptacles, and moving hand.
    Ideal for CI testing, headless environments, and offline simulation.
    """

    def __init__(
        self,
        camera_id: str = "CAM-01",
        width: int = 640,
        height: int = 480,
        fps: float = 30.0,
        total_frames: int = 300,
    ) -> None:
        super().__init__(camera_id=camera_id)
        self._width = width
        self._height = height
        self._fps = fps
        self._total_frames = total_frames
        self._frame_count = 0
        self._start_time = 0.0

        # Scenario keyframe script:
        # Phase 1: hand reaches for RED component in container (frames 0-50)
        # Phase 2: hand grasps and moves RED component to TARGET_A (frames 50-120)
        # Phase 3: hand places & releases RED component in TARGET_A (frames 120-150)
        # Phase 4: hand returns, reaches for YELLOW component (frames 150-200)
        # Phase 5: hand moves YELLOW to TARGET_B (frames 200-260)
        # Phase 6: complete & retreat (frames 260+)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps

    def start(self) -> None:
        self._is_running = True
        self._frame_count = 0
        self._start_time = time.time()

    def stop(self) -> None:
        self._is_running = False

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if not self._is_running or self._frame_count >= self._total_frames:
            return False, None, time.time()

        t = self._frame_count
        capture_time = time.time()

        # Create dark payload rack background (microgravity rack aesthetic)
        frame = np.full((self._height, self._width, 3), 35, dtype=np.uint8)

        # Draw Experiment Container (gray box: x=100..260, y=200..360)
        cv2.rectangle(frame, (100, 200), (260, 360), (90, 90, 90), -1)
        cv2.rectangle(frame, (100, 200), (260, 360), (180, 180, 180), 2)
        cv2.putText(frame, "CONTAINER", (110, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw TARGET_A receptacle (blue/cyan box: x=400..520, y=100..200)
        cv2.rectangle(frame, (400, 100), (520, 200), (70, 70, 40), -1)
        cv2.rectangle(frame, (400, 100), (520, 200), (255, 200, 0), 2)
        cv2.putText(frame, "TARGET_A", (405, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Draw TARGET_B receptacle (green box: x=400..520, y=260..360)
        cv2.rectangle(frame, (400, 260), (520, 360), (40, 70, 40), -1)
        cv2.rectangle(frame, (400, 260), (520, 360), (0, 220, 100), 2)
        cv2.putText(frame, "TARGET_B", (405, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 100), 1)

        # Compute dynamic coordinates based on frame index t
        # Hand default rest position: (560, 400)
        # Red component default position: (150, 250)
        # Yellow component default position: (210, 300)
        red_pos = [150, 250]
        yellow_pos = [210, 300]
        hand_pos = [560, 400]

        if t < 50:
            # Hand moves towards Red component
            alpha = t / 50.0
            hand_pos = [int(560 + alpha * (150 - 560)), int(400 + alpha * (250 - 400))]
        elif t < 120:
            # Hand grasps and moves Red component to Target A (center: 460, 150)
            alpha = (t - 50) / 70.0
            hand_pos = [int(150 + alpha * (460 - 150)), int(250 + alpha * (150 - 250))]
            red_pos = list(hand_pos)  # Red moves with hand!
        elif t < 150:
            # Red is in Target A, hand releases and moves towards Yellow
            red_pos = [460, 150]
            alpha = (t - 120) / 30.0
            hand_pos = [int(460 + alpha * (210 - 460)), int(150 + alpha * (300 - 150))]
        elif t < 220:
            # Hand grasps Yellow and moves to Target B (center: 460, 310)
            alpha = (t - 150) / 70.0
            hand_pos = [int(210 + alpha * (460 - 210)), int(300 + alpha * (310 - 300))]
            red_pos = [460, 150]
            yellow_pos = list(hand_pos)  # Yellow moves with hand!
        else:
            # Hand placed Yellow in Target B and retreats
            red_pos = [460, 150]
            yellow_pos = [460, 310]
            alpha = min(1.0, (t - 220) / 40.0)
            hand_pos = [int(460 + alpha * (560 - 460)), int(310 + alpha * (400 - 310))]

        # Render Red Component (red circle/box: BGR (30, 30, 230))
        cv2.circle(frame, (red_pos[0], red_pos[1]), 18, (20, 20, 225), -1)
        cv2.circle(frame, (red_pos[0], red_pos[1]), 18, (255, 255, 255), 1)

        # Render Yellow Component (yellow circle/box: BGR (30, 230, 230))
        cv2.circle(frame, (yellow_pos[0], yellow_pos[1]), 18, (20, 220, 220), -1)
        cv2.circle(frame, (yellow_pos[0], yellow_pos[1]), 18, (255, 255, 255), 1)

        # Render Astronaut Hand (skin-tone polygon/circle: BGR (140, 175, 220))
        cv2.circle(frame, (hand_pos[0], hand_pos[1]), 22, (140, 175, 220), -1)
        cv2.circle(frame, (hand_pos[0], hand_pos[1]), 22, (255, 255, 255), 2)
        cv2.putText(frame, "HAND", (hand_pos[0] - 20, hand_pos[1] - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Timestamp overlay
        cv2.putText(
            frame,
            f"FRAME {t:04d} | T: {capture_time:.2f}s",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

        self._frame_count += 1
        return True, frame, capture_time


class FileCamera(Camera):
    """Camera implementation reading from video files (.mp4, .avi, etc.)."""

    def __init__(
        self,
        file_path: str,
        camera_id: str = "CAM-FILE-01",
        loop: bool = False,
        realtime_pacing: bool = False,
    ) -> None:
        super().__init__(camera_id=camera_id)
        self.file_path = file_path
        self.loop = loop
        self.realtime_pacing = realtime_pacing
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._width: int = 640
        self._height: int = 480
        self._last_frame_time: float = 0.0

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.file_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {self.file_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._is_running = True
        self._last_frame_time = time.time()

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if not self._is_running or self._cap is None:
            return False, None, time.time()

        if self.realtime_pacing:
            delay = (1.0 / self._fps) - (time.time() - self._last_frame_time)
            if delay > 0:
                time.sleep(delay)
            self._last_frame_time = time.time()

        success, frame = self._cap.read()
        capture_time = time.time()

        if not success and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self._cap.read()

        return success, frame, capture_time

    def stop(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._is_running = False

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps


class OpenCVCamera(Camera):
    """Camera implementation connecting to live USB webcams or RTSP feeds."""

    def __init__(
        self,
        device_index: int | str = 0,
        camera_id: str = "CAM-LIVE-01",
        target_fps: float = 30.0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        super().__init__(camera_id=camera_id)
        self.device_index = device_index
        self._target_fps = target_fps
        self._req_width = width
        self._req_height = height
        self._cap: cv2.VideoCapture | None = None
        self._width: int = width
        self._height: int = height
        self._fps: float = target_fps

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open live video device: {self.device_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._req_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self._req_width
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self._req_height
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or self._target_fps
        self._is_running = True

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if not self._is_running or self._cap is None:
            return False, None, time.time()

        success, frame = self._cap.read()
        capture_time = time.time()
        return success, frame, capture_time

    def stop(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._is_running = False

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fps(self) -> float:
        return self._fps
