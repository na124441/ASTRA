"""Object, human, and hand detection engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import cv2
import numpy as np
from astra.perception.types import RawDetection


class BaseDetector(ABC):
    """Abstract interface for pluggable detection models (YOLO, RT-DETR, Color, etc.)."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        """Process image frame and return list of raw detected entities."""
        pass


class ColorExperimentDetector(BaseDetector):
    """
    High-speed, deterministic, fully offline detector using HSV segmentation and contour analysis.
    Identifies experiment components:
    - RED_COMPONENT
    - YELLOW_COMPONENT
    - CONTAINER
    - TARGET_A
    - TARGET_B
    - HAND
    """

    def __init__(self, min_area: float = 80.0) -> None:
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        if frame is None or frame.size == 0:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections: list[RawDetection] = []

        # 1. Detect RED_COMPONENT (wrap-around HSV hue)
        mask_red1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([12, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([168, 120, 100]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        detections.extend(self._extract_contours(mask_red, "RED_COMPONENT", 0.96))

        # 2. Detect YELLOW_COMPONENT
        mask_yellow = cv2.inRange(hsv, np.array([20, 120, 120]), np.array([38, 255, 255]))
        detections.extend(self._extract_contours(mask_yellow, "YELLOW_COMPONENT", 0.95))

        # 3. Detect TARGET_A (blue/cyan)
        mask_target_a = cv2.inRange(hsv, np.array([90, 80, 50]), np.array([130, 255, 255]))
        detections.extend(self._extract_contours(mask_target_a, "TARGET_A", 0.94, min_area=300))

        # 4. Detect TARGET_B (green)
        mask_target_b = cv2.inRange(hsv, np.array([39, 80, 50]), np.array([85, 255, 255]))
        detections.extend(self._extract_contours(mask_target_b, "TARGET_B", 0.94, min_area=300))

        # 5. Detect CONTAINER (medium gray/slate with low saturation)
        mask_container = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([180, 40, 120]))
        detections.extend(self._extract_contours(mask_container, "CONTAINER", 0.92, min_area=800))

        # 6. Detect HAND (light skin-tone: hue 8..25, sat 30..150, val 120..255)
        mask_hand = cv2.inRange(hsv, np.array([8, 25, 140]), np.array([25, 140, 255]))
        hand_dets = self._extract_contours(mask_hand, "HAND", 0.91, min_area=150)
        detections.extend(hand_dets)

        # If a hand is present, infer human presence
        if hand_dets:
            h_det = hand_dets[0]
            # Approximate astronaut human box around the workspace
            detections.append(
                RawDetection(
                    class_name="HUMAN",
                    bbox=[0.0, 0.0, float(frame.shape[1]), float(frame.shape[0])],
                    confidence=0.98,
                    attributes={"role": "astronaut"},
                )
            )

        return detections

    def _extract_contours(
        self,
        mask: np.ndarray,
        class_name: str,
        confidence: float,
        min_area: float | None = None,
    ) -> list[RawDetection]:
        threshold_area = min_area if min_area is not None else self.min_area
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: list[RawDetection] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= threshold_area:
                x, y, w, h = cv2.boundingRect(cnt)
                bbox = [float(x), float(y), float(x + w), float(y + h)]
                results.append(
                    RawDetection(
                        class_name=class_name,
                        bbox=bbox,
                        confidence=confidence,
                        attributes={"area": area},
                    )
                )
        return results
