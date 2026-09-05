"""Observable Kinematic Feature Extractor (Zero Ground-Truth Leakage).

Computes 26 continuous physical features representing kinematics, Euclidean distances,
first-order distance derivatives, and tracking confidence.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import time
import numpy as np
from astra.contracts.perception import SceneObservation, scene_observation_to_detections
from astra.interaction.spatial import bbox_centroid, euclidean_distance


class KinematicFeatureExtractor:
    """
    Extracts 26-dimensional observable spatial-temporal feature vectors from detector predictions.
    Detector-Agnostic: Consumes standardized `detections` dictionary format without dependency
    on whether predictions originate from YOLO, MediaPipe, synthetic simulation, or edge cameras.
    Satisfies strict physical causality and zero ground-truth leakage constraints.
    """

    NUM_FEATURES = 26

    def __init__(
        self,
        frame_width: float = 640.0,
        frame_height: float = 480.0,
    ) -> None:
        self.width = frame_width
        self.height = frame_height
        self.diag = math.hypot(frame_width, frame_height)

        # Previous state for derivative computation
        self._prev_time: float | None = None
        self._prev_dist_hand_red: float | None = None
        self._prev_dist_hand_yellow: float | None = None
        self._prev_dist_red_tgtA: float | None = None
        self._prev_dist_yellow_tgtB: float | None = None
        self._prev_dist_hand_cnt: float | None = None

        # Last known positions for occlusion recovery
        self._last_hand_pos: list[float] = [frame_width * 0.8, frame_height * 0.8]
        self._last_red_pos: list[float] = [frame_width * 0.25, frame_height * 0.5]
        self._last_yellow_pos: list[float] = [frame_width * 0.35, frame_height * 0.6]

    def reset(self) -> None:
        """Reset internal history and cached positions for a new run."""
        self._prev_time = None
        self._prev_dist_hand_red = None
        self._prev_dist_hand_yellow = None
        self._prev_dist_red_tgtA = None
        self._prev_dist_yellow_tgtB = None
        self._prev_dist_hand_cnt = None

        self._last_hand_pos = [self.width * 0.8, self.height * 0.8]
        self._last_red_pos = [self.width * 0.25, self.height * 0.5]
        self._last_yellow_pos = [self.width * 0.35, self.height * 0.6]

    def extract_frame_features(self, obs: SceneObservation | dict[str, Any]) -> np.ndarray:
        """
        Extract normalized 26-D feature vector from a single SceneObservation or detector dictionary.
        """
        return self.extract(obs)

    def extract(self, obs: SceneObservation | dict[str, Any], event_time: float | None = None) -> np.ndarray:
        """
        Extract normalized 26-D feature vector from the standardized agnostic detector contract.

        Expected detector dictionary contract:
            detections = {
                "hand": {"pos": [x, y], "conf": float},
                "red": {"pos": [x, y], "conf": float},
                "yellow": {"pos": [x, y], "conf": float},
                "container": {"pos": [x, y]},
                "target_a": {"pos": [x, y]},
                "target_b": {"pos": [x, y]},
            }
        """
        if isinstance(obs, SceneObservation):
            detections = scene_observation_to_detections(obs)
        elif isinstance(obs, dict):
            detections = obs
        else:
            raise TypeError(f"Expected SceneObservation or detections dict, got {type(obs).__name__}")

        t = event_time if event_time is not None else float(detections.get("event_time", time.time()))

        dt = 1.0 / 30.0
        if self._prev_time is not None and t > self._prev_time:
            dt = max(0.001, t - self._prev_time)
        self._prev_time = t

        # 1. Parse Hand Entity (frozen contract: detections["hand"] -> {"pos": [x, y], "conf": ...})
        hand_data = detections.get("hand")
        hand_conf = 0.0
        hand_pos = list(self._last_hand_pos)
        hand_vx, hand_vy = 0.0, 0.0

        if hand_data:
            pos = hand_data.get("pos") or hand_data.get("position") if isinstance(hand_data, dict) else hand_data
            if pos:
                hand_pos = [float(pos[0]), float(pos[1])]
                hand_conf = float(hand_data.get("conf", hand_data.get("confidence", 1.0))) if isinstance(hand_data, dict) else 1.0
                hand_vx = (hand_pos[0] - self._last_hand_pos[0]) / dt
                hand_vy = (hand_pos[1] - self._last_hand_pos[1]) / dt
                self._last_hand_pos = list(hand_pos)

        # 2. Parse Red Component (frozen contract: detections["red"] -> {"pos": [x, y], "conf": ...})
        red_data = detections.get("red")
        red_conf = 0.0
        red_pos = list(self._last_red_pos)
        red_vx, red_vy = 0.0, 0.0

        if red_data:
            pos = red_data.get("pos") or red_data.get("position") if isinstance(red_data, dict) else red_data
            if pos:
                red_pos = [float(pos[0]), float(pos[1])]
                red_conf = float(red_data.get("conf", red_data.get("confidence", 1.0))) if isinstance(red_data, dict) else 1.0
                red_vx = (red_pos[0] - self._last_red_pos[0]) / dt
                red_vy = (red_pos[1] - self._last_red_pos[1]) / dt
                self._last_red_pos = list(red_pos)

        # 3. Parse Yellow Component (frozen contract: detections["yellow"] -> {"pos": [x, y], "conf": ...})
        yellow_data = detections.get("yellow")
        yellow_conf = 0.0
        yellow_pos = list(self._last_yellow_pos)
        yellow_vx, yellow_vy = 0.0, 0.0

        if yellow_data:
            pos = yellow_data.get("pos") or yellow_data.get("position") if isinstance(yellow_data, dict) else yellow_data
            if pos:
                yellow_pos = [float(pos[0]), float(pos[1])]
                yellow_conf = float(yellow_data.get("conf", yellow_data.get("confidence", 1.0))) if isinstance(yellow_data, dict) else 1.0
                yellow_vx = (yellow_pos[0] - self._last_yellow_pos[0]) / dt
                yellow_vy = (yellow_pos[1] - self._last_yellow_pos[1]) / dt
                self._last_yellow_pos = list(yellow_pos)

        # 4. Parse Static/Receptacle Targets (frozen contract: detections["target_a"], detections["target_b"], detections["container"])
        target_a_pos = [self.width * 0.72, self.height * 0.31]
        target_b_pos = [self.width * 0.72, self.height * 0.65]
        container_pos = [self.width * 0.28, self.height * 0.58]

        tgt_a_data = detections.get("target_a")
        if tgt_a_data:
            pos = tgt_a_data.get("pos") or tgt_a_data.get("position") if isinstance(tgt_a_data, dict) else tgt_a_data
            if pos:
                target_a_pos = [float(pos[0]), float(pos[1])]

        tgt_b_data = detections.get("target_b")
        if tgt_b_data:
            pos = tgt_b_data.get("pos") or tgt_b_data.get("position") if isinstance(tgt_b_data, dict) else tgt_b_data
            if pos:
                target_b_pos = [float(pos[0]), float(pos[1])]

        cnt_data = detections.get("container")
        if cnt_data:
            pos = cnt_data.get("pos") or cnt_data.get("position") if isinstance(cnt_data, dict) else cnt_data
            if pos:
                container_pos = [float(pos[0]), float(pos[1])]

        # Backward compatibility for raw dicts containing objects/hands lists
        if "hands" in detections and not hand_data and detections["hands"]:
            h = detections["hands"][0]
            pos = h.get("pos") or h.get("position") if isinstance(h, dict) else getattr(h, "position", None)
            if pos:
                hand_pos = [float(pos[0]), float(pos[1])]
                hand_conf = float(h.get("conf", h.get("confidence", 1.0))) if isinstance(h, dict) else float(getattr(h, "confidence", 1.0))
                hand_vx = (hand_pos[0] - self._last_hand_pos[0]) / dt
                hand_vy = (hand_pos[1] - self._last_hand_pos[1]) / dt
                self._last_hand_pos = list(hand_pos)

        if "objects" in detections and isinstance(detections["objects"], list):
            for obj in detections["objects"]:
                obj_type = obj.get("type", "").upper() if isinstance(obj, dict) else getattr(obj, "type", "").upper()
                pos = obj.get("pos") if isinstance(obj, dict) else None
                if pos is None:
                    bbox = obj.get("bbox") if isinstance(obj, dict) else getattr(obj, "bbox", None)
                    pos = bbox_centroid(bbox) if bbox else None
                conf = float(obj.get("confidence", 1.0)) if isinstance(obj, dict) else float(getattr(obj, "confidence", 1.0))
                if pos:
                    if "RED" in obj_type and not red_data:
                        red_conf = conf
                        red_vx = (pos[0] - self._last_red_pos[0]) / dt
                        red_vy = (pos[1] - self._last_red_pos[1]) / dt
                        red_pos = [float(pos[0]), float(pos[1])]
                        self._last_red_pos = list(red_pos)
                    elif "YELLOW" in obj_type and not yellow_data:
                        yellow_conf = conf
                        yellow_vx = (pos[0] - self._last_yellow_pos[0]) / dt
                        yellow_vy = (pos[1] - self._last_yellow_pos[1]) / dt
                        yellow_pos = [float(pos[0]), float(pos[1])]
                        self._last_yellow_pos = list(yellow_pos)
                    elif "TARGET_A" in obj_type and not tgt_a_data:
                        target_a_pos = [float(pos[0]), float(pos[1])]
                    elif "TARGET_B" in obj_type and not tgt_b_data:
                        target_b_pos = [float(pos[0]), float(pos[1])]
                    elif "CONTAINER" in obj_type and not cnt_data:
                        container_pos = [float(pos[0]), float(pos[1])]

        # 3. Calculate Normalized Euclidean Distances
        d_h_r = euclidean_distance(hand_pos, red_pos) / self.diag
        d_h_y = euclidean_distance(hand_pos, yellow_pos) / self.diag
        d_r_tgtA = euclidean_distance(red_pos, target_a_pos) / self.diag
        d_y_tgtB = euclidean_distance(yellow_pos, target_b_pos) / self.diag
        d_h_cnt = euclidean_distance(hand_pos, container_pos) / self.diag

        # 4. First-Order Distance Derivatives (approaching < 0, retreating > 0)
        def calc_derivative(curr: float, prev: float | None) -> float:
            if prev is None:
                return 0.0
            # Clamp derivative to [-3.0, 3.0]
            val = (curr - prev) / dt
            return max(-3.0, min(3.0, val))

        d_dot_h_r = calc_derivative(d_h_r, self._prev_dist_hand_red)
        d_dot_h_y = calc_derivative(d_h_y, self._prev_dist_hand_yellow)
        d_dot_r_tgtA = calc_derivative(d_r_tgtA, self._prev_dist_red_tgtA)
        d_dot_y_tgtB = calc_derivative(d_y_tgtB, self._prev_dist_yellow_tgtB)

        self._prev_dist_hand_red = d_h_r
        self._prev_dist_hand_yellow = d_h_y
        self._prev_dist_red_tgtA = d_r_tgtA
        self._prev_dist_yellow_tgtB = d_y_tgtB

        # 5. Co-Movement Relative Velocity Norm ||v_hand - v_object||
        rel_v_red = math.hypot(hand_vx - red_vx, hand_vy - red_vy) / (self.diag * 2.0)
        rel_v_yellow = math.hypot(hand_vx - yellow_vx, hand_vy - yellow_vy) / (self.diag * 2.0)

        # Assemble 26 features vector
        features = np.array([
            hand_pos[0] / self.width,          # 0: hand_x
            hand_pos[1] / self.height,         # 1: hand_y
            hand_vx / self.width,              # 2: hand_vx (normalized velocity, 1/sec)
            hand_vy / self.height,             # 3: hand_vy (normalized velocity, 1/sec)
            red_pos[0] / self.width,           # 4: red_x
            red_pos[1] / self.height,          # 5: red_y
            red_vx / self.width,               # 6: red_vx (normalized velocity, 1/sec)
            red_vy / self.height,              # 7: red_vy (normalized velocity, 1/sec)
            yellow_pos[0] / self.width,        # 8: yellow_x
            yellow_pos[1] / self.height,       # 9: yellow_y
            yellow_vx / self.width,            # 10: yellow_vx (normalized velocity, 1/sec)
            yellow_vy / self.height,           # 11: yellow_vy (normalized velocity, 1/sec)
            d_h_r,                             # 12: dist_hand_red
            d_h_y,                             # 13: dist_hand_yellow
            d_r_tgtA,                          # 14: dist_red_tgtA
            d_y_tgtB,                          # 15: dist_yellow_tgtB
            d_h_cnt,                           # 16: dist_hand_container
            d_dot_h_r,                         # 17: d_dot_hand_red
            d_dot_h_y,                         # 18: d_dot_hand_yellow
            d_dot_r_tgtA,                      # 19: d_dot_red_tgtA
            d_dot_y_tgtB,                      # 20: d_dot_yellow_tgtB
            min(1.0, rel_v_red),               # 21: co_movement_red
            min(1.0, rel_v_yellow),            # 22: co_movement_yellow
            hand_conf,                         # 23: conf_hand
            red_conf,                          # 24: conf_red
            yellow_conf,                       # 25: conf_yellow
        ], dtype=np.float32)

        return features

    def extract_window(self, observations: Sequence[SceneObservation]) -> np.ndarray:
        """
        Extract a temporal sequence of features of shape (T, 26).
        """
        window = []
        for obs in observations:
            window.append(self.extract_frame_features(obs))
        return np.array(window, dtype=np.float32)
