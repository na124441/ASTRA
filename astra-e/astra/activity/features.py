"""Observable Kinematic Feature Extractor (Zero Ground-Truth Leakage).

Computes 26 continuous physical features representing kinematics, Euclidean distances,
first-order distance derivatives, and tracking confidence.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from astra.contracts.perception import SceneObservation
from astra.interaction.spatial import bbox_centroid, euclidean_distance


class KinematicFeatureExtractor:
    """
    Extracts 26-dimensional observable spatial-temporal feature vectors from SceneObservations.
    Satisfies strict physical causality and zero-leakage constraints:
    Uses only observable coordinates, velocities, distances, and confidence flags.
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
        """Reset internal history for a new run."""
        self._prev_time = None
        self._prev_dist_hand_red = None
        self._prev_dist_hand_yellow = None
        self._prev_dist_red_tgtA = None
        self._prev_dist_yellow_tgtB = None
        self._prev_dist_hand_cnt = None

    def extract_frame_features(self, obs: SceneObservation) -> np.ndarray:
        """
        Extract normalized 26-D feature vector from a single SceneObservation.
        """
        t = obs.event_time
        dt = 1.0 / 30.0
        if self._prev_time is not None and t > self._prev_time:
            dt = max(0.001, t - self._prev_time)
        self._prev_time = t

        # 1. Parse Hand Entity
        hand_conf = 0.0
        hand_pos = list(self._last_hand_pos)
        hand_vx, hand_vy = 0.0, 0.0

        if obs.hands:
            h = obs.hands[0]
            hand_pos = [float(h.position[0]), float(h.position[1])]
            hand_conf = float(h.confidence)
            hand_vx = (hand_pos[0] - self._last_hand_pos[0]) / dt
            hand_vy = (hand_pos[1] - self._last_hand_pos[1]) / dt
            self._last_hand_pos = list(hand_pos)

        # 2. Parse Objects
        red_conf, yellow_conf = 0.0, 0.0
        red_pos = list(self._last_red_pos)
        yellow_pos = list(self._last_yellow_pos)
        red_vx, red_vy = 0.0, 0.0
        yellow_vx, yellow_vy = 0.0, 0.0

        target_a_pos = [self.width * 0.72, self.height * 0.31]
        target_b_pos = [self.width * 0.72, self.height * 0.65]
        container_pos = [self.width * 0.28, self.height * 0.58]

        for obj in obs.objects:
            c = bbox_centroid(obj.bbox)
            if "RED" in obj.type.upper():
                red_conf = float(obj.confidence)
                red_vx = (c[0] - self._last_red_pos[0]) / dt
                red_vy = (c[1] - self._last_red_pos[1]) / dt
                red_pos = list(c)
                self._last_red_pos = list(red_pos)
            elif "YELLOW" in obj.type.upper():
                yellow_conf = float(obj.confidence)
                yellow_vx = (c[0] - self._last_yellow_pos[0]) / dt
                yellow_vy = (c[1] - self._last_yellow_pos[1]) / dt
                yellow_pos = list(c)
                self._last_yellow_pos = list(yellow_pos)
            elif "TARGET_A" in obj.type.upper():
                target_a_pos = list(c)
            elif "TARGET_B" in obj.type.upper():
                target_b_pos = list(c)
            elif "CONTAINER" in obj.type.upper():
                container_pos = list(c)

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
            hand_vx / self.width,              # 2: hand_vx
            hand_vy / self.height,             # 3: hand_vy
            red_pos[0] / self.width,           # 4: red_x
            red_pos[1] / self.height,          # 5: red_y
            red_vx / self.width,               # 6: red_vx
            red_vy / self.height,              # 7: red_vy
            yellow_pos[0] / self.width,        # 8: yellow_x
            yellow_pos[1] / self.height,       # 9: yellow_y
            yellow_vx / self.width,            # 10: yellow_vx
            yellow_vy / self.height,           # 11: yellow_vy
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
