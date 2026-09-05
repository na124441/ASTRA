from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from astra.activity.features import KinematicFeatureExtractor


class VideoFeatureExtractorWorker:
    """
    Cloud worker that ingests MP4 video files and extracts normalized 26-D feature sequences
    using the unified production KinematicFeatureExtractor.
    """

    def __init__(
        self,
        frame_width: float = 640.0,
        frame_height: float = 480.0,
        detector_backend: str = "auto",  # 'auto', 'color', 'yolo'
    ) -> None:
        self.width = frame_width
        self.height = frame_height
        self.extractor = KinematicFeatureExtractor(frame_width=frame_width, frame_height=frame_height)
        self.backend = detector_backend

        # Optional color detector fallback
        from astra.perception.detector import ColorExperimentDetector
        from astra.perception.tracker import MultiObjectTracker
        self.color_detector = ColorExperimentDetector()
        self.tracker = MultiObjectTracker()

    def process_video(self, video_path: str | Path, output_file: str | Path) -> dict[str, Any]:
        """Process a single video file into an extracted 26-D NPZ archive."""
        v_path = Path(video_path)
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {v_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.extractor.reset()
        self.tracker.reset()

        features_list: list[np.ndarray] = []
        timestamps_list: list[float] = []
        frame_ids_list: list[int] = []

        frame_idx = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_time = frame_idx / fps
            raw_h, raw_w = frame.shape[:2]
            if raw_w != int(self.width) or raw_h != int(self.height):
                frame = cv2.resize(frame, (int(self.width), int(self.height)))

            # 1. Detect & Track Entities
            raw_detections = self.color_detector.detect(frame)
            active_tracks = self.tracker.update(raw_detections, timestamp=frame_time)

            detections: dict[str, Any] = {"event_time": frame_time}
            for trk in active_tracks:
                name = trk.class_name.upper()
                if name in ("HAND", "HUMAN"):
                    detections["hand"] = {"pos": list(trk.centroid), "conf": float(trk.confidence)}
                elif "RED" in name:
                    detections["red"] = {"pos": list(trk.centroid), "conf": float(trk.confidence)}
                elif "YELLOW" in name:
                    detections["yellow"] = {"pos": list(trk.centroid), "conf": float(trk.confidence)}
                elif "TARGET_A" in name:
                    detections["target_a"] = {"pos": list(trk.centroid)}
                elif "TARGET_B" in name:
                    detections["target_b"] = {"pos": list(trk.centroid)}
                elif "CONTAINER" in name:
                    detections["container"] = {"pos": list(trk.centroid)}

            # 2. Extract 26-D Kinematic Feature Vector via frozen detector contract
            feat_vec = self.extractor.extract(detections)

            features_list.append(feat_vec)
            timestamps_list.append(frame_time)
            frame_ids_list.append(frame_idx)
            frame_idx += 1

        cap.release()
        elapsed = time.time() - t0
        proc_fps = frame_idx / max(0.001, elapsed)

        features_arr = np.array(features_list, dtype=np.float32)
        timestamps_arr = np.array(timestamps_list, dtype=np.float32)
        frame_ids_arr = np.array(frame_ids_list, dtype=np.int32)

        # Save compressed NPZ
        np.savez_compressed(
            out_path,
            features=features_arr,
            timestamps=timestamps_arr,
            frame_ids=frame_ids_arr,
        )

        return {
            "video": v_path.name,
            "frames": frame_idx,
            "shape": features_arr.shape,
            "duration_sec": frame_idx / fps,
            "processing_fps": round(proc_fps, 1),
            "output_file": str(out_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA-E Cloud Feature Extractor")
    parser.add_argument("--video-dir", required=True, help="Directory containing raw MP4 video files")
    parser.add_argument("--output-dir", default="data/cloud/extracted_features", help="Output directory for NPZ features")
    parser.add_argument("--max-videos", type=int, default=None, help="Optional limit on videos to process")
    args = parser.parse_args()

    v_dir = Path(args.video_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(v_dir.glob("*.mp4")) + list(v_dir.glob("*.avi")))
    if args.max_videos:
        videos = videos[: args.max_videos]

    print(f"\nFound {len(videos)} videos in {v_dir}. Commencing 26-D feature extraction...")
    worker = VideoFeatureExtractorWorker()

    for idx, v in enumerate(videos, start=1):
        stem = v.stem
        out_f = out_dir / f"{stem}.npz"
        res = worker.process_video(v, out_f)
        print(f"[{idx}/{len(videos)}] {v.name} -> {res['frames']} frames, shape={res['shape']} ({res['processing_fps']} FPS)")

    print(f"\nFeature extraction complete. Files written to: {out_dir}\n")


if __name__ == "__main__":
    main()
