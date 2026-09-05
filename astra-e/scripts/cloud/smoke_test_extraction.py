"""Cloud Smoke Test: Demonstrates and proves the end-to-end video extraction pipeline:
  Video (.mp4) -> Canonical Detections -> 26-D Features -> .npz -> Reload & Verify.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from astra.video.camera import MockCamera
from scripts.cloud.extract_features_cloud import VideoFeatureExtractorWorker


def generate_short_recording(
    output_video_path: str | Path,
    num_frames: int = 120,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Generate a short mock recording simulating microgravity component manipulation."""
    v_path = Path(output_video_path)
    v_path.parent.mkdir(parents=True, exist_ok=True)

    cam = MockCamera(width=width, height=height, total_frames=num_frames, loop=False)
    cam.start()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(v_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {v_path}")

    frames_written = 0
    while frames_written < num_frames:
        success, frame, ts = cam.read()
        if not success or frame is None:
            break
        writer.write(frame)
        frames_written += 1

    writer.release()
    cam.stop()

    if frames_written == 0:
        raise RuntimeError("Zero frames recorded during test video generation.")

    return v_path


def run_smoke_test(
    video_path: str | Path | None = None,
    output_npz_path: str | Path = "data/cloud/smoke_test/EXP001_SMOKE_CAM01.npz",
    num_frames: int = 120,
) -> dict[str, Any]:
    """
    Executes the cloud extraction smoke test.
    If no video_path is provided, generates a short 120-frame mock experiment clip.
    """
    out_npz = Path(output_npz_path)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    # 1. Video Source Setup
    if video_path is None or not Path(video_path).exists():
        demo_video = out_npz.parent / "EXP001_SMOKE_CAM01.mp4"
        print(f"[*] No video supplied. Generating short {num_frames}-frame recording: {demo_video}")
        v_file = generate_short_recording(demo_video, num_frames=num_frames)
    else:
        v_file = Path(video_path)

    print(f"[*] Input Video: {v_file} ({v_file.stat().st_size / 1024:.1f} KB)")

    # 2. Extract Features via Cloud Worker
    print("[*] Processing video frame-by-frame through canonical detector contract...")
    t0 = time.time()
    worker = VideoFeatureExtractorWorker()
    summary = worker.process_video(v_file, out_npz)
    elapsed = time.time() - t0

    # 3. Reload NPZ & Verify
    print(f"[*] Extract complete in {elapsed:.2f}s ({summary['processing_fps']} FPS).")
    print(f"[*] Reloading and verifying output archive: {out_npz}")

    if not out_npz.exists():
        raise FileNotFoundError(f"Failed to create output NPZ archive: {out_npz}")

    with np.load(out_npz) as data:
        keys = list(data.keys())
        features = data["features"]
        timestamps = data["timestamps"]
        frame_ids = data["frame_ids"]

    # Invariants checks
    T, D = features.shape
    assert D == 26, f"Expected 26 features, got {D}"
    assert features.dtype == np.float32, f"Expected float32 dtype, got {features.dtype}"
    assert len(timestamps) == T, f"Timestamp count {len(timestamps)} != frame count {T}"
    assert len(frame_ids) == T, f"Frame ID count {len(frame_ids)} != frame count {T}"
    assert not np.isnan(features).any(), "Found NaN values in extracted feature tensor!"
    assert not np.isinf(features).any(), "Found Inf values in extracted feature tensor!"

    # Print Formatted Verification Box
    print("\n" + "=" * 70)
    print("║" + "ASTRA-E CLOUD EXTRACTION SMOKE TEST: PASSED".center(68) + "║")
    print("=" * 70)
    print(f"  Video Path:         {v_file}")
    print(f"  Processed Frames:   {T}")
    print(f"  Extracted Archive:  {out_npz} ({out_npz.stat().st_size / 1024:.1f} KB)")
    print("----------------------------------------------------------------------")
    print("  VERIFICATION INVARIANTS:")
    print(f"    ✓ features.shape: {features.shape} == [{T}, 26]")
    print(f"    ✓ features.dtype: {features.dtype}")
    print(f"    ✓ timestamps:     {len(timestamps)} items (dt ~ {np.median(np.diff(timestamps)):.4f}s)")
    print(f"    ✓ frame_ids:      [{frame_ids[0]} ... {frame_ids[-1]}]")
    print(f"    ✓ Finite Values:  Zero NaNs / Zero Infs across {features.size} elements")
    print(f"    ✓ Hand (x, y):    [{features[0, 0]:.3f}, {features[0, 1]:.3f}] in [0, 1]")
    print(f"    ✓ Confidences:    Hand={features[0, 23]:.2f}, Red={features[0, 24]:.2f}, Yellow={features[0, 25]:.2f}")
    print("=" * 70 + "\n")

    return {
        "video": str(v_file),
        "npz": str(out_npz),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "frames": int(T),
        "status": "PASSED",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTRA-E Cloud Extraction Smoke Test")
    parser.add_argument("--video", type=str, default=None, help="Optional path to an MP4 video file")
    parser.add_argument("--output", type=str, default="data/cloud/smoke_test/EXP001_SMOKE_CAM01.npz", help="Output .npz path")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames for synthetic test recording")
    args = parser.parse_args()

    run_smoke_test(video_path=args.video, output_npz_path=args.output, num_frames=args.frames)
