"""Visual and headless demonstration of Phase 2: Video Ingestion, Perception & HOI."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
from astra.events.bus import EventBus
from astra.interaction.pipeline import InteractionPipeline
from astra.perception.pipeline import PerceptionPipeline
from astra.video.buffer import FrameBuffer
from astra.video.camera import Camera, FileCamera, MockCamera, OpenCVCamera


def run_perception_demo(
    camera: Camera,
    max_frames: int = 150,
    headless: bool = False,
    verbose: bool = True,
) -> dict[str, int]:
    """
    Run real-time or benchmark perception pipeline loop.
    Annotates live bounding boxes, tracking labels, hand distances, and interactions.
    """
    event_bus = EventBus()
    buffer = FrameBuffer(capacity=120)
    perception = PerceptionPipeline(event_bus=event_bus)
    hoi = InteractionPipeline(event_bus=event_bus)

    camera.start()

    counts = {
        "frames_processed": 0,
        "observations_generated": 0,
        "interactions_detected": 0,
    }

    if verbose:
        print("\n" + "=" * 68)
        print("║" + "ASTRA-E PERCEPTION & HOI PIPELINE (PHASE 2)".center(66) + "║")
        print("=" * 68)
        print(f"Source Camera: {camera.camera_id} | Resolution: {camera.width}x{camera.height}")
        print(f"Headless Mode: {headless} | Target Frames: {max_frames}\n")

    try:
        for f_idx in range(max_frames):
            success, raw_frame, ts = camera.read()
            if not success or raw_frame is None:
                break

            counts["frames_processed"] += 1

            # 1. Video Ingestion: Store in buffer, obtain contract
            video_frame = buffer.push(
                camera_id=camera.camera_id,
                frame=raw_frame,
                event_time=ts,
                correlation_id="DEMO-RUN-01",
            )

            # 2. Perception: Detection & Tracking -> SceneObservation
            observation = perception.process_frame(video_frame, raw_frame)
            counts["observations_generated"] += 1

            # 3. Interaction: Spatial Relationship & HOI -> InteractionEvents
            interaction_events = hoi.process_observation(observation)
            counts["interactions_detected"] += len(interaction_events)

            # Console telemetry for notable interaction events
            if verbose and interaction_events:
                types_summary = ", ".join(f"{e.interaction_type}({e.object_id or ''})" for e in interaction_events)
                print(f"[FRAME {video_frame.frame_id:03d}] HOI: {types_summary}")

            # Visual overlay rendering
            if not headless:
                display_frame = raw_frame.copy()

                # Draw detected objects
                for obj in observation.objects:
                    x1, y1, x2, y2 = [int(v) for v in obj.bbox]
                    color = (0, 255, 0)
                    if "RED" in obj.type:
                        color = (0, 0, 255)
                    elif "YELLOW" in obj.type:
                        color = (0, 255, 255)
                    elif "TARGET" in obj.type:
                        color = (255, 150, 0)

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{obj.id} ({obj.type})"
                    cv2.putText(display_frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw hands
                for hand in observation.hands:
                    hx, hy = int(hand.position[0]), int(hand.position[1])
                    cv2.circle(display_frame, (hx, hy), 8, (255, 0, 255), -1)
                    cv2.putText(display_frame, hand.id, (hx + 10, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                # Show frame in OpenCV GUI window
                cv2.imshow("ASTRA-E Perception & HOI Stream", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.stop()
        if not headless:
            cv2.destroyAllWindows()

    if verbose:
        print("\n" + "─" * 68)
        print("PERCEPTION PIPELINE SUMMARY:")
        print(f"  Frames Ingested:          {counts['frames_processed']}")
        print(f"  Observations Emitted:     {counts['observations_generated']}")
        print(f"  Interactions Detected:    {counts['interactions_detected']}")
        print("=" * 68 + "\n")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA-E Perception & HOI Layer Demo")
    parser.add_argument("--file", type=str, default=None, help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use live USB webcam")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to process")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window display")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose console printing")
    args = parser.parse_args()

    if args.file:
        cam = FileCamera(file_path=args.file)
    elif args.webcam:
        cam = OpenCVCamera()
    else:
        cam = MockCamera(total_frames=args.frames)

    run_perception_demo(
        camera=cam,
        max_frames=args.frames,
        headless=args.headless,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
