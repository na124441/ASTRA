"""ASTRA-E Standalone Edge Server Entrypoint (BAS)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

# Ensure repository root is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from apps.api.main import app, set_orchestrator
from astra.runtime.orchestrator import EdgeRuntimeOrchestrator
from astra.video.camera import Camera, FileCamera, MockCamera, OpenCVCamera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("astra.edge.server")


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA-E Autonomous Space Task Recognition Edge Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Network interface binding (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--mock", action="store_true", default=True, help="Use synthetic MockCamera (default: True)")
    parser.add_argument("--file", type=str, default=None, help="Replay from a local video file")
    parser.add_argument("--webcam", type=int, default=None, help="Live USB / payload webcam device index")
    parser.add_argument("--db", type=str, default="data/runs/astra_runtime.db", help="Path to SQLite ledger database")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech audio assistance")
    parser.add_argument("--mock-tts", action="store_true", help="Use silent mock TTS for testing")
    parser.add_argument("--fps", type=float, default=30.0, help="Target capture and loop FPS")

    parser.add_argument("--auto-start", action="store_true", default=True, help="Automatically launch EXP001 on server boot")
    parser.add_argument("--no-auto-start", action="store_false", dest="auto_start", help="Wait for manual start from dashboard")

    args = parser.parse_args()

    # Configure Camera Source
    camera: Camera
    if args.file:
        logger.info(f"Using FileCamera source: {args.file}")
        camera = FileCamera(file_path=args.file, fps=args.fps)
    elif args.webcam is not None:
        logger.info(f"Using OpenCVCamera source index: {args.webcam}")
        camera = OpenCVCamera(device_index=args.webcam, fps=args.fps)
    else:
        logger.info("Using synthetic MockCamera source.")
        camera = MockCamera(fps=args.fps)

    # Initialize Unified Orchestrator
    orchestrator = EdgeRuntimeOrchestrator(
        camera=camera,
        db_path=args.db,
        tts_enabled=not args.no_tts,
        mock_tts=args.mock_tts,
    )
    if args.auto_start:
        orchestrator.start_experiment(experiment_id="EXP001")
        logger.info("Auto-started default procedure: EXP001")

    set_orchestrator(orchestrator)

    print("\n" + "=" * 68)
    print("║" + "ASTRA-E AUTONOMOUS EDGE SERVER (BAS)".center(66) + "║")
    print("=" * 68)
    print(f"  Mode:           EDGE OFFLINE-FIRST (Zero Cloud Dependencies)")
    print(f"  Interface:      http://{args.host}:{args.port}")
    print(f"  Dashboard:      http://{args.host}:{args.port}/dashboard")
    print(f"  MJPEG Stream:   http://{args.host}:{args.port}/api/v1/video/stream")
    print(f"  WebSocket:      ws://{args.host}:{args.port}/ws/telemetry")
    print(f"  SQLite Ledger:  {args.db}")
    print(f"  Audio TTS:      {'DISABLED' if args.no_tts else ('MOCK' if args.mock_tts else 'ENABLED')}")
    print("=" * 68 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
