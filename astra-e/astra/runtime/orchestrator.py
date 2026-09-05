"""EdgeRuntimeOrchestrator: Central synchronized edge pipeline coordinating ASTRA-E subsystems."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from astra.activity.confirmation import ActionConfirmationEngine
from astra.activity.features import KinematicFeatureExtractor
from astra.activity.recognizer import ActivityRecognizer
from astra.assistance.manager import AssistanceManager
from astra.assistance.notifier import MultimodalNotifier
from astra.assistance.tts import AudioAssistanceEngine
from astra.configuration.loader import load_procedure_from_yaml
from astra.contracts.activity import ConfirmedAction
from astra.contracts.base import DecisionType, default_uuid
from astra.contracts.perception import SceneObservation
from astra.contracts.procedure import ProcedureDefinition
from astra.contracts.system import EventTopic
from astra.events.bus import EventBus
from astra.interaction.pipeline import InteractionPipeline
from astra.perception.pipeline import PerceptionPipeline
from astra.procedure.engine import ProcedureEngine
from astra.storage.sqlite import SQLiteLedger
from astra.video.buffer import FrameBuffer
from astra.video.camera import Camera, MockCamera
from astra.violation.detector import ViolationDetector

logger = logging.getLogger("astra.runtime.orchestrator")


class EdgeRuntimeOrchestrator:
    """
    Unified ASTRA-E Edge Runtime Daemon.
    Orchestrates the real-time loop:
      Camera -> FrameBuffer -> Perception -> Features -> Causal ML -> Confirmation
      -> Procedure Validation -> Violation Detection -> Assistance/TTS -> SQLite Ledger
      -> WebSockets & MJPEG Streaming
    """

    def __init__(
        self,
        camera: Camera | None = None,
        db_path: str | Path = "data/runs/astra_runtime.db",
        model_weights_path: str | Path | None = "models/activity/temporal_model.pt",
        model_card_path: str | Path | None = "models/manifests/model_card.json",
        tts_enabled: bool = True,
        mock_tts: bool = False,
    ) -> None:
        self.db_path = db_path
        self.camera = camera or MockCamera()
        self.frame_buffer = FrameBuffer(capacity=100)

        # 1. Core Event Bus & Persistence
        self.event_bus = EventBus()
        self.ledger = SQLiteLedger(db_path=self.db_path, event_bus=self.event_bus)

        # 2. Perception & Interaction
        self.perception_pipeline = PerceptionPipeline(event_bus=self.event_bus)
        self.interaction_pipeline = InteractionPipeline(event_bus=self.event_bus)

        # 3. Kinematics & Temporal ML
        self.feature_extractor = KinematicFeatureExtractor(frame_width=640.0, frame_height=480.0)
        self.recognizer = ActivityRecognizer(
            model_path=model_weights_path,
            window_size=16,
            device="cpu",
        )
        if model_card_path and Path(model_card_path).exists():
            import json
            try:
                with open(model_card_path, "r", encoding="utf-8") as f:
                    mcard = json.load(f)
                temps = mcard.get("calibration", {}).get("temperatures", {})
                self.recognizer.set_temperatures(
                    temps.get("verb", 1.0),
                    temps.get("object", 1.0),
                    temps.get("target", 1.0),
                )
            except Exception as e:
                logger.warning(f"Could not load temperatures from model card: {e}")
        self.confirmation_engine = ActionConfirmationEngine(
            min_support_frames=4,
            confirmation_threshold=0.70,
            abstain_threshold=0.50,
        )

        # 4. Deterministic Reasoning & Violation Detection
        self.violation_detector = ViolationDetector(suppression_cooldown_seconds=3.0)
        self.procedure_engine = ProcedureEngine(
            event_bus=self.event_bus,
            violation_detector=self.violation_detector,
        )

        # 5. Multimodal Assistance (TTS + GUI)
        from astra.assistance.tts import MockAudioBackend
        backend = MockAudioBackend() if mock_tts else None
        self.tts = AudioAssistanceEngine(backend=backend, enabled=tts_enabled)
        self.assistance_manager = AssistanceManager(event_bus=self.event_bus)
        self.notifier = MultimodalNotifier(event_bus=self.event_bus, tts_engine=self.tts)

        # Runtime State
        self.run_id: str | None = None
        self.experiment_id: str | None = None
        self.procedure: ProcedureDefinition | None = None
        self.status: str = "IDLE"  # IDLE, RUNNING, PAUSED, COMPLETED

        self._latest_frame: np.ndarray | None = None
        self._latest_scene: SceneObservation | None = None
        self._latest_confirmed_action: ConfirmedAction | None = None
        self._last_fps: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.time()
        self._lock = threading.Lock()

        # Background processing loop
        self._loop_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._telemetry_listeners: list[Callable[[dict[str, Any]], None]] = []

    def register_telemetry_listener(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register listener for real-time WebSocket telemetry dispatch."""
        self._telemetry_listeners.append(callback)

    def start_experiment(
        self,
        experiment_id: str = "EXP001",
        run_id: str | None = None,
        procedure_file: str | Path | None = None,
    ) -> str:
        """
        Initialize and launch an experiment execution run.
        """
        self.experiment_id = experiment_id
        self.run_id = run_id or f"RUN-{default_uuid()[:8].upper()}"

        # Resolve procedure file
        if not procedure_file:
            default_path = Path("experiments") / experiment_id / "procedure.yaml"
            if default_path.exists():
                procedure_file = default_path
            else:
                # Fallback to EXP001
                procedure_file = Path("experiments/EXP001/procedure.yaml")

        self.procedure = load_procedure_from_yaml(procedure_file)
        self.assistance_manager.set_procedure(self.procedure)

        # Reset components
        self.recognizer.reset()
        self.confirmation_engine.reset()
        self.violation_detector.reset()

        # Ensure camera is running
        if not self.camera.is_running:
            self.camera.start()

        # Start engines and ledger
        self.procedure_engine.start(run_id=self.run_id, procedure=self.procedure)
        self.ledger.start_run(
            run_id=self.run_id,
            experiment_id=self.experiment_id,
            procedure_id=self.procedure.id,
            metadata={"name": self.procedure.name, "version": self.procedure.version},
        )
        self.status = "RUNNING"
        logger.info(f"Experiment started: {self.experiment_id} | Run: {self.run_id}")
        return self.run_id

    def pause_experiment(self) -> None:
        """Pause experiment processing."""
        self.status = "PAUSED"
        logger.info(f"Experiment paused: {self.run_id}")

    def resume_experiment(self) -> None:
        """Resume experiment processing."""
        if self.run_id:
            self.status = "RUNNING"
            logger.info(f"Experiment resumed: {self.run_id}")

    def reset_experiment(self) -> None:
        """Reset the orchestrator to IDLE state."""
        if self.run_id and self.status == "RUNNING":
            self.ledger.end_run(self.run_id, status="ABORTED")
        self.status = "IDLE"
        self.run_id = None
        self.procedure = None
        self._latest_confirmed_action = None
        self.recognizer.reset()
        self.confirmation_engine.reset()
        logger.info("Experiment state reset to IDLE.")

    def step_frame(self, input_frame: np.ndarray | None = None) -> dict[str, Any]:
        """
        Execute one complete perception, ML, confirmation, and procedure validation step.
        """
        with self._lock:
            # 1. Video Capture & Buffer
            if input_frame is not None:
                frame = input_frame
                ts = time.time()
                video_frame = self.frame_buffer.push(
                    camera_id=self.camera.camera_id,
                    frame=frame,
                    event_time=ts,
                    correlation_id=self.run_id or "RUN-DEFAULT",
                )
            else:
                if not self.camera.is_running:
                    self.camera.start()
                success, frame, ts = self.camera.read()
                if not success or frame is None:
                    return self.get_telemetry()
                video_frame = self.frame_buffer.push(
                    camera_id=self.camera.camera_id,
                    frame=frame,
                    event_time=ts,
                    correlation_id=self.run_id or "RUN-DEFAULT",
                )

            self._latest_frame = frame
            self._frame_count += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                self._last_fps = self._frame_count / (now - self._fps_timer)
                self._frame_count = 0
                self._fps_timer = now

            if self.status != "RUNNING":
                return self.get_telemetry()

            # 2. Perception & Multi-Object Tracking
            scene_obs = self.perception_pipeline.process_frame(video_frame, frame)
            self._latest_scene = scene_obs

            # 3. Spatial HOI Relationships
            self.interaction_pipeline.process_observation(scene_obs)

            # 4. Leak-free Kinematic Feature Extraction
            feat_vec = self.feature_extractor.extract(scene_obs)

            # 5. Temporal ML Action Recognition (Causal LSTM)
            obs = self.recognizer.process_feature_vector(feat_vec)

            # 6. Temporal Confirmation Layer (EMA, Persistence, Abstention)
            confirmed: ConfirmedAction | None = None
            if obs:
                obs.correlation_id = self.run_id
                confirmed = self.confirmation_engine.process_observation(obs)

            if confirmed:
                confirmed.correlation_id = self.run_id
                self._latest_confirmed_action = confirmed
                self.event_bus.publish(EventTopic.ACTION_CONFIRMED, confirmed)

                # 7. Deterministic Procedure State Machine
                decision = self.procedure_engine.process(confirmed)
                logger.info(f"[DECISION] Action: {confirmed.action} -> {decision.decision.value}")

                # Check completion
                if self.procedure_engine.is_completed:
                    self.status = "COMPLETED"
                    self.ledger.end_run(self.run_id, status="COMPLETED")
                    logger.info(f"Procedure completed successfully: {self.run_id}")

            telemetry = self.get_telemetry()

        # Dispatch telemetry to registered listeners
        for listener in self._telemetry_listeners:
            try:
                listener(telemetry)
            except Exception as e:
                logger.error(f"Error in telemetry listener: {e}")

        return telemetry

    def start_loop(self, target_fps: float = 30.0) -> None:
        """Start background daemon processing thread."""
        if self._loop_thread and self._loop_thread.is_alive():
            return

        self._stop_event.clear()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            args=(target_fps,),
            name="astra-orchestrator-loop",
            daemon=True,
        )
        self._loop_thread.start()
        logger.info("Started ASTRA-E Edge Runtime loop.")

    def stop_loop(self) -> None:
        """Stop background daemon processing thread."""
        self._stop_event.set()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2.0)
        logger.info("Stopped ASTRA-E Edge Runtime loop.")

    def _run_loop(self, target_fps: float) -> None:
        frame_delay = 1.0 / target_fps
        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                self.step_frame()
            except Exception as e:
                logger.error(f"Error in edge runtime loop: {e}", exc_info=True)
            elapsed = time.time() - t0
            sleep_time = max(0.001, frame_delay - elapsed)
            time.sleep(sleep_time)

    def get_telemetry(self) -> dict[str, Any]:
        """Compile comprehensive system telemetry packet for UI dashboards."""
        engine_state = None
        if self.procedure_engine and getattr(self.procedure_engine, "state_manager", None) is not None:
            try:
                engine_state = self.procedure_engine.state
            except RuntimeError:
                engine_state = None

        next_steps = engine_state.next_expected if engine_state else []
        current_step_id = engine_state.current_step if engine_state else None

        # Resolve next step description
        next_step_desc = None
        if next_steps and self.procedure and getattr(self.procedure_engine, "graph", None) is not None:
            s_obj = self.procedure_engine.graph.get_step(next_steps[0])
            if s_obj:
                next_step_desc = s_obj.description

        # Detections for HUD
        detections: list[dict[str, Any]] = []
        if self._latest_scene:
            for d in self._latest_scene.objects:
                detections.append({
                    "id": d.id,
                    "label": d.label,
                    "bbox": [d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max],
                    "confidence": d.confidence,
                })
            for h in self._latest_scene.humans:
                detections.append({
                    "id": h.id,
                    "label": "ASTRONAUT",
                    "bbox": [h.bbox.x_min, h.bbox.y_min, h.bbox.x_max, h.bbox.y_max],
                    "confidence": h.confidence,
                })

        # Calculate progress
        total_steps = len(self.procedure.steps) if self.procedure else 1
        history_steps = len(engine_state.history) if engine_state else 0
        progress_pct = min(100.0, round((history_steps / max(1, total_steps)) * 100.0, 1))

        # Recent assistance
        latest_assist = self.assistance_manager.history[-1].message if self.assistance_manager.history else None
        is_completed = self.procedure_engine.is_completed if (self.procedure_engine and getattr(self.procedure_engine, "state_manager", None) is not None) else False

        return {
            "system": "ASTRA-E",
            "status": self.status,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "fps": round(self._last_fps, 1),
            "timestamp": time.time(),
            "procedure": {
                "id": self.procedure.id if self.procedure else None,
                "name": self.procedure.name if self.procedure else None,
                "current_step": current_step_id,
                "next_expected": next_steps,
                "next_step_description": next_step_desc,
                "progress_percent": progress_pct,
                "completed": is_completed,
            },
            "latest_action": {
                "action": self._latest_confirmed_action.action if self._latest_confirmed_action else None,
                "object_id": self._latest_confirmed_action.object_id if self._latest_confirmed_action else None,
                "target_id": self._latest_confirmed_action.target_id if self._latest_confirmed_action else None,
                "confidence": self._latest_confirmed_action.confidence if self._latest_confirmed_action else None,
            },
            "detections": detections,
            "latest_guidance": latest_assist,
            "violations_count": len(self.violation_detector.history),
        }

    def get_annotated_jpeg(self) -> bytes:
        """Render HUD annotations and encode frame to JPEG for MJPEG browser feed."""
        with self._lock:
            if self._latest_frame is None:
                # Blank placeholder canvas
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, "WAITING FOR VIDEO FEED...", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                img = self._latest_frame.copy()

            # Render HUD Overlays
            h, w = img.shape[:2]
            # Header banner
            cv2.rectangle(img, (0, 0), (w, 40), (20, 20, 20), -1)
            status_color = (0, 255, 0) if self.status == "RUNNING" else (0, 165, 255)
            cv2.putText(img, f"ASTRA-E | STATUS: {self.status}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(img, f"FPS: {self._last_fps:.1f}", (w - 110, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Draw Detections
            if self._latest_scene:
                for obj in self._latest_scene.objects:
                    bx1, by1, bx2, by2 = int(obj.bbox.x_min * w), int(obj.bbox.y_min * h), int(obj.bbox.x_max * w), int(obj.bbox.y_max * h)
                    color = (0, 0, 255) if "RED" in obj.label else (0, 255, 255)
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
                    cv2.putText(img, obj.label, (bx1, max(15, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                for human in self._latest_scene.humans:
                    hx1, hy1, hx2, hy2 = int(human.bbox.x_min * w), int(human.bbox.y_min * h), int(human.bbox.x_max * w), int(human.bbox.y_max * h)
                    cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (255, 100, 0), 2)
                    cv2.putText(img, "HAND", (hx1, max(15, hy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)

            # Footer banner with latest confirmed action
            cv2.rectangle(img, (0, h - 35), (w, h), (20, 20, 20), -1)
            if self._latest_confirmed_action:
                act_str = f"ACTION: {self._latest_confirmed_action.action} {self._latest_confirmed_action.object_id or ''}"
                cv2.putText(img, act_str, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            elif self.status == "RUNNING":
                cv2.putText(img, "OBSERVING ASTRONAUT ACTIVITY...", (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            _, jpeg_bytes = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return jpeg_bytes.tobytes()

    def shutdown(self) -> None:
        """Gracefully terminate edge runtime services."""
        self.stop_loop()
        self.camera.stop()
        self.tts.stop()
        self.ledger.close()
        logger.info("ASTRA-E Edge Runtime Orchestrator terminated cleanly.")
