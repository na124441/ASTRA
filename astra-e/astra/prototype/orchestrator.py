"""Asynchronous Closed-Loop Prototype Orchestrator (Section 15).

Decoupled Multi-Threaded Real-Time Architecture:
  [Camera Thread]
        │
        ▼
   Frame Queue / Buffer
        │
  ┌─────┴──────────────┐
  ▼                    ▼
[Vision Worker]    [UI / Preview]
  │
  ▼
Event Queue
  │
  ▼
[Reasoning Worker]
  │
  ▼
TTS Queue
  │
  ▼
[Audio Worker] ──► Speaker / Audio Stream
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable
import cv2
import numpy as np

from astra.prototype.alignment_engine import TemporalAlignmentEngine
from astra.prototype.assistant import ConversationalAssistant
from astra.prototype.cloud_client import CloudVisionClient, CloudVoiceClient
from astra.prototype.schemas import (
    AssistanceMode,
    ExperimentRuntimeState,
    ProceduralAlignmentStatus,
    SeverityLevel,
    StructuredObservation,
    VoiceInstruction,
)
from astra.prototype.temporal_buffer import TemporalSlidingWindowBuffer

logger = logging.getLogger("astra.prototype.orchestrator")


class PrototypeOrchestrator:
    """
    Central Asynchronous Pipeline for ASTRA-E Prototype v0.1.
    Connects Camera -> Window Buffer -> Cloud Vision (Colab 1) ->
    ASTRA Reasoner -> Cloud Voice (Colab 2) -> Speaker.
    """

    def __init__(
        self,
        vision_server_url: str = "http://localhost:8001",
        voice_server_url: str = "http://localhost:8002",
        procedure_yaml_path: str | None = None,
        camera_index: int | str = 0,
        guidance_mode: AssistanceMode = AssistanceMode.GUIDANCE,
        muted: bool = False,
    ) -> None:
        self.vision_client = CloudVisionClient(base_url=vision_server_url)
        self.voice_client = CloudVoiceClient(base_url=voice_server_url)

        self.temporal_buffer = TemporalSlidingWindowBuffer(window_size=16, stride=6)
        self.alignment_engine = TemporalAlignmentEngine(
            procedure_yaml_path=procedure_yaml_path,
            mode=guidance_mode,
            cooldown_seconds=5.0,
        )
        self.assistant = ConversationalAssistant(self.alignment_engine)

        self.camera_source = camera_index
        self.muted = muted
        self._simulation_hint_queue: queue.Queue[str] = queue.Queue()

        # Inter-thread Queues (Section 15)
        self._event_queue: queue.Queue[StructuredObservation] = queue.Queue(maxsize=20)
        self._tts_queue: queue.PriorityQueue[tuple[int, float, VoiceInstruction]] = queue.PriorityQueue()

        # Shared thread-safe state for UI
        self._state_lock = threading.Lock()
        self._latest_preview_frame: np.ndarray | None = None
        self._latest_audio_bytes: bytes | None = None
        self._latest_spoken_text: str = ""

        # Lifecycle control
        self._running = False
        self._threads: list[threading.Thread] = []

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Launch all background workers in asynchronous threads."""
        if self._running:
            return

        self._running = True
        logger.info("[ORCHESTRATOR] Launching asynchronous closed-loop pipeline...")

        # 1. Camera Thread
        t_cam = threading.Thread(target=self._camera_worker, name="astra-camera-worker", daemon=True)
        # 2. Vision Worker (Server #1)
        t_vis = threading.Thread(target=self._vision_worker, name="astra-vision-worker", daemon=True)
        # 3. Reasoning Worker (ASTRA State Machine)
        t_reas = threading.Thread(target=self._reasoning_worker, name="astra-reasoning-worker", daemon=True)
        # 4. Audio Worker (Server #2 / Speaker)
        t_aud = threading.Thread(target=self._audio_worker, name="astra-audio-worker", daemon=True)

        self._threads = [t_cam, t_vis, t_reas, t_aud]
        for t in self._threads:
            t.start()

        # Announce initial guidance if in guidance mode
        if self.alignment_engine.mode == AssistanceMode.GUIDANCE and not self.muted:
            init_inst = VoiceInstruction(
                severity=SeverityLevel.INFO,
                message="ASTRA experiment monitoring initiated.",
                guidance="Please begin step 1: Pick up the sample container.",
            )
            self.enqueue_voice_instruction(init_inst)

    def stop(self) -> None:
        """Stop all workers gracefully."""
        self._running = False
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=1.0)
        self._threads.clear()
        logger.info("[ORCHESTRATOR] Closed-loop pipeline terminated.")

    def reset_experiment(self) -> None:
        """Reset procedure state machine to step 1."""
        self.alignment_engine.reset()
        self.temporal_buffer.clear()
        with self._state_lock:
            self._latest_spoken_text = "Experiment reset to initial step."

    def set_guidance_mode(self, mode: AssistanceMode) -> None:
        """Switch between MONITOR and GUIDANCE modes."""
        self.alignment_engine.mode = mode

    def update_urls(self, vision_url: str, voice_url: str) -> None:
        """Dynamically update Colab Server endpoints."""
        if vision_url:
            self.vision_client.base_url = vision_url.rstrip("/")
        if voice_url:
            self.voice_client.base_url = voice_url.rstrip("/")
        logger.info(f"[ORCHESTRATOR] Updated endpoints: Vision={self.vision_client.base_url}, Voice={self.voice_client.base_url}")

    def set_muted(self, muted: bool) -> None:
        """Mute / Unmute voice output."""
        self.muted = muted

    def inject_simulation_event(self, scenario_hint: str) -> None:
        """Inject a scenario trigger (e.g. 'step_1_pick', 'deviation_skipped_step')."""
        self._simulation_hint_queue.put(scenario_hint)

    def enqueue_voice_instruction(self, instruction: VoiceInstruction) -> None:
        """Enqueue a speech alert with priority."""
        priority_map = {
            SeverityLevel.CRITICAL: 1,
            SeverityLevel.DEVIATION: 2,
            SeverityLevel.WARNING: 3,
            SeverityLevel.INFO: 4,
        }
        p_weight = priority_map.get(instruction.severity, 4)
        self._tts_queue.put((p_weight, time.time(), instruction))

    # ================= Workers =================

    def _camera_worker(self) -> None:
        """Continuous frame ingestion loop (Webcam or procedural simulation)."""
        cap = None
        # Attempt opening real camera if integer index provided
        if isinstance(self.camera_source, int):
            try:
                cap = cv2.VideoCapture(self.camera_source)
                if not cap.isOpened():
                    cap = None
            except Exception:
                cap = None

        frame_idx = 0
        while self._running:
            frame: np.ndarray | None = None
            if cap is not None and cap.isOpened():
                ret, raw = cap.read()
                if ret:
                    frame = raw

            if frame is None:
                # Generate dynamic synthetic experiment frame
                frame = self._generate_synthetic_camera_frame(frame_idx)

            frame_idx += 1
            now = time.time()

            # Push to temporal sliding window buffer
            self.temporal_buffer.push(frame, timestamp=now)

            # Update preview frame for UI
            with self._state_lock:
                self._latest_preview_frame = frame.copy()

            time.sleep(0.04)  # ~25 FPS

        if cap is not None:
            cap.release()

    def _vision_worker(self) -> None:
        """Dispatches sliding windows to Colab Server #1."""
        while self._running:
            # Check for simulated hint override first
            hint: str | None = None
            try:
                hint = self._simulation_hint_queue.get_nowait()
            except queue.Empty:
                pass

            if hint is not None or self.temporal_buffer.is_window_ready():
                window_data = self.temporal_buffer.get_window()
                if window_data is not None or hint is not None:
                    if window_data is not None:
                        w_start, w_end, images, _ = window_data
                    else:
                        w_start, w_end = time.time(), time.time() + 3.0
                        images = [np.zeros((240, 320, 3), dtype=np.uint8)]

                    try:
                        observation = self.vision_client.analyze_window(
                            window_start=w_start,
                            window_end=w_end,
                            images=images,
                            simulation_hint=hint,
                        )
                        if not self._event_queue.full():
                            self._event_queue.put(observation)
                    except Exception as e:
                        logger.error(f"[VISION WORKER] Exception: {e}")

            time.sleep(0.08)

    def _reasoning_worker(self) -> None:
        """Processes structured observations through the state machine."""
        while self._running:
            try:
                obs = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            status, voice_inst = self.alignment_engine.process_observation(obs)

            if voice_inst is not None:
                self.enqueue_voice_instruction(voice_inst)

    def _audio_worker(self) -> None:
        """Synthesizes voice instructions into audio bytes and plays speech."""
        while self._running:
            try:
                item = self._tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            _, _, instruction = item
            spoken_text = instruction.to_speech_text()

            with self._state_lock:
                self._latest_spoken_text = spoken_text

            if not self.muted:
                try:
                    audio_bytes = self.voice_client.synthesize(instruction)
                    if audio_bytes:
                        with self._state_lock:
                            self._latest_audio_bytes = audio_bytes
                        # Play locally
                        self.voice_client.play_audio_bytes(audio_bytes)
                except Exception as e:
                    logger.error(f"[AUDIO WORKER] Speech playback error: {e}")

    # ================= State Accessors =================

    def get_runtime_state(self) -> ExperimentRuntimeState:
        """Returns the current state machine snapshot."""
        return self.alignment_engine.get_runtime_state()

    def get_latest_preview_frame(self) -> np.ndarray | None:
        """Returns latest preview frame."""
        with self._state_lock:
            if self._latest_preview_frame is not None:
                return self._latest_preview_frame.copy()
            return None

    def get_latest_spoken_text(self) -> str:
        with self._state_lock:
            return self._latest_spoken_text

    def get_latest_audio_bytes(self) -> bytes | None:
        with self._state_lock:
            return self._latest_audio_bytes

    def answer_operator_query(self, query: str) -> str:
        """Answers astronaut conversational inquiry."""
        ans = self.assistant.answer_query(query)
        # Speak the answer if not muted
        if not self.muted:
            inst = VoiceInstruction(
                severity=SeverityLevel.INFO,
                message=ans.answer,
                guidance="",
            )
            self.enqueue_voice_instruction(inst)
        return ans.answer

    # ================= Synthetic Frame Generator =================

    def _generate_synthetic_camera_frame(self, frame_idx: int) -> np.ndarray:
        """Generates a visual simulation frame depicting the BAS experiment bay."""
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Background: Laboratory workstation
        frame[:] = (40, 42, 48)

        # Draw Experiment Chamber on the right
        cv2.rectangle(frame, (400, 120), (600, 380), (120, 120, 130), -1)
        cv2.rectangle(frame, (400, 120), (600, 380), (180, 180, 190), 3)
        cv2.putText(frame, "CHAMBER BAY", (420, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

        # Draw Sample Container (Red/Amber) on the left
        offset = int(10 * np.sin(frame_idx * 0.1))
        cv2.rectangle(frame, (100, 260 + offset), (220, 360 + offset), (40, 50, 210), -1)
        cv2.rectangle(frame, (100, 260 + offset), (220, 360 + offset), (100, 120, 255), 2)
        cv2.putText(frame, "SAMPLE", (125, 315 + offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw Chamber Door State
        is_open = "S03" in self.alignment_engine.completed_step_ids and "S05" not in self.alignment_engine.completed_step_ids
        door_color = (0, 200, 50) if is_open else (50, 50, 200)
        door_status = "DOOR: OPEN" if is_open else "DOOR: CLOSED"
        cv2.putText(frame, door_status, (430, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.55, door_color, 2)

        # Status overlay on top
        cv2.rectangle(frame, (0, 0), (w, 40), (25, 25, 30), -1)
        status_text = f"ASTRA-E LIVE | STATUS: {self.alignment_engine.alignment_status.value} | STEP: {self.alignment_engine.current_step_index + 1}/5"
        cv2.putText(frame, status_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 180), 2)

        return frame
