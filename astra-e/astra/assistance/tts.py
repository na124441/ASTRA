"""Offline Audio & Text-To-Speech Assistance Engine for ASTRA-E (FR-016)."""

from __future__ import annotations

import logging
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Protocol

from astra.contracts.assistance import AssistanceChannel, AssistanceEvent, AssistancePriority

logger = logging.getLogger("astra.assistance.tts")


class AudioBackend(Protocol):
    """Protocol defining text-to-speech backend synthesizers."""
    def synthesize(self, text: str) -> None:
        ...


class MockAudioBackend:
    """Thread-safe mock synthesizer for headless testing and CI."""
    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> None:
        with self._lock:
            self.history.append({"text": text, "timestamp": time.time()})
        logger.debug(f"[MOCK TTS] Synthesized: '{text}'")


class WindowsSAPIBackend:
    """Offline Windows Speech API synthesizer using PowerShell (zero pip dependencies)."""
    def synthesize(self, text: str) -> None:
        # Sanitize text for PowerShell string literal
        safe_text = text.replace("'", "''").replace('"', '`"')
        cmd = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe_text}')",
        ]
        try:
            subprocess.run(cmd, timeout=8.0, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.warning(f"Windows SAPI speech execution failed: {e}")


class PyTTSx3Backend:
    """Offline synthesizer using the cross-platform pyttsx3 engine."""
    def __init__(self) -> None:
        import pyttsx3
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 160)

    def synthesize(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()


def get_default_backend() -> AudioBackend:
    """Selects best available offline audio backend based on platform and packages."""
    try:
        return PyTTSx3Backend()
    except (ImportError, Exception):
        if platform.system().lower() == "windows":
            return WindowsSAPIBackend()
        return MockAudioBackend()


class AudioAssistanceEngine:
    """
    Multimodal Offline Audio Assistance Engine (FR-016).
    Features:
      - Priority queue: High-priority violation alerts jump ahead of routine step guidance.
      - Acoustic fatigue suppression: Cooldown window drops identical repeated voice alerts.
      - Decoupled background synthesis worker thread.
    """

    PRIORITY_WEIGHTS = {
        AssistancePriority.CRITICAL: 1,
        AssistancePriority.HIGH: 2,
        AssistancePriority.MEDIUM: 3,
        AssistancePriority.LOW: 4,
    }

    def __init__(
        self,
        backend: AudioBackend | None = None,
        cooldown_seconds: float = 2.5,
        enabled: bool = True,
    ) -> None:
        self.backend = backend or get_default_backend()
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled

        # Item tuple: (priority_int, timestamp, text)
        self._queue: queue.PriorityQueue[tuple[int, float, str] | None] = queue.PriorityQueue()
        self._last_spoken: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        if self.enabled:
            self._worker = threading.Thread(
                target=self._speech_loop,
                name="astra-tts-worker",
                daemon=True,
            )
            self._worker.start()

    def speak(self, text: str, priority: AssistancePriority = AssistancePriority.LOW) -> bool:
        """Enqueue speech utterance with priority and cooldown validation."""
        if not self.enabled or not text.strip():
            return False

        normalized = text.strip().lower()
        now = time.time()

        with self._lock:
            last_time = self._last_spoken.get(normalized, 0.0)
            if (now - last_time) < self.cooldown_seconds:
                logger.info(f"[TTS COOLDOWN] Suppressed repetitive alert: '{text}'")
                return False
            self._last_spoken[normalized] = now

        p_weight = self.PRIORITY_WEIGHTS.get(priority, 4)
        self._queue.put((p_weight, now, text.strip()))
        return True

    def process_assistance_event(self, event: AssistanceEvent) -> bool:
        """Process typed assistance event and speak if TTS channel is active."""
        if AssistanceChannel.TTS in event.channels:
            return self.speak(event.message, priority=event.priority)
        return False

    def _speech_loop(self) -> None:
        """Background thread worker synthesizing queued utterances in priority order."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                break

            p_weight, ts, text = item
            try:
                self.backend.synthesize(text)
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}", exc_info=True)
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Block until all queued audio utterances have finished playing."""
        if self.enabled and self._worker and self._worker.is_alive():
            self._queue.join()

    def stop(self) -> None:
        """Terminate speech synthesis worker thread."""
        self.flush()
        if self._worker and self._worker.is_alive():
            self._stop_event.set()
            self._queue.put(None)
            self._worker.join(timeout=2.0)
