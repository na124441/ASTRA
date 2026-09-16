"""Cloud Client for Colab Server #1 (Vision) and Server #2 (Voice).

Provides robust HTTP/REST communication with both Colab servers.
Includes seamless local fallback execution if Colab servers are not running,
ensuring zero-downtime demonstration on local hardware.
"""

from __future__ import annotations

import asyncio
import io
import logging
import platform
import subprocess
import tempfile
import time
from typing import Any
import httpx
import numpy as np

from astra.prototype.schemas import StructuredObservation, TemporalEvent, VoiceInstruction
from astra.prototype.temporal_buffer import encode_window_to_payload

logger = logging.getLogger("astra.prototype.cloud_client")


class CloudVisionClient:
    """HTTP Client communicating with Colab Server #1 (Video Understanding)."""

    def __init__(self, base_url: str = "http://localhost:8001", timeout_s: float = 4.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._fallback_estimator = None

    def analyze_window(
        self,
        window_start: float,
        window_end: float,
        images: list[np.ndarray],
        simulation_hint: str | None = None,
    ) -> StructuredObservation:
        """Sends temporal window payload to Colab Server #1."""
        url = f"{self.base_url}/api/v1/vision/analyze_window"
        payload = encode_window_to_payload(window_start, window_end, images)
        if simulation_hint:
            payload["simulation_hint"] = simulation_hint

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    events = [
                        TemporalEvent(
                            action=e.get("action", "IDLE"),
                            object=e.get("object"),
                            target=e.get("target"),
                        )
                        for e in data.get("events", [])
                    ]
                    return StructuredObservation(
                        window_start=data.get("window_start", window_start),
                        window_end=data.get("window_end", window_end),
                        observation=data.get("observation", "Observed activity."),
                        events=events,
                        confidence=float(data.get("confidence", 0.90)),
                    )
        except Exception as e:
            logger.debug(f"[VISION CLIENT] Remote server at {self.base_url} unavailable ({e}); using local vision engine.")

        # Local fallback processing
        return self._local_fallback_analyze(window_start, window_end, images, simulation_hint)

    def _local_fallback_analyze(
        self,
        window_start: float,
        window_end: float,
        images: list[np.ndarray],
        simulation_hint: str | None = None,
    ) -> StructuredObservation:
        """Local vision evaluator if Colab server is offline."""
        if self._fallback_estimator is None:
            from apps.prototype.server_vision import TemporalActionEstimator, WindowAnalysisRequest
            self._fallback_estimator = TemporalActionEstimator()

        from apps.prototype.server_vision import WindowAnalysisRequest
        payload = encode_window_to_payload(window_start, window_end, images)
        req = WindowAnalysisRequest(
            window_start=window_start,
            window_end=window_end,
            num_frames=len(images),
            encoded_frames=payload["encoded_frames"],
            simulation_hint=simulation_hint,
        )
        return self._fallback_estimator.analyze(req)


class CloudVoiceClient:
    """HTTP Client communicating with Colab Server #2 (TTS / Voice Model)."""

    def __init__(self, base_url: str = "http://localhost:8002", timeout_s: float = 6.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def synthesize(
        self, instruction: VoiceInstruction | str, voice: str = "en-IN-NeerjaNeural"
    ) -> bytes | None:
        """
        Sends text or VoiceInstruction to Colab Server #2.
        Returns WAV/MP3 audio bytes.
        """
        url = f"{self.base_url}/api/v1/voice/synthesize"
        if isinstance(instruction, VoiceInstruction):
            spoken_text = instruction.to_speech_text()
            payload = {"instruction": instruction.model_dump(), "voice": voice}
        else:
            spoken_text = str(instruction)
            payload = {"text": spoken_text, "voice": voice}

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    return resp.content
        except Exception as e:
            logger.debug(f"[VOICE CLIENT] Remote server at {self.base_url} unavailable ({e}); using local synthesizer.")

        # Local fallback synthesis
        return self._local_fallback_synthesize(spoken_text)

    def _local_fallback_synthesize(self, text: str) -> bytes | None:
        """Local speech synthesis using SAPI or tone generator."""
        if platform.system().lower() == "windows":
            safe_text = text.replace("'", "''").replace('"', '`"')
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                ps_script = f"""
                Add-Type -AssemblyName System.Speech;
                $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
                $synth.SetOutputToWaveFile('{tmp_path}');
                $synth.Speak('{safe_text}');
                $synth.Dispose();
                """
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                    timeout=8.0,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                with open(tmp_path, "rb") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Local SAPI speech fallback failed: {e}")
            finally:
                import os
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        return None

    def play_audio_bytes(self, audio_bytes: bytes) -> None:
        """Plays audio bytes locally through speakers via PowerShell / Windows Media Player."""
        if not audio_bytes:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            if platform.system().lower() == "windows":
                ps_script = f"""
                $player = New-Object System.Media.SoundPlayer('{tmp_path}');
                $player.PlaySync();
                $player.Dispose();
                """
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                    timeout=10.0,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            logger.debug(f"Audio playback error: {e}")
        finally:
            import os
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
