"""Colab Server #2: Cloud Text-To-Speech / Natural Voice Guidance Microservice.

Runs on Colab (Server B) or locally.
Receives structured VoiceInstruction or alert text from the ASTRA Reasoner,
synthesizes natural speech using neural TTS (edge-tts / SAPI / pyttsx3),
and streams back the synthesized WAV/MP3 audio.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import platform
import subprocess
import tempfile
import time
from typing import Any
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from astra.prototype.schemas import SeverityLevel, VoiceInstruction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("astra.server_voice")

app = FastAPI(
    title="ASTRA-E Voice & TTS Server (Server #2)",
    description="Natural Voice Assistance Service for Bharatiya Antariksh Station (BAS)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpeechSynthesisRequest(BaseModel):
    """Synthesis request containing text or structured VoiceInstruction."""
    text: str | None = None
    instruction: VoiceInstruction | None = None
    voice: str = Field(default="en-IN-NeerjaNeural", description="Neural voice identifier")
    rate: str = Field(default="+0%", description="Speech speed adjustment")
    volume: str = Field(default="+0%", description="Speech volume adjustment")


class NeuralVoiceSynthesizer:
    """
    Multi-engine Speech Synthesizer:
    1. edge-tts: high-fidelity neural speech (supports Indian English, US English, etc.)
    2. pyttsx3 / Windows SAPI: offline local fallback
    3. Silent mock WAV fallback if offline
    """

    def __init__(self, default_voice: str = "en-IN-NeerjaNeural") -> None:
        self.default_voice = default_voice

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Synthesize text into WAV or MP3 audio bytes."""
        text = text.strip()
        if not text:
            return self._generate_silence_wav()

        target_voice = voice or self.default_voice

        # 1. Try edge-tts if installed and online
        try:
            import edge_tts
            communicate = edge_tts.Communicate(text, target_voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            if audio_data:
                logger.info(f"[VOICE SERVER] edge-tts synthesized {len(audio_data)} bytes: '{text}'")
                return audio_data
        except ImportError:
            logger.debug("edge-tts not installed; trying local fallback.")
        except Exception as e:
            logger.warning(f"edge-tts synthesis failed: {e}; falling back.")

        # 2. Try Windows SAPI (offline on Windows)
        if platform.system().lower() == "windows":
            try:
                wav_bytes = self._synthesize_windows_sapi(text)
                if wav_bytes:
                    logger.info(f"[VOICE SERVER] Windows SAPI synthesized {len(wav_bytes)} bytes: '{text}'")
                    return wav_bytes
            except Exception as e:
                logger.warning(f"Windows SAPI synthesis failed: {e}")

        # 3. Fallback: generate a clean valid WAV tone / speech stub
        return self._generate_tone_wav(duration_s=1.0)

    def _synthesize_windows_sapi(self, text: str) -> bytes:
        """Use Windows PowerShell System.Speech to generate a WAV file."""
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
                timeout=10.0,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _generate_silence_wav(self) -> bytes:
        """Generate a minimal valid 16-bit PCM WAV with silence."""
        import struct
        sample_rate = 16000
        num_samples = 1600  # 0.1s
        header = self._wav_header(sample_rate, num_samples, num_channels=1)
        data = b"\x00\x00" * num_samples
        return header + data

    def _generate_tone_wav(self, duration_s: float = 0.5) -> bytes:
        """Generate an alert chime WAV."""
        import math
        import struct
        sample_rate = 16000
        num_samples = int(sample_rate * duration_s)
        header = self._wav_header(sample_rate, num_samples, num_channels=1)
        samples = []
        freq = 440.0
        for i in range(num_samples):
            t = i / sample_rate
            value = int(8000.0 * math.sin(2.0 * math.pi * freq * t))
            samples.append(struct.pack("<h", value))
        return header + b"".join(samples)

    def _wav_header(self, sample_rate: int, num_samples: int, num_channels: int = 1) -> bytes:
        import struct
        byte_rate = sample_rate * num_channels * 2
        block_align = num_channels * 2
        data_size = num_samples * block_align
        chunk_size = 36 + data_size
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            chunk_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            16,
            b"data",
            data_size,
        )


synthesizer = NeuralVoiceSynthesizer()


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Health check for Colab Server #2."""
    return {
        "status": "healthy",
        "service": "astra_voice_server",
        "voice": synthesizer.default_voice,
        "timestamp": time.time(),
    }


@app.post("/api/v1/voice/synthesize")
async def synthesize_speech(request: SpeechSynthesisRequest) -> Response:
    """
    Synthesizes voice alert / guidance into WAV audio bytes.
    Accepts direct text or a structured VoiceInstruction contract.
    """
    spoken_text = ""
    if request.instruction:
        spoken_text = request.instruction.to_speech_text()
    elif request.text:
        spoken_text = request.text

    if not spoken_text.strip():
        raise HTTPException(status_code=400, detail="Either 'text' or 'instruction' must be provided.")

    try:
        audio_bytes = await synthesizer.synthesize(spoken_text, voice=request.voice)
        # Determine media type based on format (MP3 if edge-tts raw, else audio/wav)
        media_type = "audio/mpeg" if audio_bytes.startswith(b"\xff\xfb") or b"ID3" in audio_bytes[:10] else "audio/wav"
        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": "inline; filename=astra_guidance.wav",
                "X-Astra-Spoken-Text": spoken_text.replace("\n", " "),
            },
        )
    except Exception as e:
        logger.error(f"Speech synthesis failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
