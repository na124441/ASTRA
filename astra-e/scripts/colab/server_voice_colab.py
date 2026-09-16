"""Google Colab Launcher for ASTRA-E Server #2 (Voice / TTS).

Run this in a Google Colab notebook (CPU or GPU runtime).
It installs requirements (edge-tts / pyttsx3), starts the FastAPI voice server,
and exposes a public HTTPS tunnel via ngrok or pyngrok.
"""

# In Google Colab, paste and run:
# !pip install -q fastapi uvicorn edge-tts pydantic pyngrok

import os
import subprocess
import time

COLAB_VOICE_CODE = '''
import asyncio
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

app = FastAPI(title="ASTRA-E Voice & TTS (Colab Server #2)")

class SpeechRequest(BaseModel):
    text: str | None = None
    voice: str = "en-IN-NeerjaNeural"

@app.get("/health")
def health():
    return {"status": "healthy", "service": "astra_colab_voice", "voice_engine": "edge-tts"}

@app.post("/api/v1/voice/synthesize")
async def synthesize(req: SpeechRequest):
    spoken_text = req.text or ""
    if not spoken_text.strip():
        raise HTTPException(status_code=400, detail="Text required")

    try:
        import edge_tts
        communicate = edge_tts.Communicate(spoken_text, req.voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=guidance.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
'''

def main() -> None:
    print("Writing colab voice server script...")
    with open("voice_server.py", "w") as f:
        f.write(COLAB_VOICE_CODE)

    print("Starting Uvicorn Voice Server in background...")
    proc = subprocess.Popen(["uvicorn", "voice_server:app", "--host", "0.0.0.0", "--port", "8002"])
    time.sleep(2)

    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(8002)
        print("\n" + "=" * 60)
        print(f"🔊 ASTRA-E Voice / TTS Server is LIVE on Colab!")
        print(f"Public URL: {public_url}")
        print(f"Set this URL in your ASTRA-E Local Runtime Client config!")
        print("=" * 60 + "\n")
    except ImportError:
        print("pyngrok not installed. Access locally at http://localhost:8002")

if __name__ == "__main__":
    main()
