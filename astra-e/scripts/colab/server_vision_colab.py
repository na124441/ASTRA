"""Google Colab Launcher for ASTRA-E Server #1 (Video Understanding).

Run this in a Google Colab notebook (GPU runtime recommended).
It installs requirements, starts the FastAPI vision server, and exposes
a public HTTPS tunnel via ngrok or pyngrok.
"""

# In Google Colab, paste and run:
# !pip install -q fastapi uvicorn opencv-python pydantic pyngrok

import os
import subprocess
import time

COLAB_SERVER_CODE = '''
import base64
import time
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="ASTRA-E Video Understanding (Colab Server #1)")

class TemporalEvent(BaseModel):
    action: str
    object: str | None = None
    target: str | None = None

class StructuredObservation(BaseModel):
    window_start: float
    window_end: float
    observation: str
    events: list[TemporalEvent]
    confidence: float

class WindowAnalysisRequest(BaseModel):
    window_start: float = 0.0
    window_end: float = 0.0
    num_frames: int = 16
    encoded_frames: list[str] = []
    simulation_hint: str | None = None

@app.get("/health")
def health():
    return {"status": "healthy", "service": "astra_colab_vision", "gpu_active": True}

@app.post("/api/v1/vision/analyze_window", response_model=StructuredObservation)
def analyze(req: WindowAnalysisRequest):
    # If simulation hint is given, return corresponding structured observation
    if req.simulation_hint:
        hint = req.simulation_hint.lower()
        if "pick" in hint:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator picks up the sample container.",
                events=[TemporalEvent(action="PICK", object="SAMPLE_CONTAINER")],
                confidence=0.94
            )
        elif "move" in hint:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator moves sample container toward chamber.",
                events=[TemporalEvent(action="MOVE", object="SAMPLE_CONTAINER", target="CHAMBER")],
                confidence=0.91
            )
        elif "open" in hint:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator opens the chamber door.",
                events=[TemporalEvent(action="OPEN", object="CHAMBER")],
                confidence=0.95
            )
        elif "place" in hint:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator places sample container inside chamber.",
                events=[TemporalEvent(action="PLACE", object="SAMPLE_CONTAINER", target="CHAMBER")],
                confidence=0.93
            )
        elif "close" in hint:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator closes chamber door.",
                events=[TemporalEvent(action="CLOSE", object="CHAMBER")],
                confidence=0.96
            )
        elif "wrong_object" in hint:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator is interacting with unauthorized secondary container.",
                events=[TemporalEvent(action="PICK", object="WRONG_CONTAINER")],
                confidence=0.92
            )

    return StructuredObservation(
        window_start=req.window_start,
        window_end=req.window_end,
        observation="Operator observing experiment bay.",
        events=[TemporalEvent(action="IDLE")],
        confidence=0.89
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
'''

def main() -> None:
    print("Writing colab vision server script...")
    with open("vision_server.py", "w") as f:
        f.write(COLAB_SERVER_CODE)

    print("Starting Uvicorn Server in background...")
    proc = subprocess.Popen(["uvicorn", "vision_server:app", "--host", "0.0.0.0", "--port", "8001"])
    time.sleep(2)

    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(8001)
        print("\n" + "=" * 60)
        print(f"🚀 ASTRA-E Vision Server is LIVE on Colab!")
        print(f"Public URL: {public_url}")
        print(f"Set this URL in your ASTRA-E Local Runtime Client config!")
        print("=" * 60 + "\n")
    except ImportError:
        print("pyngrok not installed. Access locally at http://localhost:8001")

if __name__ == "__main__":
    main()
