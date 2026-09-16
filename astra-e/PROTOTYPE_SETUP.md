# ASTRA-E Prototype v0.1: Complete Setup & Deployment Guide

This guide walks you through setting up and running the complete closed-loop **ASTRA-E Prototype v0.1**:

$$\textbf{Live Video} \longrightarrow \textbf{Temporal Understanding} \longrightarrow \textbf{Experiment-State Tracking} \longrightarrow \textbf{Deviation Detection} \longrightarrow \textbf{Natural Voice Guidance}$$

---

## 1. System Topology Overview

```
┌────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE                      │
│                                                        │
│  [Camera] ──► [16-Frame Buffer] ──► [UI / Dashboard]   │
│                      │                                 │
│                      │ 16-frame video windows          │
│                      ▼                                 │
│        [CloudVisionClient] (HTTP POST)                 │
└──────────────────────┬─────────────────────────────────┘
                       │
             Public HTTPS Tunnel (ngrok)
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│          GOOGLE COLAB SERVER #1 (GPU Runtime)          │
│                  VIDEO UNDERSTANDING                   │
│                                                        │
│  • Video Transformer / Temporal Action Recognizer      │
│  • Extracts: Verb, Object, Target, Confidence          │
│  • Emits: Structured Observation JSON                  │
└──────────────────────┬─────────────────────────────────┘
                       │
             Structured Observation JSON
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE                      │
│                    ASTRA REASONER                      │
│                                                        │
│  • Deterministic State Machine (Sample Transfer EXP001)│
│  • Temporal Alignment & Deviation Model                │
│  • Alert Policy Engine (5s Cooldown, Fatigue Control)  │
│  • Guidance Mode vs. Monitor Mode                      │
│  • Conversational Assistant ('Ask ASTRA')              │
└──────────────────────┬─────────────────────────────────┘
                       │
             Voice Instruction JSON
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│             GOOGLE COLAB SERVER #2 (CPU/GPU)           │
│                   VOICE / TTS MODEL                    │
│                                                        │
│  • Neural Speech Synthesizer (Edge-TTS / Custom Model) │
│  • Generates: WAV / MP3 Audio Stream                   │
└──────────────────────┬─────────────────────────────────┘
                       │
             Streaming Audio (WAV)
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│  [Audio Worker] ──► Local Speakers / Headphones        │
└────────────────────────────────────────────────────────┘
```

---

## 2. Setting Up Google Colab Server #1 (Video Understanding)

### Step 2.1: Open Colab & Select GPU Runtime
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Click **New Notebook**.
3. In the menu, go to **Runtime $\to$ Change runtime type** and select **T4 GPU**.

### Step 2.2: Paste and Run Server #1 Code
In the first cell, paste and run:

```python
# ==============================================================================
# ASTRA-E COLAB SERVER #1: VIDEO UNDERSTANDING
# ==============================================================================
!pip install -q fastapi uvicorn opencv-python pydantic pyngrok httpx

import os
import subprocess
import time
import cv2
import numpy as np
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pyngrok import ngrok

# 1. OPTIONAL: Paste your free ngrok auth token from https://dashboard.ngrok.com
# ngrok.set_auth_token("YOUR_NGROK_TOKEN")

app = FastAPI(title="ASTRA-E Video Understanding Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class WindowRequest(BaseModel):
    window_start: float = 0.0
    window_end: float = 0.0
    num_frames: int = 16
    encoded_frames: list[str] = []
    simulation_hint: str | None = None

# --- WHERE TO LOAD YOUR VIDEO TRANSFORMER MODEL ---
# Example:
# import torch
# model = torch.load("/content/models/video_transformer.pt").cuda().eval()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "astra_colab_vision", "gpu": True}

@app.post("/api/v1/vision/analyze_window", response_model=StructuredObservation)
def analyze(req: WindowRequest):
    # Decode incoming 16-frame window from Base64
    frames = []
    for b64 in req.encoded_frames:
        try:
            arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        except Exception:
            pass

    # Handle simulation triggers if provided
    if req.simulation_hint:
        h = req.simulation_hint.lower()
        if "pick" in h:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator picks up the sample container.",
                events=[TemporalEvent(action="PICK", object="SAMPLE_CONTAINER")],
                confidence=0.94
            )
        elif "move" in h:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator moves the sample container toward the chamber.",
                events=[TemporalEvent(action="MOVE", object="SAMPLE_CONTAINER", target="CHAMBER")],
                confidence=0.91
            )
        elif "open" in h:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator opens the chamber door.",
                events=[TemporalEvent(action="OPEN", object="CHAMBER")],
                confidence=0.95
            )
        elif "place" in h:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator places sample container inside chamber.",
                events=[TemporalEvent(action="PLACE", object="SAMPLE_CONTAINER", target="CHAMBER")],
                confidence=0.93
            )
        elif "close" in h:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator closes chamber door.",
                events=[TemporalEvent(action="CLOSE", object="CHAMBER")],
                confidence=0.96
            )
        elif "wrong_object" in h:
            return StructuredObservation(
                window_start=req.window_start, window_end=req.window_end,
                observation="Operator interacts with unauthorized secondary container.",
                events=[TemporalEvent(action="PICK", object="WRONG_CONTAINER")],
                confidence=0.92
            )

    # --- MODEL INFERENCE ---
    # Put your video transformer / feature inference here:
    # tensor = preprocess(frames).cuda()
    # output = model(tensor)
    
    return StructuredObservation(
        window_start=req.window_start,
        window_end=req.window_end,
        observation="Operator observing experiment bay.",
        events=[TemporalEvent(action="IDLE")],
        confidence=0.88
    )

# Save and run background daemon
with open("colab_vision.py", "w") as f:
    pass

import uvicorn, threading
t = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8001), daemon=True)
t.start()
time.sleep(2)

# Expose through tunnel
tunnel_url = ngrok.connect(8001)
print("\n" + "="*60)
print(f"🚀 SERVER #1 IS READY!")
print(f"PUBLIC URL: {tunnel_url}")
print("Copy this URL and paste it into your local dashboard!")
print("="*60 + "\n")
```

4. **Copy the printed Public URL** (e.g. `https://xxxx.ngrok-free.app`).

---

## 3. Setting Up Google Colab Server #2 (Voice / TTS)

### Step 3.1: Open Second Colab Notebook
1. Open a second notebook in Google Colab (CPU runtime is fine).

### Step 3.2: Paste and Run Server #2 Code
In the first cell, paste and run:

```python
# ==============================================================================
# ASTRA-E COLAB SERVER #2: VOICE & TEXT-TO-SPEECH
# ==============================================================================
!pip install -q fastapi uvicorn edge-tts pydantic pyngrok

import os, time, subprocess, threading
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyngrok import ngrok

# 1. OPTIONAL: Set ngrok auth token:
# ngrok.set_auth_token("YOUR_NGROK_TOKEN")

app = FastAPI(title="ASTRA-E Voice & TTS Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SpeechRequest(BaseModel):
    text: str | None = None
    voice: str = "en-IN-NeerjaNeural"

@app.get("/health")
def health():
    return {"status": "healthy", "service": "astra_colab_voice"}

@app.post("/api/v1/voice/synthesize")
async def synthesize(req: SpeechRequest):
    spoken = req.text or ""
    if not spoken.strip():
        raise HTTPException(status_code=400, detail="Text required")

    try:
        import edge_tts
        communicate = edge_tts.Communicate(spoken, req.voice)
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

t = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8002), daemon=True)
t.start()
time.sleep(2)

tunnel_url = ngrok.connect(8002)
print("\n" + "="*60)
print(f"🔊 SERVER #2 IS READY!")
print(f"PUBLIC URL: {tunnel_url}")
print("Copy this URL and paste it into your local dashboard!")
print("="*60 + "\n")
```

4. **Copy the printed Public URL** (e.g. `https://yyyy.ngrok-free.app`).

---

## 4. Running the Local ASTRA-E Prototype

### Option A: Launch the Interactive Streamlit Dashboard (Recommended)

1. Open PowerShell on your local machine in `e:\ASTRA\astra-e`:
   ```powershell
   streamlit run apps/prototype/dashboard.py
   ```
2. Your browser will automatically open to `http://localhost:8501`.
3. In the left sidebar under **Cloud Microservices**:
   - Paste Colab Server #1 URL into **Server #1 (Vision)**
   - Paste Colab Server #2 URL into **Server #2 (Voice / TTS)**
4. The dashboard is now connected to your Colab GPUs!
   - Watch the live camera feed with experiment status overlays.
   - Watch the checklist (`✓ PICK`, `✓ MOVE`, `➔ OPEN`, `○ PLACE`, `○ CLOSE`).
   - Hear synthesized voice guidance and deviation alerts through your headphones.
   - Use the **Ask ASTRA** box to ask questions like *"What am I supposed to do now?"*.

---

### Option B: Run Automated Closed-Loop Milestone Demo (CLI)

To test the entire 5-step procedure, deviation injection, 5-second cooldown suppression, and conversational Q&A in the console:

```powershell
python apps/prototype/run_prototype.py --mode demo
```

---

## 5. Running Models Locally (Offline Without Colab)

If you have a local NVIDIA GPU or want to run everything offline:

### Where Model Weights Live
Place model files in:
```text
e:\ASTRA\astra-e\models\
├── activity\
│   └── temporal_model.pt               <-- Video transformer model weights
└── tts\
    └── voice_model.onnx                <-- Offline TTS model weights (optional)
```

### Starting Local Servers
Open two terminals in `e:\ASTRA\astra-e`:
* **Terminal 1**:
  ```powershell
  python apps/prototype/server_vision.py
  ```
  *(Runs locally on `http://localhost:8001`)*

* **Terminal 2**:
  ```powershell
  python apps/prototype/server_voice.py
  ```
  *(Runs locally on `http://localhost:8002`)*

* **Terminal 3**:
  ```powershell
  streamlit run apps/prototype/dashboard.py
  ```

---

## 6. Verifying Connectivity (Quick Smoke Tests)

You can verify both servers from PowerShell before running the dashboard:

```powershell
# Test Vision Server:
curl http://localhost:8001/health
# (or curl https://xxxx.ngrok-free.app/health)

# Test Voice Server:
curl http://localhost:8002/health
# (or curl https://yyyy.ngrok-free.app/health)
```

Both should return `{"status": "healthy"}`.
