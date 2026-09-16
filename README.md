# 🚀 ASTRA: Autonomous Space Task Recognition & Assistance for Experiments

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15.1-black?logo=next.js&logoColor=white)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19.0-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![SIH Problem](https://img.shields.io/badge/SIH%202024-SIH%2026174-orange.svg)](#-mission-context--problem-statement)

**AI-Based Human Activity and Experiment Procedure Recognition System for On-Board Bharatiya Antariksh Station (BAS) Experiments**

[Key Features](#-key-features) • [System Architecture](#-system-architecture) • [Repository Structure](#-repository-structure) • [Milestone Walkthrough](#-milestone-walkthrough-p0--p5) • [Quickstart](#-quickstart-guide) • [Component Guide](#-component-guide) • [Testing](#-testing--verification)

</div>

---

## 🛰️ Mission Context & Problem Statement

Future Indian human-spaceflight missions and the **Bharatiya Antariksh Station (BAS)** will support complex scientific experiments within microgravity payload racks. During these missions:

1. **High Latency & Communication Blackouts**: Communication between on-orbit astronauts and ground control suffers from intermittent blackouts, high latency, and constrained bandwidth, making continuous real-time ground supervision impossible.
2. **Cognitive Load & Procedure Complexity**: Scientific experiments require rigorous multi-step handling of sensitive samples, specialized apparatus, and chemical/biological specimens under demanding zero-gravity conditions.
3. **Microgravity Spatial Disorientation**: Conventional visual systems rely on gravity-based assumptions (e.g., floor, ceiling, standard human upright posture). In microgravity, crew members and objects float at arbitrary 3D orientations relative to the payload rack.

### The Solution: ASTRA-E
**ASTRA** (**A**utonomous **S**pace **T**ask **R**ecognition & **A**ssistance for **E**xperiments) is an autonomous, on-station AI system that observes experiment execution through fixed payload cameras, analyzes human-object interactions (HOI), tracks experiment state machines in real time, flags procedural deviations, and provides spoken voice guidance—all running **locally at the edge without ground dependency**.

---

## ⚡ Core Architectural Invariants

ASTRA is built upon foundational engineering principles designed for mission-critical spaceflight:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ASTRA SYSTEM PHILOSOPHY                               │
├──────────────────────────────┬──────────────────────────────────────────────┤
│ 1. Perception-Reasoning      │ AI models detect WHAT is happening.          │
│    Decoupling                │ A deterministic State Machine decides        │
│                              │ whether it SHOULD happen at that step.       │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ 2. Microgravity Orientation  │ Coordinates and spatial bounding are pegged  │
│    Invariance                │ to the payload rack frame, NOT gravity "up". │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ 3. Causal Temporal Inference │ Frame windows are strictly causal (past to   │
│                              │ present) with zero future lookahead leakage. │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ 4. Acoustic Fatigue Control  │ 4-5s cooldown windows & semantic suppression │
│                              │ prevent audio alert spam during high stress. │
├──────────────────────────────┼──────────────────────────────────────────────┤
│ 5. Fail-Closed Edge Offline  │ Zero internet dependency; all procedural     │
│    Resilience                │ validation and telemetry logging are local.  │
└──────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 🏛️ System Architecture

```
                  ┌───────────────────────────────┐
                  │      FIXED PAYLOAD CAMERA     │
                  └───────────────┬───────────────┘
                                  │ Video Stream
                                  ▼
                  ┌───────────────────────────────┐
                  │    16/30-Frame Causal Buffer  │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │    PERCEPTION & ACTION RECOGNITION  │
               │   • MobileNetV3 + Unidirectional    │
               │     LSTM Action Recognizer          │
               │   • CausalMicroGLSTM (MicroG-4M)    │
               │   • Human-Object Interaction (HOI)  │
               └──────────────────┬──────────────────┘
                                  │ Structured Observation JSON
                                  │ (Action, Object, Target, Conf)
                                  ▼
               ┌─────────────────────────────────────┐
               │     TEMPORAL ALIGNMENT ENGINE       │
               │   • Deterministic Procedure Graph   │
               │     (FSM / DAG from YAML spec)      │
               │   • Step Progress & History Tracker │
               └──────────────────┬──────────────────┘
                                  │
               ┌──────────────────┴──────────────────┐
               ▼                                     ▼
     [NOMINAL PROGRESSION]                 [DEVIATION DETECTED]
     • Step Validated                      • Skipped Step
     • Next Step Unlocked                  • Out-of-Order Action
                                           • Wrong Tool / Container
                                           • Safety Invariant Breach
                                                     │
                                                     ▼
                                           ┌───────────────────┐
                                           │ Alert Policy      │
                                           │ Engine (Cooldown) │
                                           └─────────┬─────────┘
                                                     │
                                                     ▼
                  ┌─────────────────────────────────────┐
                  │     ASTRONAUT ASSISTANCE INTERFACE  │
                  ├──────────────────┬──────────────────┤
                  │ Voice Synthesis  │ Streamlit HUD &  │
                  │ (Neural TTS /    │ Web Portal       │
                  │ 'Ask ASTRA' Q&A) │ Mission Control  │
                  └──────────────────┴──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │    LOCAL EXPERIMENT FLIGHT LOG      │
                  │ (Lightweight Timestamped Telemetry) │
                  └─────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
ASTRA/
├── astra-e/                         # Core ASTRA-E Engine & Runtime
│   ├── android/                     # Mobile collection client (Android Studio project)
│   │   └── astra-collector/         # Multi-angle camera video recording app
│   ├── api/                         # Edge REST & WebSocket API endpoints
│   ├── apps/                        # Application entrypoints & servers
│   │   ├── prototype/               # Closed-loop demonstration runners
│   │   │   ├── run_prototype.py     # Master CLI runner (demo, ui, servers)
│   │   │   ├── dashboard.py         # Streamlit Mission Operator HUD
│   │   │   ├── server_vision.py     # GPU Video Understanding FastAPI server
│   │   │   └── server_voice.py      # Neural TTS Voice Guidance FastAPI server
│   │   ├── upload_api/              # Authenticated dataset ingestion microservice
│   │   └── runtime/                 # Production daemon runtime
│   ├── astra/                       # Core ASTRA python package
│   │   ├── activity/                # Action recognition, smoothing & hysteresis
│   │   ├── assistance/              # Voice prompts, guidance & conversational Q&A
│   │   ├── contracts/               # Pydantic schemas (events, observations, states)
│   │   ├── interaction/             # Human-Object Interaction (HOI) spatial reasoning
│   │   ├── perception/              # Astronaut tracking, pose & bounding boxes
│   │   ├── procedure/               # Deterministic state machine & DAG compiler
│   │   ├── prototype/               # Orchestration & temporal alignment engines
│   │   ├── storage/                 # Local telemetry & append-only flight loggers
│   │   └── violation/               # Deviation detection & alert cooldown policies
│   ├── configs/                     # Global system configurations
│   ├── dataset/                     # Sequence dataset specs & annotation rules
│   ├── experiments/                 # Declarative experiment procedures (YAML)
│   │   └── EXP001/                  # Sample Two-Component Extraction & Placement
│   ├── ml/                          # Synthetic generators, datasets & evaluation
│   ├── tests/                       # Comprehensive test suite (unit, e2e, replay)
│   └── PROTOTYPE_SETUP.md           # Distributed multi-server setup instructions
│
├── astra-portal/                    # Web Portal & Mission Control Hub
│   ├── frontend/                    # Next.js 15 + React 19 + Tailwind CSS web app
│   │   ├── src/app/page.tsx         # Modern landing page & architecture showcase
│   │   ├── src/app/demo/page.tsx    # Live browser-based experiment sandbox
│   │   ├── src/app/downloads/       # Model checkpoint distribution & checksums
│   │   └── src/components/          # UI components, model cards, demo player
│   └── backend/                     # FastAPI backend
│       ├── api/models.py            # Model registry & catalog endpoints
│       ├── api/downloads.py         # Model artifact binary streaming
│       ├── api/checksum.py          # Real-time SHA-256 integrity verifier
│       ├── api/telemetry.py         # Experiment telemetry stream
│       └── main.py                  # API gateway entrypoint (Port 8000)
│
├── astra/                           # Root Real-Video Inference Package
│   └── inference/
│       ├── realvideo.py             # MobileNetV3 + LSTM action recognition engine
│       └── __init__.py              # Public inference API exports
│
├── docs/                            # Deep-dive documentation & training guides
│   └── microg_training.md           # MicroG-4M temporal baseline training manual
│
├── models/                          # Pretrained weights & model cards
│   └── realvideo/
│       ├── astra_realvideo_lstm_best.pt  # Trained 6-class temporal action model
│       └── README.md                     # Model architecture & evaluation report
│
├── scripts/                         # Machine learning training & utility scripts
│   └── training/
│       ├── train_microg.py          # MicroG-4M causal LSTM training pipeline
│       └── infer_microg.py          # MicroG baseline inference CLI
│
├── tests/                           # Root test suite (real-video & MicroG unit tests)
│   └── unit/
├── Problem-Statement.md             # SIH 26174 Problem Statement specification
└── README.md                        # Master Project Documentation (this file)
```

---

## 🎯 Milestone Walkthrough (P0 – P5)

ASTRA-E fulfills all core objectives through a verified sequence of engineering milestones:

| Milestone | Subsystem | Description | Output Artifact |
| :--- | :--- | :--- | :--- |
| **P0** | Video Windowing | Uniform 16/30-frame temporal buffer ingestion from camera feed | Sliding frame tensor `[B, 16, C, H, W]` |
| **P1** | Structured Observation | Visual encoding & temporal classification | JSON Observation (`action`, `object`, `confidence`) |
| **P2** | Procedural Alignment | Deterministic validation against `procedure.yaml` | `ProceduralAlignmentStatus.ALIGNED` |
| **P3** | Deviation Detection | Detect skipped, repeated, incorrect, or out-of-order steps | `SeverityLevel.DEVIATION` / `WARNING` |
| **P4** | Voice Guidance & Cooldown | Neural voice alerts with 4-5s fatigue cooldown | WAV/MP3 speech stream + Suppressed duplicates |
| **P5** | Closed-Loop Integration | Full end-to-end loop from camera to audio + 'Ask ASTRA' | Real-time interactive operation |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Operating System**: Linux (Ubuntu 22.04+), macOS, or Windows 10/11
- **Python**: `3.10` or higher
- **Node.js**: `v18.18+` or `v20+` (for `astra-portal`)
- **Git** & **Git LFS**

---

### 2. Running the ASTRA-E Prototype

#### Clone the Repository
```bash
git clone https://github.com/na124441/ASTRA.git
cd ASTRA
```

#### Set Up Python Environment
```bash
cd astra-e
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

#### Run the Closed-Loop Milestone Walkthrough (CLI)
Executes an automated end-to-end simulated mission run showing nominal execution, deviation injection (skipped step), alert cooldown suppression, and conversational assistance:
```bash
python apps/prototype/run_prototype.py --mode demo
```

#### Launch the Interactive Streamlit Mission HUD
Launches the full graphical astronaut operator dashboard with simulated camera feeds, interactive deviation injectors, real-time procedure checklists, and the "Ask ASTRA" Q&A assistant:
```bash
# Ensure streamlit is installed (pip install streamlit)
python apps/prototype/run_prototype.py --mode ui
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

### 3. Running the ASTRA Web Portal & Mission Control Hub

The web portal provides a centralized mission dashboard, model catalog, downloads with checksum verification, and live telemetry streaming.

#### Start the FastAPI Backend
```bash
cd astra-portal/backend
python -m venv .venv
# Activate virtual environment (.venv\Scripts\activate or source .venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
- API Swagger Documentation: **[http://localhost:8000/docs](http://localhost:8000/docs)**

#### Start the Next.js 15 Frontend
In a new terminal:
```bash
cd astra-portal/frontend
npm install
npm run dev
```
- Web Application: **[http://localhost:3000](http://localhost:3000)**

---

### 4. Running Real-Video Action Recognition Inference

ASTRA includes an auxiliary temporal action recognition model (`MobileNetV3` + 2-layer LSTM) trained on real RGB video clips:

```powershell
# Checkpoint verification
python -m astra.inference.realvideo --verify --model models/realvideo/astra_realvideo_lstm_best.pt

# Run inference on a video file
python -m astra.inference.realvideo `
    --model models/realvideo/astra_realvideo_lstm_best.pt `
    --video path/to/experiment_clip.mp4 `
    --top-k 3
```

#### Python Programmatic Inference
```python
from pathlib import Path
from astra.inference.realvideo import ASTRARealVideoModel, predict_video

model = ASTRARealVideoModel.from_checkpoint(
    Path("models/realvideo/astra_realvideo_lstm_best.pt"),
    device="cpu",  # or "cuda"
)

result = predict_video(model, Path("path/to/clip.mp4"))
print(f"Predicted Action: {result.action} ({result.confidence * 100:.1f}%)")
print(f"Inference Latency: {result.latency_ms:.2f} ms")
for rank, (label, score) in enumerate(result.top_k, start=1):
    print(f"  {rank}. {label}: {score * 100:.1f}%")
```

---

### 5. Running MicroG-4M Temporal Baseline Training

For microgravity action modeling research, ASTRA includes training scripts supporting the MicroG-4M benchmark with fail-closed integrity checks:

```bash
# Verify dataset mapping and split manifest in dry-run mode
python scripts/training/train_microg.py --dry-run

# Run full causal training on GPU
python scripts/training/train_microg.py \
    --feature-dir data/microg/features \
    --output-dir outputs/microg_baseline \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4
```
For full details, see [`docs/microg_training.md`](docs/microg_training.md).

---

## 🔬 Experiment Specification (`procedure.yaml`)

Experiments are declared in human-readable, machine-executable YAML specifications that define valid states, transitions, required tools, and targets:

```yaml
procedure:
  id: "PROC-EXP-001"
  name: "Sample Two-Component Extraction and Placement"
  version: "1.0"
  experiment_id: "EXP-001"
  objects:
    - "CONTAINER"
    - "RED_COMPONENT"
    - "YELLOW_COMPONENT"
  targets:
    - "TARGET_A"
    - "TARGET_B"
  initial_step_id: "S01"
  terminal_step_ids:
    - "S06"
  steps:
    - id: "S01"
      action: "OPEN_CONTAINER"
      object: "CONTAINER"
      description: "Open the payload experiment container"
      allowed_next: ["S02"]
      optional: false

    - id: "S02"
      action: "PICK"
      object: "RED_COMPONENT"
      description: "Pick the red component from container"
      allowed_next: ["S03"]

    - id: "S03"
      action: "PLACE"
      object: "RED_COMPONENT"
      target: "TARGET_A"
      description: "Place the red component into Target A"
      allowed_next: ["S04"]
```

---

## 📱 ASTRA Mobile Collector (`astra-collector`)

The `android/astra-collector` subproject provides a native Android application designed to collect synchronized multi-camera experiment runs:
- **Synchronized Timecodes**: Frames are stamped with epoch milliseconds for temporal cross-camera alignment.
- **Zero In-App Secrets**: Devices authenticate with short-lived tokens; backend proxies upload to Hugging Face (`na124441/astra-e-raw`).
- **Fail-Closed Storage**: Videos remain queued on local device storage until remote upload integrity is cryptographically validated.

See [`astra-e/docs/collector_api_spec.md`](astra-e/docs/collector_api_spec.md) for full endpoint specifications.

---

## 🧪 Testing & Verification

The repository includes a comprehensive, multi-layer testing matrix:

```bash
# Run root unit tests (Realvideo & MicroG)
pytest tests/unit/ -v

# Run ASTRA-E unit, integration, and e2e tests
cd astra-e
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
pytest tests/replay/ -v       # Determinism & state machine replay
pytest tests/robustness/ -v   # Occlusion & observation dropout robustness
```

---

## ⚙️ Hardware Targets & Edge Deployment

| Platform | Role | Target Latency | Precision |
| :--- | :--- | :--- | :--- |
| **NVIDIA Jetson AGX Orin** | On-Station Flight Edge Node | $< 45$ ms / window | FP16 / INT8 TensorRT |
| **x86_64 Edge Workstation** | Ground Station / Dev Testbed | $< 30$ ms / window | FP32 / FP16 PyTorch |
| **Google Colab (T4/V100)** | Remote Accelerated Development | $< 60$ ms / window | FP16 AMP PyTorch |
| **Edge Mobile (Android)** | Fixed Rack Multi-Angle Camera | 30 FPS Capture | Hardware H.264 / HEVC |

---

## 🛡️ License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Developed for Smart India Hackathon (SIH 26174) • Dedicated to Indian Human Spaceflight & Bharatiya Antariksh Station (BAS)**

</div>
