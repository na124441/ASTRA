# 🛰️ ASTRA-E: Core Engine & Runtime

> **Autonomous Space Task Recognition & Assistance for Experiments**  
> Problem Statement: **SIH 26174** (Bharatiya Antariksh Station / BAS)

This subpackage contains the core runtime, deterministic state engine, perceptual contracts, assistance interfaces, and prototype demonstration suites for ASTRA-E.

---

## ⚡ Key Subsystems

- **`astra/activity/`**: Causal temporal action recognition, temporal window smoothing, and confidence calibration.
- **`astra/procedure/`**: Deterministic procedure graph engine & state machine (FSM / DAG) validating steps against declarative YAML protocols.
- **`astra/violation/`**: Multi-tier deviation detection (skipped steps, out-of-order execution, incorrect tools/objects) with alert cooldown policies to mitigate acoustic fatigue.
- **`astra/assistance/`**: Spoken voice guidance generator and grounded conversational dialogue ("Ask ASTRA").
- **`astra/interaction/`**: Human-Object Interaction (HOI) tracking and spatial relation reasoning pegged to the payload rack reference frame.
- **`astra/contracts/`**: Strict Pydantic models governing observations, actions, states, telemetry, and events.
- **`astra/storage/`**: Offline, lightweight, timestamped append-only flight telemetry logging.
- **`android/astra-collector/`**: Native Android companion application for multi-camera video dataset collection.

---

## 🚀 Quickstart

### 1. Installation

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

### 2. Run Closed-Loop Demonstration (Milestones P0 – P5)

Runs the complete closed-loop simulated execution:
```bash
python apps/prototype/run_prototype.py --mode demo
```

### 3. Launch Mission Operator HUD (Streamlit)

```bash
pip install streamlit
python apps/prototype/run_prototype.py --mode ui
```
Navigate to **http://localhost:8501**.

### 4. Run Automated Test Suite

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
pytest tests/replay/ -v
pytest tests/robustness/ -v
```

---

## 📖 Additional Documentation

- [Master Project README](../README.md)
- [Distributed Prototype Setup Guide (Colab + Edge)](PROTOTYPE_SETUP.md)
- [MicroG-4M Temporal Baseline Guide](docs/microg_training.md)
- [ASTRA Collector API Specification](docs/collector_api_spec.md)
- [Sequence Dataset Guidelines](dataset/sequence_dataset_format.md)
