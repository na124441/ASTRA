# ASTRA Collector — Synthesizer Operator Onboarding Guide

## Overview
**ASTRA Collector** is the standardized mobile data collection terminal for ASTRA-E (Autonomous Space Task Recognition & Assistance for Experiments), designed for the Bhartiya Antariksh Station (BAS / SIH 26174).

It operates as a **zero-install Progressive Web Application (PWA)** served directly by the ASTRA Upload API service. Synthesizers open a URL in their phone's browser, record experimental procedures, and the app automatically uploads the video and metadata directly into our private Hugging Face Dataset repository (`na124441/astra-e-raw`).

---

## 1. Zero-Install Launch (Phone Browser)

1. **Connect to Wi-Fi**: Connect your phone (Android or iPhone) to the same Wi-Fi network as the collection server.
2. **Open the Terminal URL**: In Chrome, Safari, or Firefox, navigate to:
   ```text
   http://<SERVER_IP>:8000/collector
   ```
   *(e.g., `http://192.168.1.15:8000/collector` or your hosted domain)*.
3. **Add to Home Screen (Optional & Recommended)**:
   - **Android (Chrome)**: Tap the three dots (⋮) -> **Install app** or **Add to Home screen**.
   - **iOS (Safari)**: Tap the Share button -> **Add to Home Screen**.
   - The app now launches full-screen without any browser address bar.
4. **Grant Permissions**:
   - When prompted, tap **Allow** for **Camera** and **Microphone**.

---

## 2. Connecting & Fetching Tasks

On the **Collector Login** screen:
1. **Server URL**: Defaults automatically to the current server origin.
2. **Collector ID**: Enter your assigned synthesizer identifier (e.g., `COL-001`, `COL-007`).
3. Tap **CONNECT TERMINAL**.
4. The terminal registers your device and immediately dispenses your next assigned experimental task (`EXP001`).

---

## 3. Recording Standards & Protocol Rigor

To ensure pristine training data for our 26-D kinematic feature extractor:

- **Orientation**: **Always mount phone in Landscape mode** on a stable tripod or fixed mount facing the experimental workspace.
- **Lighting**: Ensure even illumination across the workstation. Avoid direct harsh glare or deep shadows.
- **Workstation Objects**:
  - `RED_COMPONENT` (Primary target object)
  - `YELLOW_COMPONENT` (Secondary / distractor object)
  - `TARGET_A` (Designated placement zone)
  - `TARGET_B` (Alternative placement zone)
- **Framing**:
  - The astronaut / operator's hand and both objects must be clearly visible in frame throughout the entire procedure.
- **Duration Bounds**:
  - Standard procedure duration is **30 to 60 seconds**.
  - The review screen automatically validates if the recording satisfies duration constraints.

---

## 4. Collection Step-by-Step

```text
┌────────────────────────────────────────────────────────┐
│ 1. Review Task Screen                                  │
│    - Note Run ID (e.g., RUN-0042)                      │
│    - Note Camera ID (e.g., CAM-01)                     │
│    - Check Scenario (NOMINAL vs FAULT INJECTION)       │
│    - Read 8-step protocol instructions                 │
│                                                        │
│ 2. Tap [START RECORDING]                               │
│    - Workstation objects stationary                    │
│    - Perform protocol smoothly                         │
│    - Monitor blinking REC timer                        │
│                                                        │
│ 3. Tap [STOP RECORDING]                                │
│                                                        │
│ 4. Review Screen                                       │
│    - Preview video playback                            │
│    - Verify duration, resolution (1920x1080)           │
│    - Wait for Web Crypto SHA-256 checksum generation   │
│    - Tap [UPLOAD TO CLOUD]                             │
│                                                        │
│ 5. Resumable Chunk Streaming                           │
│    - 8 MB chunks streamed to backend                   │
│    - Backend reassembles file and validates SHA-256    │
│    - Server commits video + metadata to Hugging Face   │
│    - Green badge: DATASET UPLOAD VERIFIED              │
│    - Local IndexedDB storage automatically cleaned     │
│                                                        │
│ 6. Tap [DISPENSE NEXT TASK]                            │
└────────────────────────────────────────────────────────┘
```

---

## 5. Critical Fail-Closed Safety

> [!IMPORTANT]
> **No Verified Remote Upload $\implies$ No Local Delete**:
> When a recording finishes, the video blob is immediately preserved in browser `IndexedDB`. If your Wi-Fi drops, the phone runs out of battery, or the browser closes, **the local video is never lost**.
>
> When you reopen the terminal, simply tap **RETRY UPLOAD**. The local file is purged **strictly after** the backend returns cryptographic proof (`status: "verified"`) of remote persistence on Hugging Face.
