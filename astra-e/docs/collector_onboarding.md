# ASTRA Collector — Synthesizer Operator Onboarding Guide

## Overview
**ASTRA Collector** is the standardized data collection terminal for ASTRA-E (Autonomous Space Task Recognition & Assistance for Experiments), designed for the Bhartiya Antariksh Station (BAS / SIH 26174).

Instead of requiring manual file transfers, renaming, and local disk management, ASTRA Collector turns your phone into a guided recording terminal:
1. Displays assigned experiment task (`EXP001`).
2. Guides procedure execution step-by-step.
3. Records standardized 1080p landscape video at 30 FPS.
4. Validates execution duration and metrics.
5. Computes cryptographic SHA-256 checksums.
6. Streams resumable chunks directly to the private dataset backend.
7. Automatically deletes local recordings **only after remote persistence is cryptographically verified**.

---

## 1. Installation & Initial Setup

1. **Download the APK**: Obtain `astra-collector-release.apk` from your research lead.
2. **Install**: Open the APK file on your Android device (Android 8.0+ / API 26 or higher).
3. **Permissions**: When prompted, grant:
   - **Camera**: Required for procedure capture.
   - **Microphone**: Required for audio recording.
   - **Network**: Required for authenticated upload.
4. **Launch**: Open **ASTRA Collector** from your app drawer.

---

## 2. Connecting to the Collection Backend

On the first screen:
1. **Server URL**: Enter the provided backend endpoint (e.g., `https://astra-upload.internal.org` or local network address `http://192.168.1.100:8000`).
2. **Collector ID**: Enter your assigned collector identifier (e.g., `COL-001`, `COL-007`).
3. Tap **CONNECT TERMINAL**.
4. The app authenticates your device and fetches your first assigned experimental recording task.

---

## 3. Recording Standards & Rigor

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
  - Review screen will alert you if recording is too short or too long.

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
│    - Verify duration, resolution (1920x1080)           │
│    - Wait for SHA-256 checksum generation              │
│    - Tap [UPLOAD]                                      │
│                                                        │
│ 5. Uploading & Verifying                               │
│    - 8 MB chunks streamed to backend                   │
│    - Server verifies hash & commits to Hugging Face    │
│    - Green badge: DATASET UPLOAD VERIFIED              │
│    - Local temporary MP4 automatically deleted         │
│                                                        │
│ 6. Tap [DISPENSE NEXT TASK]                            │
└────────────────────────────────────────────────────────┘
```

---

## 5. Critical Fail-Closed Safety

> [!IMPORTANT]
> **Your data is safe**: If your network connection drops or the app is closed during an upload, **the local video is never deleted**. The background WorkManager will automatically resume chunk streaming once connectivity is restored.
>
> You can also manually tap **RETRY UPLOAD** at any time. Local files are purged **strictly after** the backend returns cryptographic proof of remote persistence.
