# ASTRA-E Real-Video Temporal Action Recognition Prototype

> [!WARNING]
> **SURROGATE-DOMAIN PROTOTYPE ONLY**
> This model is an **AUXILIARY REAL-VIDEO PROTOTYPE** evaluated on a 6-class human-action subset from HMDB51.
> It is **NOT** the final BAS (Basic Action System) model for ASTRA-E, and does **NOT** replace the core 26-D kinematic feature pipeline or perception contracts.

---

## 1. Model Purpose
To validate the temporal recurrent action recognition pipeline on real-world RGB video files before physical spaceflight experiment procedure videos are recorded for the ASTRA-E system.

---

## 2. Training Dataset & Domain
- **Benchmark**: HMDB51 six-class auxiliary human action benchmark.
- **Total Videos**: 420 real video clips.
  - **Training Split**: 300 videos.
  - **Validation Split**: 120 videos.
- **Hardware & Environment**: Trained on Google Colab (Tesla T4 GPU).
- **Validation Metrics**:
  - Validation Accuracy: **57.50%**
  - Validation Macro-F1: **56.89%**

---

## 3. Six Action Classes & Index Mapping
The classifier outputs logits over exactly 6 classes in this strict deterministic order:
| Class Index | Action Label |
|---|---|
| 0 | `brush_hair` |
| 1 | `drink` |
| 2 | `eat` |
| 3 | `pour` |
| 4 | `clap` |
| 5 | `wave` |

---

## 4. Model Architecture
```
Video File (.mp4, .avi, .mov, .mkv, .webm)
    ↓
16 Uniformly Sampled Frames across [0, N-1]
    ↓
Frame Preprocessing: BGR → RGB → 160 × 160 → ImageNet Normalization
    ↓
MobileNetV3-Small Visual Encoder (classifier = Identity())
    ↓
576-D Visual Feature Vector per frame
    ↓
16 × 576 Temporal Sequence
    ↓
2-Layer Unidirectional LSTM (hidden_size = 128, bidirectional = False)
    ↓
Endpoint Hidden State [1, 128] (at t = 15)
    ↓
Linear Classifier [128 → 6] + Softmax
    ↓
Predicted Action + Confidence + Top-K + Latency (ms)
```

---

## 5. Temporal Sampling Specifications
- Exactly **16 frames** are sampled per sequence.
- Uniform distribution across `[0, N-1]`: `indices = np.linspace(0, N-1, num=16).round().astype(int)`.
- Videos with fewer than 16 frames are resampled/interpolated deterministically to guarantee a complete 16-frame sequence without crashing.

---

## 6. Checkpoint Location
- Canonical repository location:
  ```text
  models/realvideo/astra_realvideo_lstm_best.pt
  ```
- Also accessible from:
  ```text
  astra-e/models/realvideo/astra_realvideo_lstm_best.pt
  ```

---

## 7. How to Run Inference

### CLI Execution
```powershell
python -m astra.inference.realvideo `
    --model models/realvideo/astra_realvideo_lstm_best.pt `
    --video data/cloud/smoke_test/EXP001_SMOKE_CAM01.mp4 `
    --top-k 3
```

### Checkpoint Verification Only
```powershell
python -m astra.inference.realvideo --verify --model models/realvideo/astra_realvideo_lstm_best.pt
```

### Python Programmatic API
```python
from astra.inference.realvideo import predict_video

result = predict_video(
    video_path="path/to/video.avi",
    model_path="models/realvideo/astra_realvideo_lstm_best.pt",
    top_k=3,
)

print(result["prediction"])      # e.g. "pour"
print(result["confidence"])      # e.g. 0.8742
print(result["top_predictions"]) # e.g. [{"action": "pour", "confidence": 0.8742}, ...]
print(result["latency_ms"])      # e.g. 182.4
```

---

## 8. Example Output
```text
============================================================
ASTRA-E REAL-VIDEO TEMPORAL INFERENCE
============================================================

Model:
  astra_realvideo_lstm_best.pt

Device:
  cpu

Video:
  EXP001_SMOKE_CAM01.mp4

Frames:
  16

Prediction:
  EAT

Confidence:
  18.05%

Top Predictions:
  1. EAT          18.05%
  2. BRUSH_HAIR   17.71%
  3. POUR         16.87%

Inference latency:
  194.90 ms

============================================================
```

---

## 9. Known Limitations
1. **Domain Sensitivity**: The prototype is trained exclusively on 6 terrestrial actions. Non-action videos, background scenes, or unmodeled tasks will produce low-confidence out-of-domain predictions.
2. **Surrogate Role**: This prototype tests the MobileNet+LSTM inference pipeline mechanics and latency. It does **not** evaluate microgravity experiment manipulation tasks.
3. **Lighting & Occlusion**: Terrestrial HMDB51 footage exhibits varying resolutions, compression artifacts, and aspect ratios.
