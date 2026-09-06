# AUXILIARY REAL-VIDEO PROTOTYPE

> [!WARNING]
> This model is an **AUXILIARY REAL-VIDEO PROTOTYPE** evaluated on a 6-class subset of HMDB51.
> It is **NOT** the final BAS (Basic Action System) model for ASTRA-E, and does **NOT** replace the core 26-D kinematic feature pipeline.

## Model Metadata

- **Model Name**: ASTRA-E Real-Video Temporal Prototype
- **Dataset**: HMDB51 Six-Class Auxiliary Benchmark
- **Classes**:
  1. `brush_hair` (Class 0)
  2. `drink` (Class 1)
  3. `eat` (Class 2)
  4. `pour` (Class 3)
  5. `clap` (Class 4)
  6. `wave` (Class 5)
- **Input Specifications**:
  - Temporal sampling: 16 uniformly sampled frames across [0, N-1]
  - Spatial resolution: 160 × 160 RGB
  - Normalization: ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Neural Architecture**:
  - Visual Backbone: `MobileNetV3-Small` (ImageNet-pretrained)
  - Feature Dimension: 576
  - Temporal Model: 2-layer unidirectional LSTM
  - Hidden Size: 128
  - Bidirectional: False
  - Classification Head: Linear(128, 6)
- **Reported Benchmark Performance**:
  - Validation Accuracy: 57.50%
  - Validation Macro-F1: 56.89%
  - Hardware: Tesla T4 (Google Colab)
