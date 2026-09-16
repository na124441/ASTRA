# 🛰️ ASTRA Data Pipeline Architecture & Specification

> **Subsystem**: Data Ingestion, Kinematic Feature Extraction, Temporal Windowing & Dataset Generation  
> **Mission Context**: Bharatiya Antariksh Station (BAS) / SIH 26174  
> **Target Dataset Package**: `astra-e-features` / `ml.datasets`

---

## 1. Architectural Overview

The ASTRA Data Pipeline transforms raw multi-view video streams and mobile collection recordings from experimental payload racks into strictly causal, supervised temporal sequence datasets suitable for deep recurrent and attention networks.

```
                 RAW VIDEO / FIXED CAMERAS / ANDROID COLLECTOR
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │ 1. Frame Ingestion & Agnostic Detection  │
                  │    • Fixed payload rack reference frame  │
                  │    • Astronaut hand & object bounding    │
                  │    • SceneObservation contract           │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │ 2. Kinematic Feature Extraction (26-D)   │
                  │    • Normalized Euclidean distances      │
                  │    • First-order velocity derivatives    │
                  │    • Detector confidence metrics         │
                  │    • ZERO ground-truth state leakage     │
                  └────────────────────┬─────────────────────┘
                                       │ Continuous feature stream [T, 26]
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │ 3. Annotation Alignment Engine           │
                  │    • Action segments → Frame-level labels│
                  │    • Multi-head: Verb × Object × Target  │
                  │    • Evaluates & excludes violations     │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │ 4. Causal Temporal Windowing Engine      │
                  │    • Window size W = 30 frames (1.0s)    │
                  │    • Stride s = 1                        │
                  │    • X_i = F[i : i+30], y_i = Y[i+29]    │
                  │    • Strict causality (no lookahead)     │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │ 5. Leakage-Safe Dataset Partitioning     │
                  │    • Group-level split: by subject / run │
                  │    • Adjacent windows never cross splits │
                  │    • Multi-camera synchronized binding   │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │ 6. Physical Storage & PyTorch Loaders    │
                  │    • Memory-mapped binary arrays (.npy)  │
                  │    • Zero RAM saturation via mmap        │
                  │    • Multi-head PyTorch DataLoader       │
                  └──────────────────────────────────────────┘
```

---

## 2. Agnostic Visual Detection & Coordinate Normalization

In microgravity, crew members float in arbitrary orientations without a gravitational upright reference. All spatial tracking is therefore pegged to the **experimental payload rack coordinate frame**:
- **Normalized coordinates**: All bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ and centroids $(x_c, y_c)$ are normalized to $[0.0, 1.0]$ relative to the fixed rack camera dimensions.
- **Detector-Agnostic Contract**: Upstream detectors (YOLO, MediaPipe, synthetic simulation) emit standardized `SceneObservation` schemas containing hand coordinates, object coordinates (`CONTAINER`, `RED_COMPONENT`, `YELLOW_COMPONENT`), and target zone centroids (`TARGET_A`, `TARGET_B`).

---

## 3. Observable Kinematic Feature Vector ($26\text{-D}$)

Rather than directly feeding high-dimensional pixel arrays into temporal models, ASTRA extracts an observable, physically grounded **$26\text{-D}$ feature vector** (`KinematicFeatureExtractor`):

| Feature Indices | Description | Type / Scale |
| :--- | :--- | :--- |
| **0 – 1** | Normalized Hand centroid $(x, y)$ | Spatial $[0.0, 1.0]$ |
| **2 – 3** | Red Component centroid $(x, y)$ | Spatial $[0.0, 1.0]$ |
| **4 – 5** | Yellow Component centroid $(x, y)$ | Spatial $[0.0, 1.0]$ |
| **6 – 10** | Normalized Euclidean Distances: Hand to Red, Yellow, Container, Target A, Target B | Relative Distance $[0.0, 1.0]$ |
| **11 – 12** | Component-to-Target Distances: Red to Target A, Yellow to Target B | Relative Distance $[0.0, 1.0]$ |
| **13 – 17** | First-Order Distance Derivatives ($\Delta d / \Delta t$): closure velocities | Rate of Change $[-1.0, 1.0]$ |
| **18 – 20** | Component velocities $(\Delta x / \Delta t, \Delta y / \Delta t)$ | Motion Kinematics |
| **21 – 25** | Visual detector confidence scores for hand, components, and targets | Probability $[0.0, 1.0]$ |

### Zero Ground-Truth Leakage Guarantee
Features represent strictly observable geometric and kinematic properties. No procedure state, current step index, task progress flag, or ground-truth label is ever injected into the feature array.

---

## 4. Multi-Head Label Formulation & Supervision

Supervision at each window endpoint is structured into three independent multi-class heads (`schemas.py`):

### 1. Action Verb Vocabulary (`VERB_VOCAB` — 11 Classes)
```python
VERB_VOCAB = [
    "IDLE", "APPROACH", "TOUCH", "GRASP", "PICK",
    "MOVE", "PLACE", "RELEASE", "OPEN_CONTAINER", "CLOSE_CONTAINER",
    "UNKNOWN"
]
```

### 2. Interacted Object Vocabulary (`OBJECT_VOCAB` — 5 Classes)
```python
OBJECT_VOCAB = [
    "NONE", "RED_COMPONENT", "YELLOW_COMPONENT", "CONTAINER", "UNKNOWN"
]
```

### 3. Target Receptacle Vocabulary (`TARGET_VOCAB` — 5 Classes)
```python
TARGET_VOCAB = [
    "NONE", "TARGET_A", "TARGET_B", "CONTAINER", "UNKNOWN"
]
```

### Why Procedural Violations Are Excluded from Model Heads
The procedural deviation taxonomy (`VIOLATION_VOCAB`: `SKIPPED_STEP`, `WRONG_OBJECT`, `OUT_OF_SEQUENCE`, etc.) is retained **exclusively for evaluation benchmarking**, not as an LSTM prediction head:
1. **Perception vs. Policy Decoupling**: The neural network's role is strictly physical action recognition ($\text{Verb} \times \text{Object} \times \text{Target}$).
2. **Deterministic Verification**: Procedure correctness is verified downstream by the symbolic `ProcedureGraph` engine against flight rules.
3. **Generalization**: Neural networks trained directly on error classes overfit to observed failure modes and degrade on nominal actions.

---

## 5. Strict Causal Temporal Windowing

For continuous recording $\mathbf{F} \in \mathbb{R}^{T \times 26}$ and frame labels $\mathbf{Y} \in \mathbb{N}^T$:
- **Window Size**: $W = 30$ frames ($1.0$ second at $30$ FPS).
- **Stride**: $s = 1$ frame.
- **Total Windows**: $N = T - W + 1 = T - 29$.
- **Observation Tensor**: $\mathbf{X}_i = \mathbf{F}[i : i + 30] \in \mathbb{R}^{30 \times 26}$.
- **Causal Endpoint Label**: $y_i = \mathbf{Y}[i + 29] = \mathbf{Y}[t_{\text{end}}]$.

```
Frame Index:  0  1  2 ... 28  29  30 ... T-1
Window 0:    [------------------]  -> Label = Y[29]
Window 1:       [------------------]  -> Label = Y[30]
                                 ▲
                    Strictly Causal Endpoint Supervision
```

- **Invariant**: The model has zero access to future frames ($t > i + 29$).
- **Boundary Policy**: Temporal windows never cross recording or run boundaries.

---

## 6. Anti-Leakage Partitioning Policy

Adjacent sliding windows share **29 out of 30 frames (96.7% identical feature data)**. 
- **Forbidden**: Random window shuffling into train/val/test splits. Doing so produces artificial $>99\%$ validation accuracy that collapses during actual mission deployment.
- **Mandatory Policy**: Grouped partitioning:
  - **Preferred (`group_by = subject`)**: All runs and recordings of a specific crew member belong to exactly one split ($70\%$ train, $15\%$ val, $15\%$ test).
  - **Fallback (`group_by = run`)**: All camera angles observing a single experiment run belong to the same split.
- **Disjointness Invariant**:
  $$\text{Train}_{\text{run}} \cap \text{Val}_{\text{run}} = \emptyset, \quad \text{Train}_{\text{run}} \cap \text{Test}_{\text{run}} = \emptyset, \quad \text{Val}_{\text{run}} \cap \text{Test}_{\text{run}} = \emptyset$$

---

## 7. Physical Storage & PyTorch Data Loading

To prevent OS file-system inode exhaustion and slow random seeks from millions of JSON files, ASTRA-E uses a binary memory-mapped architecture:

```text
data/processed/EXP001/
├── features.npy           # Contiguous float32 array [Total_Frames, 26]
├── labels.npz             # Memory-mapped integer arrays (verbs, objects, targets)
├── split_manifest.json    # Grouped split partition indices and hashes
└── metadata.json          # Experiment provenance, camera angles, sampling rates
```

### PyTorch Integration (`FeatureSequenceDataset`)
- Instantiates memory-mapped handles (`np.load(..., mmap_mode="r")`).
- Enables instant dataloader startup with zero RAM exhaustion even on datasets exceeding host memory.
- Supports worker parallelism (`DataLoader(num_workers=4, pin_memory=True)`).
