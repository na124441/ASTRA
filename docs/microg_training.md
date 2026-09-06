# ASTRA-E MicroG-4M Temporal Baseline Training Guide

## 1. Overview & Role of MicroG-4M in ASTRA-E
MicroG-4M is used as a **temporary surrogate baseline** for action recognition to validate the temporal modeling and training infrastructure for the ASTRA-E prototype demonstration while our real experimental procedure videos are being collected.

### Key Architectural Invariant: Modular Isolation
- The MicroG adapter is strictly modular and decoupled from the core ASTRA-E contracts.
- Existing ASTRA-E schemas and contracts (`ml/datasets/schemas.py`, `ml/datasets/sequence_generator.py`, `ml/datasets/splits.py`) remain completely untouched and uncontaminated.
- Once the physical ASTRA-E dataset is collected, the MicroG adapter can be swapped out for the native ASTRA-E dataset loader with zero modification to the causal temporal modeling framework.

---

## 2. Dataset Reality Check & Video Availability

### Hugging Face Repository Audit (`lei-qi-233/MicroG-4M`)
Inspection of the official Hugging Face repository revealed:
- **Available metadata**:
  - `annotation_files/actions.csv`: 13,261 annotated action segments across 4,759 unique video IDs.
  - `annotation_files/bounding_boxes.csv`: Person bounding box annotations.
  - `label_map/label_map.pbtxt`: Protobuf text label map specifying 50 action categories.
  - `videos/video_id_list.pdf`: Listing of original source videos.
- **Unavailable files**: Raw `.mp4` video files are **not** hosted on Hugging Face due to copyright and licensing constraints (clips are sampled from films such as *Apollo 13*, *Gravity*, *The Martian*, and public spaceflight archives).

### Scientific Integrity & Fail-Closed Design
In accordance with ASTRA-E scientific integrity requirements:
- **Never Fake Video Data**: The training script does not fabricate synthetic or random video frames to train on.
- **Fail-Closed Validation**: If neither `--video-dir` nor `--feature-dir` is provided (and `--dry-run` is not enabled), the training script exits immediately with exit code 1 and prints an actionable diagnostic banner.

---

## 3. Architecture & Data Flow

```
MicroG Annotations (actions.csv)
       ↓
Deterministic Taxonomy (Sparse 1..80 → Contiguous 0..49)
       ↓
Grouped Video Splitter (Seed 42, Zero Overlap Between Splits)
       ↓
Causal 30-Frame Temporal Windowing (X_i = F[i:i+30], y_i = Y[i+29])
       ↓
CausalMicroGLSTM (InputProj → 2-layer Causal LSTM → Endpoint LayerNorm → Classifier)
       ↓
Loss & Optimization (Class-Weighted CrossEntropy + AdamW + AMP)
       ↓
Outputs (best.pt, label_map.json, split_manifest.json, metrics.json)
```

### Deterministic Taxonomy Mapping
MicroG defines 50 sparse action IDs (e.g. 1, 3, 5, ..., 80). The `MicroGTaxonomy` module maps these deterministically to contiguous indices `0..49`, providing bidirectional conversion:
- Sparse to Contiguous: `taxonomy.to_contiguous(action_id)`
- Contiguous to Sparse: `taxonomy.to_sparse(class_idx)`
- Serialized to: `outputs/microg_baseline/label_map.json`

### Grouped Video Splitting (Zero Temporal Leakage)
To prevent temporal data leakage:
- Videos are grouped strictly by `video_id`.
- All temporal windows belonging to a video are assigned together to exactly one split (70% train / 15% val / 15% test).
- `train_videos`, `val_videos`, and `test_videos` are strictly disjoint:
  $$	ext{train} \cap 	ext{val} = \emptyset, \quad 	ext{train} \cap 	ext{test} = \emptyset, \quad 	ext{val} \cap 	ext{test} = \emptyset$$
- Serialized to: `outputs/microg_baseline/split_manifest.json`

### Causal 30-Frame Temporal Windowing
- Window size $T = 30$ frames (1 second at 30 fps).
- Frame sequence: $X_i = F[i : i + 30]$
- Ground-truth target: $y_i = Y[i + 29]$ (causal endpoint).
- **Strictly Causal**: The model only has access to past and present frames $t \le i+29$, with zero future lookahead.

### Temporal Model (`CausalMicroGLSTM`)
- **Input Dimension**: 128 (configurable feature vector per frame)
- **Input Projection**: `Linear(128, 256)` + `LayerNorm(256)` + `ReLU` + `Dropout(0.2)`
- **Recurrent Core**: 2-layer unidirectional `nn.LSTM(hidden_size=256, dropout=0.2, bidirectional=False)`
- **Endpoint Representation**: `LayerNorm` applied to hidden state at $t = 29$
- **Classification Head**: `Linear(256, 50)` producing logits over the 50 classes.

---

## 4. Google Colab GPU Setup & Execution

### Option A: Running on Google Colab (Standard Workflow)

1. Open a new Google Colab notebook and select a GPU runtime (**Runtime** $	o$ **Change runtime type** $	o$ **T4 GPU**).
2. Clone the repository and install requirements:
```bash
!git clone https://github.com/na124441/ASTRA.git
%cd ASTRA
!pip install torch torchvision torchaudio datasets huggingface_hub scikit-learn
```

3. Prepare your video or feature directory:
- If you have downloaded MicroG video clips according to `video_id_list.pdf`:
  ```bash
  # Mount Google Drive or upload clips to data/microg/videos
  !mkdir -p data/microg/videos
  ```
- Or if you have pre-extracted feature tensors:
  ```bash
  !mkdir -p data/microg/features
  ```

4. Run training:
```bash
# Using raw video directory
python scripts/training/train_microg.py     --video-dir data/microg/videos     --output-dir outputs/microg_baseline     --epochs 15     --batch-size 32     --lr 1e-3     --use-class-weights

# Or using pre-extracted feature directory
python scripts/training/train_microg.py     --feature-dir data/microg/features     --output-dir outputs/microg_baseline     --epochs 15     --batch-size 32     --lr 1e-3     --use-class-weights
```

5. Run inference on a 30-frame window:
```bash
python scripts/training/infer_microg.py     --checkpoint outputs/microg_baseline/best.pt     --synthetic-test
```

### Option B: Quick Dry-Run Verification (Pipeline Mechanics)
To verify the complete training pipeline, model checkpointing, and evaluation loop without raw videos:
```bash
python scripts/training/train_microg.py     --dry-run     --output-dir outputs/microg_baseline
```

---

## 5. CLI Arguments Reference

### `scripts/training/train_microg.py`
| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | `lei-qi-233/MicroG-4M` | Hugging Face dataset identifier |
| `--config` | str | `actions` | Dataset configuration sub-split |
| `--actions-csv` | path | `None` | Optional local path to `actions.csv` |
| `--pbtxt` | path | `None` | Optional local path to `label_map.pbtxt` |
| `--video-dir` | path | `None` | Path to directory containing `.mp4` video clips |
| `--feature-dir` | path | `None` | Path to directory containing `.npz` or `.npy` features |
| `--output-dir` | path | `outputs/microg_baseline` | Output directory for checkpoints & metrics |
| `--epochs` | int | `15` | Total training epochs |
| `--batch-size` | int | `32` | Batch size |
| `--lr` | float | `1e-3` | Learning rate for AdamW |
| `--hidden-size` | int | `256` | LSTM hidden dimension |
| `--num-layers` | int | `2` | Number of LSTM layers |
| `--dropout` | float | `0.2` | Dropout probability |
| `--window-size` | int | `30` | Temporal window length in frames |
| `--use-class-weights` | flag | `True` | Balance CrossEntropyLoss with inverse frequencies |
| `--seed` | int | `42` | Global random seed for full determinism |
| `--dry-run` | flag | `False` | Run lightweight single-epoch sanity check |

### `scripts/training/infer_microg.py`
| Argument | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | path | `outputs/microg_baseline/best.pt` | Path to trained model checkpoint |
| `--feature-file` | path | `None` | Path to `.npy` file containing `[30, D]` temporal window |
| `--synthetic-test`| flag | `False` | Run mock causal window inference test |
| `--device` | str | `None` | Target device (`cuda` or `cpu`) |

---

## 6. Generated Output Artifacts

| File | Purpose |
|---|---|
| `outputs/microg_baseline/best.pt` | Best model weights, optimizer state, and architecture hyperparameters |
| `outputs/microg_baseline/label_map.json` | Contiguous 0..49 class index to MicroG sparse action mapping |
| `outputs/microg_baseline/split_manifest.json` | Complete train/val/test video partition audit trail |
| `outputs/microg_baseline/metrics.json` | Final test accuracy, macro precision/recall/F1, and weighted F1 |
| `outputs/microg_baseline/classification_report.json` | Detailed per-class precision, recall, and F1 metrics |
