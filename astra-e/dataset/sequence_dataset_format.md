# ASTRA-E Sequence Dataset Format Specification

This document defines the formal schema, physical storage architecture, and PyTorch data loader interface for the ASTRA-E sequence dataset (`astra-e-features`).

---

## 1. Overview & Architectural Motivation

In temporal action recognition for microgravity experiments, training recurrent or causal attention networks on video streams requires slicing long feature sequences into historical sliding windows.

- **Sample Duration**: 30 frames (1.0 second at 30 FPS).
- **Spatial Feature Dimension**: 26 physical kinematic features (zero ground-truth leakage).
- **Supervision Heads**: Multi-head prediction at endpoint frame $t$:
  - `verb`: Action verb class index
  - `object`: Interacted component class index
  - `target`: Target receptacle class index

### Why not store individual JSON files?
Writing 1,000,000 individual JSON files creates extreme file-system inode exhaustion, high metadata lookup latency, and random disk seeks during training. 

Instead, ASTRA-E separates the dataset into:
1. **Logical Sample Schema**: The conceptual JSON representation for individual items, inspection, and auditing.
2. **Physical Storage Format**: Contiguous binary memory-mapped arrays (`features.npy`) and structured index catalogs (`labels.json`), enabling instant dataset startup, zero RAM consumption via `mmap`, and maximum GPU data-loader throughput.

---

## 2. Logical Sample Representation

Each individual sample logically corresponds to the following schema:

```json
{
  "sequence_id": "EXP001_RUN_001_CAM01_000001",
  "run_id": "RUN-0001",
  "subject_id": "ASTRONAUT-01",
  "video_id": "EXP001_RUN_001_CAM01",
  "start_frame": 0,
  "end_frame": 29,
  "features": [
    [0.50, 0.42, 0.00, 0.00, 0.25, 0.50, 0.00, 0.00, 0.35, 0.60, 0.00, 0.00, 0.28, 0.23, 0.49, 0.38, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.95, 0.92, 0.89],
    "... 30 frames total ..."
  ],
  "verb": 1,
  "object": 1,
  "target": 0
}
```

### Schema Fields
| Field | Type | Description |
|---|---|---|
| `sequence_id` | `str` | Unique sample identifier (`{video_id}_{index:06d}`) |
| `run_id` | `str` | Provenance run tag (e.g., `RUN-0001`) |
| `subject_id` | `str` | Astronaut / operator identifier (e.g., `ASTRONAUT-01`) |
| `video_id` | `str` | Source video clip identifier (e.g., `EXP001_RUN_001_CAM01`) |
| `start_frame` | `int` | Inclusive start frame index of the historical window |
| `end_frame` | `int` | Inclusive end frame index (endpoint of observation $t$) |
| `features` | `list[list[float]]` | Matrix of shape `[30, 26]` of type `float32` |
| `verb` | `int` | Action verb label index from `VERB_VOCAB` |
| `object` | `int` | Interacted object label index from `OBJECT_VOCAB` |
| `target` | `int` | Target zone label index from `TARGET_VOCAB` |

---

## 3. Physical Storage Architecture

```text
astra-e-features/
├── train/
│   ├── features.npy       # Shape: [N_train, 30, 26], float32 (mmap-ready)
│   └── labels.json        # List of sample metadata & labels
├── validation/
│   ├── features.npy       # Shape: [N_val, 30, 26], float32 (mmap-ready)
│   └── labels.json
├── test/
│   ├── features.npy       # Shape: [N_test, 30, 26], float32 (mmap-ready)
│   └── labels.json
└── metadata/
    ├── dataset_manifest.json
    └── feature_contract.json
```

### 3.1 Tensor Dimensions

| Array | Shape | Dtype | Description |
|---|---|---|---|
| **`X` (`features.npy`)** | `[N, 30, 26]` | `float32` | Contiguous sliding windows for all samples |
| **`verb`** | `[N]` | `int64` | Integer class index for verb |
| **`object`** | `[N]` | `int64` | Integer class index for object |
| **`target`** | `[N]` | `int64` | Integer class index for target |

### 3.2 Structure of `labels.json`

The `labels.json` file in each split directory contains an array of label entries with exact 1-to-1 correspondence to the first dimension of `features.npy`:

```json
[
  {
    "sample_idx": 0,
    "sequence_id": "EXP001_RUN_0001_CAM01_000001",
    "run_id": "RUN-0001",
    "subject_id": "ASTRONAUT-01",
    "video_id": "EXP001_RUN_0001_CAM01",
    "start_frame": 0,
    "end_frame": 29,
    "verb": 1,
    "object": 1,
    "target": 0
  },
  {
    "sample_idx": 1,
    "sequence_id": "EXP001_RUN_0001_CAM01_000002",
    "run_id": "RUN-0001",
    "subject_id": "ASTRONAUT-01",
    "video_id": "EXP001_RUN_0001_CAM01",
    "start_frame": 1,
    "end_frame": 30,
    "verb": 1,
    "object": 1,
    "target": 0
  }
]
```

### 3.3 Metadata Files

#### `metadata/feature_contract.json`
Contains the frozen 26-D feature definitions and vocabulary mappings:
```json
{
  "feature_schema_version": "kinematic-26d-v1.0",
  "num_features": 26,
  "window_size": 30,
  "features": [
    {"index": 0, "name": "hand_x", "description": "Perception hand centroid x", "units": "normalized [0, 1]"},
    {"index": 1, "name": "hand_y", "description": "Perception hand centroid y", "units": "normalized [0, 1]"},
    {"index": 2, "name": "hand_vx", "description": "Hand velocity vx", "units": "normalized velocity, 1/sec"},
    {"index": 3, "name": "hand_vy", "description": "Hand velocity vy", "units": "normalized velocity, 1/sec"},
    {"index": 23, "name": "conf_hand", "description": "Hand tracking confidence", "units": "[0, 1], 0 if occluded/lost"}
  ],
  "vocabularies": {
    "verb": ["IDLE", "APPROACH", "TOUCH", "GRASP", "PICK", "MOVE", "PLACE", "RELEASE", "OPEN_CONTAINER", "CLOSE_CONTAINER", "UNKNOWN"],
    "object": ["NONE", "RED_COMPONENT", "YELLOW_COMPONENT", "CONTAINER", "UNKNOWN"],
    "target": ["NONE", "TARGET_A", "TARGET_B", "CONTAINER", "UNKNOWN"]
  }
}
```

#### `metadata/dataset_manifest.json`
Catalogs split counts, total samples, and provenance:
```json
{
  "dataset_name": "astra-e-features",
  "dataset_version": "2026.09.05",
  "num_features": 26,
  "window_size": 30,
  "tensor_layout": "X.shape = [N, 30, 26], float32",
  "labels_layout": "verb = [N], object = [N], target = [N], int64",
  "total_samples": 54200,
  "split_counts": {
    "train": 38100,
    "validation": 8100,
    "test": 8000
  }
}
```

---

## 4. PyTorch Training Integration (Zero-Copy Memmap)

Using `np.load(..., mmap_mode="r")`, multiple PyTorch DataLoader workers read directly from OS page cache without duplicating memory:

```python
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MmapFeatureDataset(Dataset):
    def __init__(self, split_dir: str | Path, mmap_mode: str = "r"):
        self.split_dir = Path(split_dir)
        self.features = np.load(self.split_dir / "features.npy", mmap_mode=mmap_mode)
        
        with open(self.split_dir / "labels.json", "r", encoding="utf-8") as f:
            self.labels = json.load(f)
            
        self.verbs = torch.tensor([item["verb"] for item in self.labels], dtype=torch.long)
        self.objects = torch.tensor([item["object"] for item in self.labels], dtype=torch.long)
        self.targets = torch.tensor([item["target"] for item in self.labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(np.array(self.features[idx], dtype=np.float32)),  # [30, 26]
            "verb": self.verbs[idx],       # scalar long
            "object": self.objects[idx],   # scalar long
            "target": self.targets[idx],   # scalar long
        }

# Usage:
train_ds = MmapFeatureDataset("data/astra-e-features/train")
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)

for batch in train_loader:
    x = batch["features"]  # [64, 30, 26]
    y_verb = batch["verb"] # [64]
    # Train multi-head model
```
