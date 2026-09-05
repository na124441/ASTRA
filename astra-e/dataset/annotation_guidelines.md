# ASTRA-E Dataset Annotation Guidelines v1.0

**Project**: Autonomous Space Task Recognition & Assistance for Experiments (ASTRA-E)  
**Problem Statement**: SIH 26174 (Bhartiya Antariksh Station / BAS)  
**Schema Version**: `kinematic-26d-v1.0`  
**Target Procedure**: `EXP001` (Microgravity Receptacle & Component Transfer)

---

## 1. Scope & Objective

These guidelines define the standardized protocol for annotating video recordings of astronaut payload experiments. 

### Core Decoupling Principle
The machine learning pipeline predicts **observable physical interactions**:
$$\text{Observation}_t = (\text{Verb}_t, \text{Object}_t, \text{Target}_t)$$

The **Violation Classification** is a **procedural ground-truth annotation**, NOT an LSTM output head:
- **LSTM Responsibility**: "The astronaut is moving the red component toward Target B."
- **Procedure Engine Responsibility**: "The procedure expected placement in Target A; moving to Target B is a `WRONG_TARGET` violation."

Keeping procedural violations separate from perception prevents training/inference fragility and ensures zero ground-truth leakage into the neural network.

---

## 2. Action Taxonomy & Boundary Definitions

Every segment is annotated with a **Verb** from `VERB_VOCAB`, an optional **Object** from `OBJECT_VOCAB`, and an optional **Target** from `TARGET_VOCAB`.

```text
VERB_VOCAB = [
  "IDLE", "APPROACH", "TOUCH", "GRASP", "PICK", "MOVE",
  "PLACE", "RELEASE", "OPEN_CONTAINER", "CLOSE_CONTAINER", "UNKNOWN"
]
```

### 2.1 Action Verbs

| Action Verb | Physical Definition | Onset / Start Criterion | Offset / End Criterion | Physical Indicator |
|:---|:---|:---|:---|:---|
| **`APPROACH`** | The hand travels through free space toward a target object or container with intent to interact. | Hand begins continuous trajectory toward object ($\|\mathbf{v}_{\text{hand}}\| > 0.05$). | Hand reaches within touch contact boundary ($d \le 0.04$ normalized distance). | $\dot{d}(\text{hand}, \text{object}) < 0$, hand deceleration toward target. |
| **`TOUCH`** | Hand makes physical tactile contact with the surface of an object without establishing a firm grip. | First frame where hand boundary overlaps/contacts object. | Hand fingers wrap into a grip, or hand breaks contact. | $d(\text{hand}, \text{object}) \approx 0$, $\|\mathbf{v}_{\text{obj}}\| \approx 0$. |
| **`GRASP`** | Hand fingers curl and establish a stable mechanical closure/coupling with the object. | Fingers begin closing around the object boundary. | Stable grip is formed; hand and object become kinematically locked. | Co-movement velocity norm $\|\mathbf{v}_{\text{hand}} - \mathbf{v}_{\text{obj}}\| \approx 0$. |
| **`PICK`** | Hand lifts, dislodges, or extracts the grasped component away from its resting receptacle or container. | Object begins displacement from its resting socket/slot. | Object is completely clear of container walls or socket boundary. | Vertical/lateral object displacement from base coordinate. |
| **`MOVE`** | Grasped component is transported through the workspace toward another location or receptacle. | Component begins continuous transit through free workspace. | Component reaches target zone perimeter and begins decelerating. | Co-movement norm $\approx 0$, continuous trajectory across workspace. |
| **`PLACE`** | The component is inserted, seated, or positioned into its destination receptacle or target zone. | Component penetrates target boundary zone. | Component is seated at target position; velocity drops to near zero. | $\dot{d}(\text{obj}, \text{target}) < 0$, final alignment at target. |
| **`RELEASE`** | Hand fingers open and disengage, breaking physical coupling with the placed component. | Fingers begin opening or detaching from the object. | Hand has completely separated from object boundary ($d > 0.04$). | Object remains stationary while hand velocity $\mathbf{v}_{\text{hand}}$ directs away. |
| **`OPEN_CONTAINER`** | Hand manipulates and unlatches/opens the payload container door or lid. | Hand contacts container latch/handle and begins opening motion. | Container lid/door reaches fully open stationary position. | Container boundary expands or state transitions to open. |
| **`CLOSE_CONTAINER`**| Hand pulls, swings, or latches the container door/lid shut. | Hand contacts container lid/handle and begins closing motion. | Container is fully shut and secured/latched. | Container boundary returns to closed state. |
| **`IDLE`** | Hand is stationary, resting outside active zones, or moving without directed interaction trajectory. | Hand ceases interaction and enters rest/retreat. | Hand begins an intentional approach trajectory toward a tool/component. | Low hand velocity ($\|\mathbf{v}\| \le 0.02$) or no active interaction. |
| **`UNKNOWN`** | Activity is unclassifiable, out of camera view, or irrelevant to the experiment protocol. | Activity becomes obscured or unclassifiable. | Activity returns to identifiable protocol interactions. | Obscured view or non-standard astronaut behavior. |

---

### 2.2 When is a Frame Considered `AMBIGUOUS`?

Label quality should be marked as **`ambiguous`** under the following specific conditions:

1. **Occlusion & View Shadow**:
   - The astronaut's torso, arm, or external cabling blocks direct line-of-sight of the hand-object interface.
   - Example: Hand reaches into container, but container walls block view of whether the hand is `TOUCH`ing or `GRASP`ing.
2. **Transition Boundary Ambiguity (1–3 Frames)**:
   - The boundary interval between `TOUCH` and `GRASP`, or between `PLACE` and `RELEASE`, where finger micro-movements are sub-pixel at 30 FPS.
   - Rule: Extend the preceding state until unambiguous separation or coupling is observed.
3. **Trajectory Hesitation / Hovering**:
   - Astronaut hovers hand equidistant between two components (`RED_COMPONENT` and `YELLOW_COMPONENT`) without a clear velocity vector.
   - Mark as `IDLE` with `label_quality: "ambiguous"`.
4. **2D Projection Overlap without Depth Coupling**:
   - In single-camera fixed rack views, the hand appears over an object in 2D pixel space, but is floating centimeters above it in depth.
   - If velocity vectors do not couple, annotate as `APPROACH` or `IDLE`, not `GRASP`.

---

## 3. Procedural Violation Semantics

Procedural violations are **ground-truth benchmarks** used to evaluate the Deterministic State Machine (`ProcedureEngine`) and the Confirmation Layer. **They are NOT trained as neural network softmax classes.**

```text
VIOLATION_VOCAB = [
  "NONE", "WRONG_OBJECT", "WRONG_TARGET", "SKIPPED_STEP",
  "REPEATED_STEP", "PREMATURE_CLOSE", "OUT_OF_SEQUENCE", "AMBIGUOUS"
]
```

### Violation Definitions

| Violation Type | Trigger Condition | Example in EXP001 |
|:---|:---|:---|
| **`NONE`** | Action is valid and follows the nominal procedure specification. | Picking `RED_COMPONENT` after opening container. |
| **`WRONG_OBJECT`** | The correct action is performed on an unprescribed component. | Grasping `YELLOW_COMPONENT` when procedure requires `RED_COMPONENT`. |
| **`WRONG_TARGET`** | A component is transported or placed into the incorrect receptacle. | Placing `RED_COMPONENT` into `TARGET_B` (prescribed: `TARGET_A`). |
| **`SKIPPED_STEP`** | A mandatory prerequisite step in the procedure DAG was bypassed. | Attempting to grasp a component inside a closed container before `OPEN_CONTAINER`. |
| **`REPEATED_STEP`** | An already completed step is redundantly re-executed. | Placing a second component into `TARGET_A` after S03 was already completed. |
| **`PREMATURE_CLOSE`**| The container is closed before internal operations are finished. | Executing `CLOSE_CONTAINER` while components remain inside unplaced. |
| **`OUT_OF_SEQUENCE`**| A valid step is attempted out of topological order. | Attempting step S05 before step S02 has been validated. |
| **`AMBIGUOUS`** | Procedural intent cannot be determined due to incomplete movement. | Hand touches wrong target but retreats without placing. |

---

## 4. Complete Annotated Reference Example

### Recording Provenance Metadata
- **Video ID**: `EXP001_RUN_001_CAM01`
- **Experiment ID**: `EXP001`
- **Run ID**: `RUN-0001`
- **Subject**: `ASTRONAUT-01`
- **Camera**: `CAM-01` (Fixed Payload Overhead Rack)
- **Duration**: `8.00` seconds (240 frames @ 30.0 FPS)
- **Resolution**: `640 x 480`
- **Scenario Type**: `nominal`

### Frame-by-Frame Action Segmentation Table

| Segment ID | Frame Range | Timestamp (s) | Verb | Object | Target | Violation | Quality | Notes |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `SEG-01` | `0 – 40` | `0.00 – 1.33` | **`IDLE`** | `NONE` | `NONE` | `NONE` | `verified` | Hand resting at rack lower right rest position. |
| `SEG-02` | `41 – 60` | `1.37 – 2.00` | **`APPROACH`** | `RED_COMPONENT` | `NONE` | `NONE` | `verified` | Hand accelerates toward container interior. $\dot{d} < 0$. |
| `SEG-03` | `61 – 70` | `2.03 – 2.33` | **`GRASP`** | `RED_COMPONENT` | `NONE` | `NONE` | `verified` | Fingers wrap around red component. Velocity drops. |
| `SEG-04` | `71 – 110` | `2.37 – 3.67` | **`PICK`** | `RED_COMPONENT` | `CONTAINER` | `NONE` | `verified` | Component lifted clear of container rim. |
| `SEG-05` | `111 – 180` | `3.70 – 6.00` | **`MOVE`** | `RED_COMPONENT` | `TARGET_A` | `NONE` | `verified` | Transport across workspace. Hand-object co-moving. |
| `SEG-06` | `181 – 200` | `6.03 – 6.67` | **`PLACE`** | `RED_COMPONENT` | `TARGET_A` | `NONE` | `verified` | Component inserted and aligned into receptacle Target A. |
| `SEG-07` | `201 – 215` | `6.70 – 7.17` | **`RELEASE`** | `RED_COMPONENT` | `TARGET_A` | `NONE` | `verified` | Fingers open and disengage. Red stays seated. |
| `SEG-08` | `216 – 240` | `7.20 – 8.00` | **`IDLE`** | `NONE` | `NONE` | `NONE` | `verified` | Hand retreats to workspace neutral position. |

---

### Reference JSON Manifest (`RecordingMetadata`)

```json
{
  "video_id": "EXP001_RUN_001_CAM01",
  "recording_id": "REC-RUN-0001",
  "experiment_id": "EXP001",
  "run_id": "RUN-0001",
  "subject_id": "ASTRONAUT-01",
  "camera_id": "CAM-01",
  "duration_seconds": 8.0,
  "total_frames": 240,
  "fps": 30.0,
  "width": 640,
  "height": 480,
  "scenario_type": "nominal",
  "annotator_id": "ASTRA-LEAD-ANNOTATOR",
  "random_seed": 42,
  "created_at": 1788615600.0,
  "segments": [
    {
      "segment_id": "SEG-01",
      "start_frame": 0,
      "end_frame": 40,
      "start_time": 0.0,
      "end_time": 1.333,
      "verb": "IDLE",
      "object": "NONE",
      "target": "NONE",
      "violation_type": "NONE",
      "label_quality": "verified",
      "source": "human",
      "notes": "Hand resting at rack baseline"
    },
    {
      "segment_id": "SEG-02",
      "start_frame": 41,
      "end_frame": 60,
      "start_time": 1.367,
      "end_time": 2.0,
      "verb": "APPROACH",
      "object": "RED_COMPONENT",
      "target": "NONE",
      "violation_type": "NONE",
      "label_quality": "verified",
      "source": "human",
      "notes": "Trajectory moving toward container"
    },
    {
      "segment_id": "SEG-03",
      "start_frame": 61,
      "end_frame": 70,
      "start_time": 2.033,
      "end_time": 2.333,
      "verb": "GRASP",
      "object": "RED_COMPONENT",
      "target": "NONE",
      "violation_type": "NONE",
      "label_quality": "verified",
      "source": "human",
      "notes": "Fingers close around red component"
    },
    {
      "segment_id": "SEG-04",
      "start_frame": 71,
      "end_frame": 110,
      "start_time": 2.367,
      "end_time": 3.667,
      "verb": "PICK",
      "object": "RED_COMPONENT",
      "target": "CONTAINER",
      "violation_type": "NONE",
      "label_quality": "verified",
      "source": "human",
      "notes": "Extraction from container"
    },
    {
      "segment_id": "SEG-05",
      "start_frame": 111,
      "end_frame": 180,
      "start_time": 3.7,
      "end_time": 6.0,
      "verb": "MOVE",
      "object": "RED_COMPONENT",
      "target": "TARGET_A",
      "violation_type": "NONE",
      "label_quality": "verified",
      "source": "human",
      "notes": "Transport across rack"
    },
    {
      "segment_id": "SEG-06",
      "start_frame": 181,
      "end_frame": 200,
      "start_time": 6.033,
      "end_time": 6.667,
      "verb": "PLACE",
      "object": "RED_COMPONENT",
      "target": "TARGET_A",
      "violation_type": "NONE",
      "label_quality": "verified",
      "source": "human",
      "notes": "Seating component into Target A"
    },
    {
      "segment_id": "SEG-07",
      "start_frame": 201,
      "end_frame": 215,
      "start_time": 6.7,
      "end_time": 7.167,
      "verb": "RELEASE",
      "object": "RED_COMPONENT",
      "target": "TARGET_A",
      "violation_type": "NONE",
      "label_quality": "verified",
      "source": "human",
      "notes": "Hand disengages and separates"
    },
    {
      "segment_id": "SEG-08",
      "start_frame": 216,
      "end_frame": 240,
      "start_time": 7.2,
      "end_time": 8.0,
      "verb": "IDLE",
      "object": "NONE",
      "target": "NONE",
      "violation_type": "NONE",
      "label_quality": "verified",
      "source": "human",
      "notes": "Hand returns to rest state"
    }
  ]
}
```
