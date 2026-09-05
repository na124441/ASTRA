# Problem Statement 
AI-Based Human Activity and Experiment Procedure Recognition System for On-Board BAS Experiments
## 1. Background
Future Indian human-spaceflight missions and the Bharatiya Antariksh Station (BAS) are expected to support scientific experiments in a microgravity environment. During such missions, communication between astronauts and ground control may experience significant latency and limited bandwidth, making continuous real-time supervision from Earth impractical.
Scientific experiments conducted onboard therefore require a degree of autonomous monitoring and procedural assistance.
A fixed-camera-based AI system can locally observe an astronaut performing an experiment, identify relevant human activities and interactions with experimental objects, determine the current stage of the experiment, and verify whether the prescribed sequence is being followed.
## 2. Problem
The proposed system addresses the problem of automatically monitoring and validating astronaut execution of predefined scientific experiment procedures using onboard camera feeds.
Given video captured by one or more fixed cameras observing an experimental payload, the system shall:
identify the astronaut and relevant experimental objects;
recognize predefined human activities and human–object interactions;
determine the current experiment step from temporal video observations;
compare the observed activity sequence against the prescribed experiment procedure;
detect skipped, repeated, incorrect, or out-of-order steps;
provide the astronaut with appropriate next-step guidance and warnings;
maintain a lightweight, timestamped record of experiment execution; and
operate locally without requiring continuous communication with ground control.
The system should additionally be designed to handle the distinctive visual and spatial characteristics of microgravity, where conventional gravity-dependent assumptions about human orientation and posture may not hold.
## 3. Proposed System
We propose an AI-driven onboard Experiment Activity Recognition and Procedure Validation System.
At a high level:
             FIXED PAYLOAD CAMERA
                       │
                       ▼
              ┌─────────────────┐
              │ Video Processing│
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Human         Object       Hand/Object
       Analysis      Analysis     Interaction
          │            │            │
          └────────────┼────────────┘
                       ▼
              Temporal Activity
                 Recognition
                       │
                       ▼
              Procedure Engine
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Correct       Violation    Uncertain
       Step          Detection    Observation
          │            │            │
          └────────────┼────────────┘
                       ▼
             Guidance / Alert System
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Voice Guidance       Experiment Log
The system will separate perception from procedure reasoning. AI models will determine what is happening, while a procedure model/state machine will determine whether it should be happening at that point in the experiment.
## 4. Core Objectives
The system shall achieve the following objectives:
O1 — Human Activity Recognition
Recognize predefined astronaut activities relevant to the experiment from video.
O2 — Object Recognition
Detect and track experimental objects, tools, containers, and relevant target regions.
O3 — Human–Object Interaction Recognition
Determine interactions such as:
reaching;
grasping;
picking;
moving;
placing;
releasing;
manipulating experimental equipment.
O4 — Temporal Understanding
Use a sequence of observations rather than individual frames to determine the current experiment action.
O5 — Procedure Validation
Represent the experiment as an ordered sequence of states/actions and validate observed execution against it.
O6 — Error Detection
Detect events including:
Correct execution
      │
      ├── Skipped step
      ├── Out-of-order step
      ├── Repeated step
      ├── Incorrect action
      ├── Incorrect object
      └── Uncertain/ambiguous action
O7 — Astronaut Assistance
Provide contextual next-step instructions and warnings through an onboard interface and voice output.
O8 — Experiment Logging
Generate a lightweight timestamped record of experiment execution, including detected actions, procedure state, confidence and violations.
O9 — Edge/Offline Operation
Perform the core AI inference and procedure validation locally without depending on continuous Internet or ground communication.
O10 — Microgravity Robustness
Investigate and support orientation-independent activity recognition, preferably using the experimental payload/rack as the spatial reference frame rather than assuming a fixed gravitational "up."
## 5. System Scope
### In Scope
The system will cover:
fixed-camera video acquisition;
video preprocessing;
astronaut detection/tracking;
relevant object detection;
human pose estimation;
hand/object interaction analysis;
temporal activity recognition;
experiment-state estimation;
procedure validation;
anomaly/violation detection;
next-step recommendation;
voice alerts;
experiment logging;
local video storage;
local monitoring dashboard;
optional video streaming over a local/network interface;
offline inference;
performance monitoring.
### Out of Scope
The system will not attempt to:
autonomously control spacecraft systems;
physically manipulate experimental equipment;
replace astronaut decision-making;
replace ground mission control;
perform unrestricted/general-purpose understanding of every possible astronaut activity;
make safety-critical spacecraft decisions without human authorization.
This distinction is important: our system is an experiment-assistance and monitoring system, not an autonomous spacecraft controller.
## 6. Operating Assumptions
The initial system will assume:
The experiment procedure is predefined.
The relevant actions and objects are known.
Cameras are installed at fixed locations.
The camera view sufficiently covers the experimental workspace.
A representative dataset can be generated for the target experiment.
The system has access to local compute hardware capable of running the trained models.
The experiment can be represented as a finite sequence/state graph.
For example:
START
  ↓
OPEN_CONTAINER
  ↓
PICK_RED_OBJECT
  ↓
PLACE_RED_OBJECT
  ↓
PICK_YELLOW_OBJECT
  ↓
PLACE_YELLOW_OBJECT
  ↓
CLOSE_CONTAINER
  ↓
COMPLETE
## 7. Key System Principle
The fundamental design principle is:
The AI should not merely recognize what the astronaut is doing; it should understand where that action occurs within the prescribed experiment procedure.
For example:
AI perception:
"Close container"        confidence = 0.94
is insufficient by itself.
The procedure engine additionally evaluates:
Current experiment state:
PLACE_YELLOW
Expected:
PLACE_YELLOW
Observed:
CLOSE_CONTAINER
Result:
OUT-OF-SEQUENCE
Therefore:
System Decision=Activity Recognition+Temporal Context+Procedure State\text{System Decision} = \text{Activity Recognition} + \text{Temporal Context} + \text{Procedure State}
This is the core differentiator of our system.
## 8. Functional Requirements — High Level
The system shall provide the following major functions:
ID|Function
FR-01|Acquire video from fixed cameras
FR-02|Process video locally
FR-03|Detect and track astronaut
FR-04|Detect relevant experiment objects
FR-05|Estimate human pose
FR-06|Analyze hand-object interactions
FR-07|Recognize predefined activities
FR-08|Maintain temporal activity context
FR-09|Maintain current experiment state
FR-10|Validate action sequence
FR-11|Detect skipped steps
FR-12|Detect out-of-order steps
FR-13|Detect repeated/incorrect actions
FR-14|Determine next expected action
FR-15|Generate astronaut alerts
FR-16|Provide voice guidance
FR-17|Generate timestamped experiment logs
FR-18|Store video locally
FR-19|Provide monitoring GUI
FR-20|Operate without continuous Internet connectivity
We'll expand these considerably in the SRS.
## 9. Non-Functional Requirements
### Performance
The system should support near-real-time inference with an explicitly measured:
FPS;
end-to-end latency;
action recognition latency;
alert latency.
### Reliability
The system should avoid generating alerts from a single uncertain observation.
For example:
Frame 1 → PICK_RED 0.51
Frame 2 → PICK_RED 0.53
Frame 3 → PICK_RED 0.48
should not immediately trigger a procedure transition.
Temporal confidence and state confirmation should be used.
### Offline Capability
Core functionality must remain operational without Internet connectivity.
### Resource Efficiency
The system should be optimized for constrained onboard/edge computing environments.
### Explainability
Every procedure decision should ideally be traceable to:
Observed action
+
Confidence
+
Current state
+
Expected action
+
Decision
For example:
Current State : STEP_04
Expected      : PICK_YELLOW
Observed      : CLOSE_CONTAINER
Confidence    : 91.4%
Decision      : OUT_OF_SEQUENCE
### Extensibility
The procedure engine should allow a new experiment to be introduced through a new experiment definition rather than requiring the entire AI pipeline to be redesigned.
## 10. Experiment-Agnostic Architecture
This is something I'd explicitly put into our SRS.
We shouldn't hard-code:
RED → YELLOW → CLOSE
into the AI.
Instead, define an experiment configuration:
experiment:
  id: EXP_001
  name: Sample BAS Experiment
steps:
  - id: S01
    action: OPEN_CONTAINER
  - id: S02
    action: PICK_OBJECT
    object: RED
  - id: S03
    action: PLACE_OBJECT
    object: RED
    target: TARGET_A
  - id: S04
    action: PICK_OBJECT
    object: YELLOW
  - id: S05
    action: PLACE_OBJECT
    object: YELLOW
    target: TARGET_B
  - id: S06
    action: CLOSE_CONTAINER
Then:
                Experiment Definition
                         │
                         ▼
                  Procedure Engine
                         ▲
                         │
                  AI Observation
This transforms the project from a single SIH demo into a potentially reusable experiment monitoring platform.
## 11. Microgravity Extension
The system should optionally support a rack-relative spatial representation.
Instead of assuming:
Y=gravity-upY = \text{gravity-up}
we define:
Coordinate Frame=Payload RackCoordinate\ Frame = Payload\ Rack
Then human pose and object locations can be interpreted relative to the experiment.
Conceptually:
            PAYLOAD RACK
        ┌─────────────────────┐
        │                     │
        │       Astronaut     │
        │          ↗          │
        │                     │
        │   Experiment       │
        │   Workspace        │
        └─────────────────────┘
This makes the system more robust to:
body rotation;
inverted posture;
floating;
different astronaut orientations.
This should be treated as an advanced capability, not a dependency for the MVP.
## 12. Success Criteria
The system will be considered successful when it can demonstrate that it:
Recognition
Reliably recognizes the predefined experiment actions from camera footage.
Procedure Understanding
Correctly determines the current stage of the experiment.
Violation Detection
Detects deliberately introduced skipped and out-of-order steps.
Assistance
Provides the correct next-step guidance and appropriate warnings.
Logging
Produces an accurate timestamped experiment execution log.
Edge Operation
Performs the above pipeline locally without requiring cloud inference.
Robustness
Maintains acceptable recognition performance under variations in astronaut appearance, execution speed, camera conditions and orientation.
## 13. One-line Problem Statement
For the SRS cover/abstract, I'd condense the entire thing to:
To develop an offline, edge-deployable AI system that uses fixed-camera video to recognize astronaut activities and human–object interactions during predefined BAS scientific experiments, understand their temporal sequence, validate execution against the prescribed procedure, detect procedural deviations, and provide real-time guidance, alerts, and lightweight experiment logs.
That's the version I'd use as our official internal problem statement.
And now we can move to the SRS
I recommend we freeze this problem statement first, then structure the SRS as:
SRS
│
├── 1. Introduction
│   ├── Purpose
│   ├── Scope
│   ├── Definitions
│   └── References
│
├── 2. Overall Description
│   ├── System Context
│   ├── Users
│   ├── Operating Environment
│   ├── Constraints
│   └── Assumptions
│
├── 3. Functional Requirements
│   ├── Video Acquisition
│   ├── Human Detection
│   ├── Object Detection
│   ├── Pose
│   ├── HOI
│   ├── HAR
│   ├── Temporal Reasoning
│   ├── Procedure Engine
│   ├── Violation Detection
│   ├── Guidance
│   ├── Logging
│   └── Monitoring
│
├── 4. Non-Functional Requirements
│
├── 5. AI/ML Requirements
│   ├── Dataset
│   ├── Annotation
│   ├── Training
│   ├── Evaluation
│   └── Model Deployment
│
├── 6. System Architecture
│
├── 7. Data Requirements
│
├── 8. Interfaces
│
├── 9. Security & Reliability
│
├── 10. Performance Requirements
│
└── 11. Acceptance Criteria
One important architectural decision I'd lock now: make the Procedure Engine experiment-agnostic and deterministic, while the ML layer remains responsible for perception. That gives us a clean separation between “what did the astronaut do?” and “was that the correct thing to do now?” — and it will make the eventual SRS, implementation, testing, and SIH presentation substantially stronger. 🚀