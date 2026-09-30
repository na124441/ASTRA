# SAS — Self-Aware Spacecrafts

## High-Level Design (HLD)

**Project:** Self-Aware Spacecrafts (SAS)  
**Version:** HLD v1.0  
**Target:** SIH 2026 — Space Technology  
**Architecture Style:** Edge-first, modular, hybrid statistical–probabilistic health management  

---

# 1. System Overview

SAS is an autonomous spacecraft health-management system designed to continuously monitor spacecraft telemetry, detect abnormal behavior, identify probable faults, estimate remaining useful life, and execute safety-validated responses.

The system follows a six-stage health-management pipeline:

```text
                    ┌──────────────────────┐
                    │ Spacecraft Telemetry │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Telemetry Data Engine│
                    │ Cleaning + Validation│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature / State      │
                    │ Estimation           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Anomaly Detection    │
                    │ SPC + Online PCA     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Fault Isolation &    │
                    │ Diagnosis            │
                    │ Bayesian Network     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Prognostics          │
                    │ Degradation + RUL    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Safety Policy Engine │
                    │ Rule Engine          │
                    └──────────┬───────────┘
                               │
                         Validated Action
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Autonomous Response  │
                    └──────────────────────┘
```

The system is intentionally designed as a **hybrid architecture** rather than relying exclusively on a deep-learning model.

---

# 2. Design Objectives

SAS is designed around the following objectives:

### 2.1 Continuous Health Monitoring

Monitor spacecraft telemetry continuously and maintain an evolving representation of spacecraft health.

### 2.2 Early Anomaly Detection

Detect deviations from normal spacecraft behavior before they develop into critical failures.

### 2.3 Fault Isolation

Determine which subsystem, component, or operational condition is most likely responsible for an anomaly.

### 2.4 Explainable Diagnosis

Provide interpretable evidence for why the system believes a fault exists.

### 2.5 Predictive Maintenance / Prognostics

Estimate degradation and Remaining Useful Life (RUL) where sufficient degradation evidence exists.

### 2.6 Autonomous Safety Response

Select corrective actions using deterministic safety rules and operational constraints.

### 2.7 Edge-First Operation

The core health-management pipeline should be capable of running locally/onboard with limited dependence on ground infrastructure.

### 2.8 Fault Tolerance

The system should continue operating when individual telemetry channels become unreliable or unavailable.

---

# 3. Architectural Principles

## 3.1 Edge-First

The critical detection, diagnosis, and safety-response path should execute locally.

```text
Onboard
───────
Telemetry
   ↓
Detection
   ↓
Diagnosis
   ↓
Prognostics
   ↓
Safety Validation
   ↓
Response
```

Ground systems primarily provide:

* visualization
* historical analysis
* model/configuration updates
* mission-level monitoring
* post-event analysis

---

## 3.2 Hybrid Intelligence

SAS combines several classes of algorithms.

| Layer                   | Primary Method        | Purpose                               |
| ----------------------- | --------------------- | ------------------------------------- |
| Signal monitoring       | SPC                   | Detect statistical deviations         |
| Multivariate monitoring | Online PCA            | Detect correlated subsystem anomalies |
| Diagnosis               | Bayesian Network      | Estimate probable fault states        |
| Prognostics             | Degradation/RUL model | Estimate remaining useful life        |
| Response                | Rule Engine           | Enforce safety constraints            |

This prevents a single model from becoming the sole point of failure.

---

## 3.3 Explainability

Every major system decision should have an interpretable basis.

For example:

```text
Anomaly Detected
       ↓
Temperature: +3.2σ
Current:      +2.7σ
Voltage:      -1.8σ
       ↓
PCA residual increased
       ↓
Subsystem: Power
       ↓
Fault probability:
Thermal fault     0.11
Battery fault     0.73
Sensor fault      0.16
       ↓
Estimated RUL: 184 cycles
       ↓
Safety Rule:
Reduce load + enter protected mode
```

The system therefore produces not just a decision, but also supporting evidence.

---

# 4. High-Level Component Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     SAS PLATFORM                            │
│                                                             │
│  ┌───────────────┐       ┌──────────────────────────────┐  │
│  │ Telemetry     │──────▶│ Telemetry Processing Engine  │  │
│  │ Source        │       └──────────────┬───────────────┘  │
│  └───────────────┘                      │                  │
│                                         ▼                  │
│                           ┌───────────────────────────┐    │
│                           │ Feature / State Engine    │    │
│                           └────────────┬──────────────┘    │
│                                        │                   │
│                         ┌──────────────▼──────────────┐    │
│                         │ Anomaly Detection Engine    │    │
│                         │ SPC + Online PCA            │    │
│                         └──────────────┬──────────────┘    │
│                                        │                   │
│                         ┌──────────────▼──────────────┐    │
│                         │ Fault Diagnosis Engine      │    │
│                         │ Bayesian Network            │    │
│                         └──────────────┬──────────────┘    │
│                                        │                   │
│                         ┌──────────────▼──────────────┐    │
│                         │ Prognostics Engine          │    │
│                         │ Degradation + RUL           │    │
│                         └──────────────┬──────────────┘    │
│                                        │                   │
│                         ┌──────────────▼──────────────┐    │
│                         │ Safety Policy Engine        │    │
│                         │ Rules + Constraints         │    │
│                         └──────────────┬──────────────┘    │
│                                        │                   │
│                         ┌──────────────▼──────────────┐    │
│                         │ Response Manager            │    │
│                         └──────────────┬──────────────┘    │
│                                        │                   │
│                                        ▼                   │
│                              Spacecraft Actions           │
└─────────────────────────────────────────────────────────────┘
```

---

# 5. Telemetry Acquisition Layer

The telemetry layer provides the raw operational state of the spacecraft.

Typical telemetry categories include:

### Power

* battery voltage
* battery current
* state of charge
* bus voltage
* solar-panel current
* power consumption

### Thermal

* component temperature
* battery temperature
* processor temperature
* payload temperature

### Attitude

* angular velocity
* attitude error
* reaction-wheel speed
* gyroscope measurements

### Communication

* signal strength
* packet loss
* link status
* transmission errors

### Computing

* CPU utilization
* memory utilization
* processor temperature
* watchdog status

### Structural / Mechanical

* vibration
* actuator state
* mechanism position

The telemetry source may be either:

```text
Real spacecraft telemetry
          OR
Synthetic / physics-based telemetry simulator
```

The simulator is particularly important during development because it allows controlled fault injection and degradation experiments.

---

# 6. Telemetry Processing Engine

Raw telemetry is first transformed into a validated internal representation.

### Responsibilities

1. Timestamp validation
2. Missing-value handling
3. Range validation
4. Noise filtering
5. Unit normalization
6. Sensor-health checks
7. Windowing
8. Feature generation

Example:

```text
Raw Telemetry
     │
     ├── Timestamp validation
     ├── Range validation
     ├── Missing-value detection
     ├── Filtering
     └── Normalization
             │
             ▼
      Clean Telemetry
```

The processing layer must preserve temporal ordering because spacecraft failures are inherently time-dependent.

---

# 7. Feature and State Estimation Layer

The feature engine transforms telemetry into information useful for health assessment.

Examples:

```text
Raw signal
    ↓
Rolling mean
Rolling variance
Rate of change
Moving residual
Temperature gradient
Power imbalance
Cross-sensor correlation
    ↓
Feature vector
```

For a spacecraft state at time $t$:

$$
x_t =
[x_1,x_2,\dots,x_n]^T
$$

where each $x_i$ represents a telemetry variable or derived feature.

The system maintains a temporal state:

$$
S_t = f(S_{t-1},x_t)
$$

This allows the health-management system to reason about both the current condition and recent history.

---

# 8. Anomaly Detection Engine

The anomaly detection layer contains two complementary mechanisms.

## 8.1 Statistical Process Control

SPC monitors individual telemetry variables against statistically learned operating limits.

For a monitored variable:

$$
z_t = \frac{x_t-\mu}{\sigma}
$$

where:

* $x_t$ = current measurement
* $\mu$ = expected mean
* $\sigma$ = standard deviation

An abnormal deviation can generate an anomaly event.

SPC is useful for:

* sensor drift
* sudden deviations
* abnormal temperature
* abnormal voltage/current
* persistent excursions

---

## 8.2 Online PCA

SPC is primarily univariate. Spacecraft systems, however, exhibit correlated behavior.

Online PCA therefore models relationships between multiple telemetry variables.

Given telemetry matrix:

$$
X \in \mathbb{R}^{T\times n}
$$

PCA identifies a lower-dimensional representation:

$$
X \approx TP^T
$$

where:

* $T$ = latent scores
* $P$ = principal components

The reconstruction residual:

$$
E = X-TP^T
$$

can be used to identify abnormal multivariate behavior.

This allows SAS to detect situations where individual telemetry values may appear normal but their **relationship with other variables has become abnormal**.

---

# 9. Anomaly Fusion

SPC and Online PCA outputs are combined into a unified anomaly representation.

```text
SPC Score ───────┐
                 │
                 ├──▶ Anomaly Fusion ──▶ Anomaly Score
PCA Residual ────┘
```

A conceptual anomaly score can be represented as:

$$
A_t =
w_s A_{SPC}
+
w_p A_{PCA}
$$

where:

* $A_{SPC}$ = statistical anomaly score
* $A_{PCA}$ = multivariate anomaly score
* $w_s,w_p$ = calibrated weights

The resulting score is mapped into operational states such as:

```text
NORMAL
   ↓
WARNING
   ↓
ANOMALOUS
   ↓
CRITICAL
```

Thresholds should be configurable rather than hard-coded into the system.

---

# 10. Fault Isolation and Diagnosis

Once an anomaly is detected, SAS attempts to identify its probable source.

The diagnosis engine uses a Bayesian Network.

## 10.1 Bayesian Network Structure

A conceptual dependency graph:

```text
Telemetry
    │
    ▼
Sensor Health
    │
    ▼
Subsystem State
    │
    ▼
Fault Probability
    │
    ▼
Failure Mode
    │
    ▼
Prognostic State
```

For example:

```text
Battery Temperature
Battery Voltage
Battery Current
       │
       ▼
Battery Health
       │
       ▼
Battery Fault
       │
       ▼
Thermal Runaway Risk
```

The Bayesian network estimates:

$$
P(F_i|E)
$$

where:

* $F_i$ = candidate fault
* $E$ = observed evidence

Using Bayes' theorem:

$$
P(F_i|E)
=
\frac{P(E|F_i)P(F_i)}
{P(E)}
$$

The system can therefore maintain probabilities across multiple candidate failure modes rather than immediately assigning a single deterministic diagnosis.

---

# 11. Fault Classification

The diagnosis engine produces a structured diagnosis object.

Example:

```json
{
  "subsystem": "Power",
  "fault": "Battery degradation",
  "probability": 0.82,
  "severity": "HIGH",
  "evidence": [
    "battery temperature increase",
    "voltage instability",
    "abnormal current profile"
  ]
}
```

This object becomes the input to the prognostics and safety layers.

---

# 12. Prognostics Engine

The prognostics subsystem estimates degradation and Remaining Useful Life.

The basic pipeline is:

```text
Historical Telemetry
        ↓
Fault / Degradation Detection
        ↓
Degradation State
        ↓
Trend Estimation
        ↓
Failure Threshold
        ↓
RUL Estimate
```

Remaining Useful Life is represented as:

$$
RUL_t = T_{failure}-T_t
$$

where $T_{failure}$ is the estimated time at which the component crosses its defined failure threshold.

---

# 13. Degradation Modeling

The RUL subsystem should not depend exclusively on randomly generated failure labels.

The prototype should use:

### 13.1 Physics-Informed / Synthetic Degradation

Generate controlled degradation trajectories representing realistic component behavior.

Example:

$$
y_t = y_0 - kt + \epsilon_t
$$

where:

* $y_0$ = initial health state
* $k$ = degradation rate
* $t$ = operating time
* $\epsilon_t$ = noise

More complex degradation functions can be introduced for nonlinear behavior.

---

## 13.2 Fault Injection

Fault scenarios should be explicitly generated.

Examples:

```text
Sensor drift
Sensor bias
Battery degradation
Thermal excursion
Power instability
Communication degradation
Reaction-wheel anomaly
```

Fault injection allows the system to evaluate whether the complete pipeline correctly moves from:

```text
Healthy
  ↓
Degrading
  ↓
Anomalous
  ↓
Diagnosed
  ↓
Critical
  ↓
Response
```

---

# 14. Safety Policy Engine

The safety policy engine is the final authority before an autonomous action is executed.

It uses deterministic rules and safety constraints.

Example:

```text
IF
    battery_temperature > threshold
AND
    battery_fault_probability > threshold
THEN
    reduce_power_load
    enter_protected_mode
```

Another example:

```text
IF
    communication_health = CRITICAL
AND
    spacecraft_health = STABLE
THEN
    switch_to_backup_communication
```

The system must distinguish between:

```text
Recommended Action
```

and

```text
Validated Autonomous Action
```

Only actions satisfying safety constraints may be executed automatically.

---

# 15. Response Manager

The response manager converts validated policies into spacecraft actions.

Example:

```text
Diagnosis
    │
    ▼
Battery degradation
    │
    ▼
Safety Policy
    │
    ├── Reduce non-critical load
    ├── Disable unnecessary payload
    └── Enter protected operating state
```

Every action should produce an event record:

```json
{
  "timestamp": "...",
  "fault": "battery_degradation",
  "action": "reduce_noncritical_load",
  "reason": "...",
  "policy": "POWER_SAFE_03",
  "confidence": 0.82
}
```

This creates an auditable autonomous decision trail.

---

# 16. Spacecraft Health State

SAS maintains a continuously updated spacecraft health state.

Conceptually:

```text
┌─────────────────────────────────────┐
│         SPACECRAFT HEALTH           │
├─────────────────────────────────────┤
│ Overall Health:       DEGRADED      │
│ Anomaly Score:        0.71          │
│ Critical Faults:      1             │
│ Active Warnings:      3             │
│ Highest Fault Prob.:  0.82          │
│ Estimated RUL:        184 cycles    │
│ Current Mode:         PROTECTED     │
└─────────────────────────────────────┘
```

The health state is generated from the combined outputs of detection, diagnosis, and prognostics.

---

# 17. Data Flow

The complete data flow is:

```text
Telemetry
   │
   ▼
Validation
   │
   ▼
Preprocessing
   │
   ▼
Feature Extraction
   │
   ▼
┌──────────────────────┐
│ Anomaly Detection    │
│                      │
│ SPC + Online PCA     │
└──────────┬───────────┘
           │
           ▼
      Anomaly Event
           │
           ▼
┌──────────────────────┐
│ Fault Diagnosis      │
│ Bayesian Network     │
└──────────┬───────────┘
           │
           ▼
      Fault State
           │
           ▼
┌──────────────────────┐
│ Prognostics          │
│ Degradation + RUL    │
└──────────┬───────────┘
           │
           ▼
     Prognostic State
           │
           ▼
┌──────────────────────┐
│ Safety Policy Engine │
│ Rule Engine          │
└──────────┬───────────┘
           │
           ▼
     Validated Action
           │
           ▼
┌──────────────────────┐
│ Response Manager     │
└──────────┬───────────┘
           │
           ▼
   Spacecraft Response
```

---

# 18. Module Boundaries

The implementation should maintain clear module boundaries.

```text
sas/
│
├── telemetry/
│   ├── simulator
│   ├── ingestion
│   └── validation
│
├── preprocessing/
│   ├── filtering
│   ├── normalization
│   └── windowing
│
├── features/
│   ├── statistical
│   ├── temporal
│   └── subsystem
│
├── detection/
│   ├── spc
│   ├── online_pca
│   └── fusion
│
├── diagnosis/
│   ├── bayesian_network
│   ├── fault_models
│   └── evidence
│
├── prognostics/
│   ├── degradation
│   ├── rul
│   └── failure_models
│
├── safety/
│   ├── rules
│   ├── constraints
│   └── validator
│
├── response/
│   ├── actions
│   └── state_manager
│
├── simulation/
│   ├── fault_injection
│   └── scenarios
│
├── evaluation/
│   ├── detection_metrics
│   ├── diagnosis_metrics
│   ├── rul_metrics
│   └── latency
│
└── dashboard/
    ├── telemetry
    ├── health
    ├── faults
    └── events
```

The exact implementation language/framework can evolve without changing the logical architecture.

---

# 19. Simulation and Fault Injection Architecture

Because real spacecraft telemetry and failure events are limited, SAS requires a controlled simulation environment.

```text
┌──────────────────────────┐
│ Spacecraft State Model   │
└────────────┬─────────────┘
             │
             ▼
      Normal Telemetry
             │
             ▼
┌──────────────────────────┐
│ Fault Injection Engine   │
├──────────────────────────┤
│ Sensor faults            │
│ Component degradation    │
│ Thermal faults           │
│ Power faults             │
│ Communication faults     │
└────────────┬─────────────┘
             │
             ▼
       Faulty Telemetry
             │
             ▼
         SAS Pipeline
```

Each scenario should have:

* initial state
* fault type
* fault injection time
* degradation profile
* expected detection point
* expected diagnosis
* expected response
* failure threshold

This creates reproducible evaluation experiments.

---

# 20. Evaluation Architecture

SAS should evaluate every major stage independently and end-to-end.

## Detection

Metrics:

* Precision
* Recall
* F1-score
* False Alarm Rate
* Detection latency

## Diagnosis

Metrics:

* Fault classification accuracy
* Top-k diagnosis accuracy
* Bayesian probability calibration
* Diagnosis latency

## Prognostics

Metrics:

* MAE
* RMSE
* RUL error
* Prediction horizon
* Calibration / uncertainty where applicable

## Response

Metrics:

* Correct action rate
* Unsafe action rate
* Response latency
* Policy violation count

## System

Metrics:

* End-to-end latency
* CPU utilization
* Memory consumption
* Throughput
* Fault recovery time

---

# 21. End-to-End Execution Example

Consider a battery degradation scenario.

### Step 1 — Normal State

```text
Battery temperature = normal
Battery voltage     = stable
Battery current     = stable

Health = NORMAL
```

### Step 2 — Degradation Begins

Temperature slowly increases while voltage stability decreases.

```text
SPC → Warning
Online PCA → Increasing residual
```

### Step 3 — Anomaly Detection

The combined anomaly score crosses the warning threshold.

```text
Health = ANOMALOUS
```

### Step 4 — Diagnosis

The Bayesian network evaluates candidate faults.

```text
Battery degradation = 0.82
Sensor fault        = 0.11
Thermal subsystem   = 0.07
```

### Step 5 — Prognostics

The degradation model estimates:

```text
RUL ≈ 184 operating cycles
```

### Step 6 — Safety Policy

The rule engine evaluates the situation.

```text
Fault severity = HIGH
RUL = decreasing
Battery temperature = increasing
```

The corresponding safety policy is selected.

### Step 7 — Response

The spacecraft reduces non-critical power consumption and enters a protected operating mode.

### Step 8 — Logging

The entire chain is recorded:

```text
Telemetry
→ Anomaly
→ Evidence
→ Diagnosis
→ RUL
→ Policy
→ Action
```

This produces a complete autonomous reasoning trace.

---

# 22. Ground Segment / Dashboard

The ground-side interface should provide operational visibility without being part of the critical onboard control path.

The dashboard should expose:

### Spacecraft Overview

* overall health
* operational mode
* active faults
* anomaly score

### Telemetry

* real-time telemetry
* historical telemetry
* detected deviations

### Diagnosis

* probable faults
* probabilities
* evidence

### Prognostics

* degradation trend
* RUL
* failure threshold

### Autonomous Actions

* action history
* policy triggered
* action reason
* timestamp

The dashboard is therefore an **observability layer**, not the decision-making core.

---

# 23. Reliability and Safety

SAS must follow a fail-safe philosophy.

### Principle 1 — Detection Does Not Automatically Mean Action

An anomaly should not directly trigger a spacecraft command.

```text
Anomaly
   ↓
Diagnosis
   ↓
Safety Validation
   ↓
Action
```

### Principle 2 — Safety Rules Override Probabilistic Recommendations

Bayesian inference can suggest a fault, but the safety layer determines whether an action is permitted.

### Principle 3 — Unknown State Should Not Produce Aggressive Actions

If confidence is insufficient, the system should transition into a monitored/degraded state rather than execute an uncertain corrective action.

### Principle 4 — Every Autonomous Action Must Be Auditable

The system must retain:

```text
Input
→ Decision
→ Evidence
→ Policy
→ Action
```

---

# 24. Deployment Model

The architecture is divided into an onboard-critical path and a ground-support path.

```text
                 SPACECRAFT
┌────────────────────────────────────────────┐
│                                            │
│ Telemetry                                  │
│    ↓                                       │
│ Detection                                  │
│    ↓                                       │
│ Diagnosis                                  │
│    ↓                                       │
│ Prognostics                                │
│    ↓                                       │
│ Safety Engine                              │
│    ↓                                       │
│ Response                                   │
│                                            │
└────────────────────┬───────────────────────┘
                     │
                     │ Telemetry / Events
                     ▼
             ┌─────────────────┐
             │ Ground Segment  │
             │                 │
             │ Dashboard       │
             │ Analytics       │
             │ Logs            │
             │ Model Analysis  │
             └─────────────────┘
```

The spacecraft should remain capable of performing critical health-management functions even when communication with the ground segment is unavailable.

---

# 25. Technology-Agnostic Architecture

SAS should not tightly couple the system architecture to a particular ML framework.

The logical interfaces should remain stable:

```text
TelemetryProvider
       ↓
FeatureExtractor
       ↓
AnomalyDetector
       ↓
FaultDiagnoser
       ↓
PrognosticsEngine
       ↓
SafetyPolicy
       ↓
ResponseManager
```

This allows individual algorithms to be replaced without redesigning the entire system.

For example:

```text
SPC
 ↓
Alternative statistical detector
```

or:

```text
Bayesian Network
 ↓
Alternative probabilistic diagnosis model
```

without changing the surrounding system contracts.

---

# 26. Core System State

At any point in time, SAS maintains a unified health state:

$$
H_t =
\{
S_t,
A_t,
F_t,
D_t,
R_t,
M_t
\}
$$

where:

* $S_t$ = spacecraft state
* $A_t$ = anomaly state
* $F_t$ = fault probabilities
* $D_t$ = degradation state
* $R_t$ = RUL estimate
* $M_t$ = operational mode

This state becomes the central object exchanged between the major SAS modules.

---

# 27. Architectural Sequence

The complete operational sequence is:

```text
             ┌──────────────┐
             │   Telemetry  │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │ Preprocessing│
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │   Features   │
             └──────┬───────┘
                    ↓
             ┌──────────────┐
             │   Detection  │
             └──────┬───────┘
                    ↓
              Anomaly?
             /          \
           No            Yes
           │              │
           │              ▼
           │        ┌──────────────┐
           │        │   Diagnosis  │
           │        └──────┬───────┘
           │               ↓
           │        ┌──────────────┐
           │        │ Prognostics  │
           │        └──────┬───────┘
           │               ↓
           │        ┌──────────────┐
           │        │ Safety Policy│
           │        └──────┬───────┘
           │               ↓
           │        ┌──────────────┐
           │        │   Response   │
           │        └──────┬───────┘
           │               │
           └───────┬───────┘
                   ▼
            Health State
                   │
          ┌────────┴────────┐
          ▼                 ▼
       Dashboard          Logging
```

---

# 28. Non-Functional Requirements

| Requirement         | Target                              |
| ------------------- | ----------------------------------- |
| Execution           | Edge/onboard capable                |
| Detection           | Low-latency streaming               |
| Explainability      | Required                            |
| Fault tolerance     | Required                            |
| Safety validation   | Required before autonomous action   |
| Reproducibility     | Required for simulation             |
| Auditability        | Required                            |
| Modularity          | Required                            |
| Hardware dependency | Minimized                           |
| Ground connectivity | Non-essential for critical response |

---

# 29. Key Architectural Differentiator

The primary architectural distinction of SAS is that it does not stop at anomaly detection.

Traditional monitoring:

```text
Telemetry
   ↓
Anomaly
   ↓
Alert
```

SAS:

```text
Telemetry
   ↓
Detect
   ↓
Isolate
   ↓
Diagnose
   ↓
Prognose
   ↓
Validate
   ↓
Respond
   ↓
Observe Result
```

This creates a closed-loop spacecraft health-management architecture.

---

# 30. Final Architecture

The complete SAS architecture can therefore be summarized as:

```text
                    ┌─────────────────────┐
                    │ SPACECRAFT TELEMETRY│
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ DATA ENGINE         │
                    │ Validate / Filter   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ STATE / FEATURES    │
                    └──────────┬──────────┘
                               ↓
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
       ┌─────────────┐                   ┌─────────────┐
       │     SPC     │                   │ Online PCA  │
       └──────┬──────┘                   └──────┬──────┘
              │                                 │
              └──────────────┬──────────────────┘
                             ▼
                    ┌─────────────────────┐
                    │ ANOMALY FUSION      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ BAYESIAN DIAGNOSIS  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ DEGRADATION / RUL   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ SAFETY RULE ENGINE  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ RESPONSE MANAGER    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ SPACECRAFT ACTION   │
                    └──────────┬──────────┘
                               │
                         Feedback Loop
                               │
                               └───────────────►
```

## Architectural Principle

> **SAS transforms spacecraft telemetry from a passive monitoring stream into an autonomous health-management loop: detect the abnormal state, understand its probable cause, estimate how the condition will evolve, validate a safe response, and act while maintaining an auditable explanation of every decision.**
