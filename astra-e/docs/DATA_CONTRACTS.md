# 🛰️ ASTRA-E Data Contracts & Pipeline Architecture

See the master project documentation:
👉 [Root Data Pipeline Architecture Specification](../../docs/data_pipeline.md)

---

## Key Contracts Summary

- **Kinematic Feature Dimension**: `26-D` normalized physical kinematics.
- **Window Size**: `30` frames ($1.0$ s at $30$ FPS).
- **Multi-Head Prediction**:
  - `verb`: `VERB_VOCAB` (11 classes)
  - `object`: `OBJECT_VOCAB` (5 classes)
  - `target`: `TARGET_VOCAB` (5 classes)
- **Violation Policy**: Symbolic downstream evaluation in `ProcedureGraph`; excluded from temporal neural output heads.
- **Partitioning Policy**: Group-level disjoint splitting (`subject_id` or `run_id`). Zero random window leakage.
- **Physical Format**: Memory-mapped binary `.npy` arrays with zero-copy loading.
