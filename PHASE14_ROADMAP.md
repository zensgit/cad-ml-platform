# Phase 14 Roadmap: Active Learning Loop

**Status**: In Progress
**Start Date**: 2025-12-06
**Focus**: Uncertainty Sampling, Human Feedback, Model Fine-tuning

## 1. Objectives
- Activate uncertainty sampling in the classification pipeline.
- Enable human feedback collection via API and UI.
- Implement automated model retraining triggers.

## 2. Implementation Plan

### Week 3: Active Learning Activation
- [x] **14.1 Core Infrastructure**
  - [x] `src/core/active_learning.py` (Manager, Sample models).
  - [x] `src/api/v1/active_learning.py` (API Endpoints).
  - [x] `examples/labeling_ui.html` (Labeling Interface).
- [x] **14.2 Integration**
  - [x] Modify `src/api/v1/analyze.py` to flag uncertain predictions.
  - [x] Ensure `doc_id` is passed correctly.
- [x] **14.3 Feedback Loop**
  - [x] Test feedback submission (`tests/unit/test_active_learning_loop.py`).
  - [x] Verify adaptive weight updates (via retraining trigger).

### Week 4: Automation
- [x] **14.4 Retraining Pipeline**
  - [x] Create `scripts/finetune_from_feedback.py`.
  - [x] Implement model versioning (Timestamp-based).

## 3. Progress Log
- **2025-12-06**: Infrastructure verified. Starting integration.
- **2025-12-06**: Completed feedback loop tests and fine-tuning script. Phase 14 Complete.
