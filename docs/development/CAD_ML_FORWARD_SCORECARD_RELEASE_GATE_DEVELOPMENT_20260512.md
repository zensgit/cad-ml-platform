# CAD ML Forward Scorecard Release Gate Development

Date: 2026-05-12

## Goal

Close the Phase 3 CI/release TODO by making the forward scorecard usable from
scheduled, PR, and release-labelled CI runs.

## Changes

- Added `scripts/ci/build_forward_scorecard_optional.sh`.
  - Follows the repository's existing optional CI wrapper pattern.
  - Accepts direct `FORWARD_SCORECARD_*` artifact variables.
  - Reuses existing benchmark outputs when available:
    - Graph2D blind summary.
    - History-sequence summary.
    - STEP/B-Rep summary.
    - Qdrant readiness.
    - Active-learning review queue report.
    - Benchmark knowledge readiness.
  - Emits GitHub outputs for overall and component statuses.
- Added `scripts/ci/check_forward_scorecard_release_gate.py`.
  - Reads the forward scorecard JSON.
  - Detects release labels from explicit labels, GitHub event payload labels, and tag refs.
  - Blocks release-labelled runs unless `overall_status` is
    `release_ready` or `benchmark_ready_with_gap`.
  - Writes a JSON gate report and GitHub output fields.
- Updated `.github/workflows/evaluation-report.yml`.
  - Added forward scorecard environment variables.
  - Added `Build forward scorecard release gate (optional)`.
  - Added `Upload forward scorecard`.
- Updated the Phase 3 TODO to mark the CI input wiring and release-label gate as done.

## Operating Model

Default workflow behavior remains conservative:

- If `FORWARD_SCORECARD_ENABLE` is false and no compatible benchmark artifacts exist,
  the step skips.
- If artifacts exist, the wrapper builds the scorecard without enabling the release gate.
- If `FORWARD_SCORECARD_RELEASE_GATE_ENABLE=true`, release-labelled runs are checked.
- If `FORWARD_SCORECARD_RELEASE_GATE_REQUIRE_RELEASE=true`, the gate applies even
  without a detected release label.

## Key Environment Variables

- `FORWARD_SCORECARD_ENABLE`
- `FORWARD_SCORECARD_MODEL_READINESS_JSON`
- `FORWARD_SCORECARD_HYBRID_SUMMARY_JSON`
- `FORWARD_SCORECARD_GRAPH2D_SUMMARY_JSON`
- `FORWARD_SCORECARD_HISTORY_SUMMARY_JSON`
- `FORWARD_SCORECARD_BREP_SUMMARY_JSON`
- `FORWARD_SCORECARD_QDRANT_SUMMARY_JSON`
- `FORWARD_SCORECARD_REVIEW_QUEUE_SUMMARY_JSON`
- `FORWARD_SCORECARD_KNOWLEDGE_SUMMARY_JSON`
- `FORWARD_SCORECARD_RELEASE_GATE_ENABLE`
- `FORWARD_SCORECARD_RELEASE_GATE_REQUIRE_RELEASE`
- `FORWARD_SCORECARD_RELEASE_LABELS`
- `FORWARD_SCORECARD_RELEASE_LABEL_PREFIXES`

## Follow-Up

- Feed real STEP/IGES strict benchmark metrics into
  `FORWARD_SCORECARD_BREP_SUMMARY_JSON`.
- Decide whether the broader benchmark release decision artifact should consume the
  forward scorecard as a first-class input in a later release-control slice.
