# DEV_ANALYZE_DXF_HYBRID_DRAWING_TYPE_OVERRIDE_20260207

## Goal
Prevent DXF classification from returning drawing-type labels (for example `机械制图`, `装配图`) as the final `part_type` when `HybridClassifier` can confidently infer a concrete part label (typically from filename).

This addresses cases where the L2 fusion path (`rule_version=L2-Fusion-v1`) may return a drawing-type label and block the existing Hybrid auto-override logic (which previously only covered placeholder buckets and low-confidence `rules` outputs).

## Changes
- `src/api/v1/analyze.py`
  - Extended Hybrid auto-override to also apply when the current `part_type` is a drawing-type label (same label set used to ignore Graph2D drawing-type outputs).
  - New override mode: `mode=auto_drawing_type`.
  - Existing modes remain:
    - `auto` (placeholder `rules` bucket, `rule_version=v1`)
    - `auto_low_conf` (low-confidence base `rules` output, controlled by `HYBRID_OVERRIDE_BASE_MAX_CONF`)
    - `env` (forced by `HYBRID_CLASSIFIER_OVERRIDE=true`)

## Verification
- Ran:
  - `.venv/bin/pytest -q tests/integration/test_analyze_dxf_hybrid_override.py`
- Result:
  - `3 passed`

## Notes
- To disable auto behavior: `HYBRID_CLASSIFIER_AUTO_OVERRIDE=false`.
- Thresholds:
  - `HYBRID_OVERRIDE_MIN_CONF` (default `0.8`)
  - `HYBRID_OVERRIDE_BASE_MAX_CONF` (default `0.7`, used by `auto_low_conf`)
