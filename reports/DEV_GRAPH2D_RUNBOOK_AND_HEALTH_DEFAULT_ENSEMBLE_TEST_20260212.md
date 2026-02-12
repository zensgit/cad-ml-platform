# DEV_GRAPH2D_RUNBOOK_AND_HEALTH_DEFAULT_ENSEMBLE_TEST_20260212

## Goal
Lock in the new Graph2D model/temperature recommendations operationally, and prevent drift between:
- Graph2D runtime ensemble defaults (`src/ml/vision_2d.py`)
- `/health` visibility (`src/api/health_utils.py`)

## Changes
### 1) Runbook: recommended Graph2D runtime
- Updated `docs/runbooks/graph2d_recommended_runtime.md`:
  - Model: `models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth`
  - Calibration: `models/calibration/graph2d_training_dxf_oda_titleblock_distill_20260210_temperature_20260210.json` (`temperature=0.5`)
  - Added `GRAPH2D_MIN_MARGIN=0.01` guardrail recommendation
  - Clarified ensemble default behavior when `GRAPH2D_ENSEMBLE_MODELS` is unset

### 2) Regression test: health reflects ensemble defaults
- Added unit coverage in `tests/unit/test_health_utils_coverage.py`:
  - When `GRAPH2D_ENSEMBLE_ENABLED=true` and `GRAPH2D_ENSEMBLE_MODELS` is unset, the health payload should report:
    - `graph2d_ensemble_models_configured == 1`
    - `graph2d_ensemble_models == ["graph2d_training_dxf_oda_titleblock_distill_20260210.pth"]`

This prevents future changes from silently reverting defaults back to drawing-type ensemble models.

## Validation
- `make validate-core-fast` (passed)
