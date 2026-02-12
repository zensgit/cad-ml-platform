# DEV_GRAPH2D_ENSEMBLE_DEFAULT_MODELS_ALIGN_TO_PART_LABELS_20260212

## Summary
- Changed the Graph2D *ensemble* default model list to avoid drawing-type label-space mismatch.
- The default ensemble now points to the same part-label model used by the single-model Graph2D default.
- Updated `.env.example` to reflect the new recommended `GRAPH2D_MODEL_PATH` and corrected ensemble-default notes.

## Problem
Historically, the ensemble defaults in `src/ml/vision_2d.py` (and mirrored in health/provider checks) pointed to `graph2d_edge_sage_v3/v4`, which are **26-class drawing-type** models (labels include `零件图/机械制图/装配图/...`).

When operators enable `GRAPH2D_ENSEMBLE_ENABLED=true` without explicitly setting `GRAPH2D_ENSEMBLE_MODELS`, the runtime would load models that are not aligned with part-name classification.

## Change
Updated the default ensemble model-path list (used when `GRAPH2D_ENSEMBLE_MODELS` is empty) to:
- `models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth`

Implementation:
- `src/ml/vision_2d.py`: changed `EnsembleGraph2DClassifier` defaults
- `src/api/health_utils.py`: health visibility defaults match `src/ml/vision_2d.py`
- `src/core/providers/classifier.py`: provider health-check defaults match `src/ml/vision_2d.py`
- `.env.example`: updated `GRAPH2D_MODEL_PATH` example + corrected note about ensemble defaults

## Ops / Runtime Notes
- To run a true multi-model ensemble, set:

```bash
export GRAPH2D_ENSEMBLE_ENABLED=true
export GRAPH2D_ENSEMBLE_MODELS="models/<a>.pth,models/<b>.pth"
```

- Prefer ensembling models trained on the **same label-space**. The ensemble implementation can fall back to hard-voting when label maps mismatch, but that is not recommended for production decisions.

## Validation
- `make validate-core-fast` (passed)
