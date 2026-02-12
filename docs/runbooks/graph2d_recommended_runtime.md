# Graph2D Recommended Runtime

## Recommended Settings (2026-02-12)
These settings keep Graph2D contributions conservative while still allowing
geometry-only fallback in anonymous DXF scenarios.

```
GRAPH2D_ENABLED=true
GRAPH2D_MODEL_PATH=models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth
# Prefer calibration file to keep the chosen objective/version explicit.
GRAPH2D_TEMPERATURE_CALIBRATION_PATH=models/calibration/graph2d_training_dxf_oda_titleblock_distill_20260210_temperature_20260210.json
# Optional ambiguity guardrail (recommended for many-class checkpoints).
GRAPH2D_MIN_MARGIN=0.01
GRAPH2D_EXCLUDE_LABELS=other
GRAPH2D_ALLOW_LABELS=
GRAPH2D_FUSION_ENABLED=true
FUSION_ANALYZER_ENABLED=true
FUSION_ANALYZER_OVERRIDE=true
FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.6
```

## Notes
- If you enable `GRAPH2D_ENSEMBLE_ENABLED=true` without setting `GRAPH2D_ENSEMBLE_MODELS`,
  the runtime uses a conservative part-label default (see `src/ml/vision_2d.py`).
- Expand `GRAPH2D_ALLOW_LABELS` only after re-validating on a fixed sample set.
