# Graph2D Recommended Runtime

## Recommended Settings (2026-01-23)
These settings keep Graph2D contributions conservative while still allowing
high-confidence overrides for specific labels.

```
GRAPH2D_ENABLED=true
GRAPH2D_MODEL_PATH=models/graph2d_training_cleaned_20260123.pth
GRAPH2D_MIN_CONF=0.7
GRAPH2D_ALLOW_LABELS=再沸器
GRAPH2D_EXCLUDE_LABELS=other
GRAPH2D_FUSION_ENABLED=true
FUSION_ANALYZER_ENABLED=true
FUSION_ANALYZER_OVERRIDE=true
FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.6
```

## Notes
- This configuration produced 2/50 overrides on the DXF validation sample.
- Expand `GRAPH2D_ALLOW_LABELS` only after re-validating on a fixed sample set.
