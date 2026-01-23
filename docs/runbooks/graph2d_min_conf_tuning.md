# Graph2D Min-Confidence Tuning

## Goal
Establish a practical default for `GRAPH2D_MIN_CONF` so Graph2D predictions only
participate in fusion when confidence is strong enough.

## Recommended Default
- `GRAPH2D_MIN_CONF=0.6`
  - In a 50-file DXF sample, 39/50 predictions passed at 0.6.
  - 0.5 passed 42/50, but admitted more low-confidence outputs.

## How to Tune
1) Run two batch analyses with different thresholds:
   - `GRAPH2D_MIN_CONF=0.5`
   - `GRAPH2D_MIN_CONF=0.6`
2) Compare pass rate + downstream fusion outcomes.
3) Pick the lowest threshold that avoids noisy overrides.

## Notes
- Keep `GRAPH2D_MIN_CONF` separate from `FUSION_GRAPH2D_OVERRIDE_MIN_CONF`.
- If the model is re-trained, re-evaluate this threshold.
