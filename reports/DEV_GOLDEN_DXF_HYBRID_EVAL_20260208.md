# DEV_GOLDEN_DXF_HYBRID_EVAL_20260208

## Summary

Added a small "golden" evaluation manifest for DXF hybrid classification (filename + titleblock) and a local evaluation script that generates synthetic DXF bytes (no committed `.dxf` files) to verify deterministic behavior.

## Why

We need a fast, repeatable acceptance gate for changes to:

- filename parsing rules
- titleblock extraction / matching
- hybrid override rules and thresholds

without relying on local private training drawings.

## Changes

- `tests/golden/golden_dxf_hybrid_cases.json`
  - 20 golden cases with expected `{label, source}`.
  - Includes filename patterns (IDs, compare prefix, version suffix) and titleblock override cases.

- `scripts/eval_golden_dxf_hybrid.py`
  - Loads the manifest, generates a synthetic DXF (border + titleblock texts), runs `HybridClassifier`, and writes:
    - `summary.json`
    - `results.csv`
  - Defaults to a deterministic, dependency-light evaluation environment:
    - `GRAPH2D_ENABLED=false`
    - `PROCESS_FEATURES_ENABLED=false`
    - `TITLEBLOCK_ENABLED=true`
    - `TITLEBLOCK_OVERRIDE_ENABLED=true`
    - `TITLEBLOCK_MIN_CONF=0.6`

- `tests/unit/test_golden_dxf_hybrid_manifest.py`
  - Unit gate ensuring all golden cases match expected `label` (and `source` when specified).

## Validation

Executed:

```bash
python3 -m pytest -q tests/unit/test_golden_dxf_hybrid_manifest.py
python3 scripts/eval_golden_dxf_hybrid.py
```

Result (local run):

- total: `20`
- passed: `20`
- accuracy: `1.0`
- macro_f1: `1.0`
- low_conf_rate (threshold=0.5): `0.0`

Artifacts:

- `reports/experiments/20260208/golden_dxf_hybrid_eval/summary.json`
- `reports/experiments/20260208/golden_dxf_hybrid_eval/results.csv`

