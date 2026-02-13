# DEV_DXF_BATCH_ANALYZE_FINE_LABEL_METRICS_20260213

## Summary

Extended `scripts/batch_analyze_dxf_local.py` to treat the DXF **fine label** as a first-class
regression signal:

- Captures `fine_part_type` / `fine_confidence` / `fine_source` / `fine_rule_version` from
  `/api/v1/analyze/`.
- Adds weak-label accuracy scoring for `fine_part_type` (in addition to `part_type` and other
  diagnostic labels).
- Writes `fine_label_distribution.csv` alongside the existing `label_distribution.csv`.

This is useful because `part_type` may be produced by coarse rules/fusion logic, while
`fine_part_type` is the best-effort fine-grained decision (typically from `HybridClassifier`).

## Changes

- `scripts/batch_analyze_dxf_local.py`
  - Added `fine_part_type` weak-label scoring + summary metrics (`weak_labels.accuracy.fine_part_type`).
  - Added `fine_label_counts` + `fine` summary section (presence, source counts, confidence stats).
  - Added per-row CSV columns: `fine_part_type`, `fine_confidence`, `fine_source`, `fine_rule_version`.
  - Added `fine_label_distribution.csv` output.

## Validation

Executed:

```bash
.venv/bin/python -m py_compile scripts/batch_analyze_dxf_local.py
make validate-core-fast

.venv/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --max-files 30 \
  --output-dir /tmp/dxf_batch_eval_unmasked_20260213_v3
```

Key results (`/tmp/dxf_batch_eval_unmasked_20260213_v3/summary.json`):

- `weak_labels.covered_rate = 1.0` (`30/30`, `FilenameClassifier` exact matches)
- `weak_labels.accuracy.fine_part_type.accuracy = 1.0` (`30/30`)
- `weak_labels.accuracy.final_part_type.accuracy = 0.7667` (`23/30`)
- `graph2d.status_counts.model_unavailable = 30` (this environment does not have `torch`)

## Outputs

- `batch_results.csv` / `batch_results_sanitized.csv`
- `summary.json`
- `label_distribution.csv` (distribution for `part_type`)
- `fine_label_distribution.csv` (distribution for `fine_part_type`)

## Notes / Caveats

- Weak labels are derived from file naming conventions + synonym mappings and are not ground truth.
  Treat them as regression indicators.
- `part_type` and `fine_part_type` may live in different label spaces; prefer `fine_part_type`
  when you want fine-grained part naming for DXF.
