# DEV_DXF_BATCH_ANALYZE_WEAK_LABEL_REGRESSION_METRICS_20260213

## Summary

Enhanced `scripts/batch_analyze_dxf_local.py` to support weak-label regression metrics derived
from original DXF filenames (even when uploads are masked). This enables automated evaluation
in environments where manual DXF review is not feasible.

## Changes

- `scripts/batch_analyze_dxf_local.py`
  - Added `--weak-label-min-confidence` (default `0.8`) to accept/reject weak labels from
    `FilenameClassifier` (original filename, not upload name).
  - Added stable CSV schema generation (fieldname union) so runs do not crash when early rows
    are read/API errors.
  - Added per-file weak label columns:
    - `weak_true_label`, `weak_true_confidence`, `weak_true_status`, `weak_true_match_type`,
      `weak_true_extracted_name`, `weak_true_accepted`
    - `upload_name` (helps when `--mask-filename` is used)
  - Added summary metrics in `summary.json`:
    - weak-label coverage
    - accuracy for: final `part_type`, `hybrid_label`, `graph2d_label`, `titleblock_label`,
      and `hybrid_filename_label`
    - top confusion pairs for final `part_type`

## Usage

Run with masked uploads but weak-label evaluation based on original filenames:

```bash
.venv/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir /path/to/dxfs \
  --mask-filename \
  --weak-label-min-confidence 0.8 \
  --output-dir /tmp/dxf_batch_eval
```

## Validation

Executed:

```bash
.venv/bin/python -m py_compile scripts/batch_analyze_dxf_local.py
.venv/bin/python -m pytest tests/unit/test_freeze_graph2d_baseline.py tests/unit/test_graph2d_script_config.py -v
make validate-core-fast
```

Results:

- script compile check: OK
- selected script/unit tests: `4 passed`
- `make validate-core-fast`: passed

## Notes / Caveats

- Weak labels are derived from naming conventions and synonym mappings and are not ground truth.
  Coverage and accuracy should be interpreted as regression signals, not final model quality.

