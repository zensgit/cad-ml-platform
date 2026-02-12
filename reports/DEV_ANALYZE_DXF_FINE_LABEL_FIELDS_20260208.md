# DEV_ANALYZE_DXF_FINE_LABEL_FIELDS_20260208

## Goal
Expose HybridClassifier fine-grained labels (filename/titleblock/process) in the
`/api/v1/analyze/` response even when `part_type` is produced by L2/L3 Fusion.

This keeps the coarse/fusion decision stable while still returning the best-effort,
human-readable label that is often available from drawings metadata.

## Changes
- `src/api/v1/analyze.py`
  - When `classification.hybrid_decision.label` is present, add additive fields:
    - `classification.fine_part_type`
    - `classification.fine_confidence`
    - `classification.fine_source`
    - `classification.fine_rule_version` (`HybridClassifier-v1`)
  - This does **not** override `classification.part_type`.

- `tests/integration/test_analyze_dxf_fusion.py`
  - Added `test_analyze_dxf_adds_fine_label_fields_from_hybrid` to assert:
    - Fusion `part_type` remains unchanged
    - Fine label fields are present and stable

## Verification
Command:
```bash
python3 -m pytest -q \
  tests/integration/test_analyze_dxf_fusion.py::test_analyze_dxf_adds_fine_label_fields_from_hybrid
```

Result: `1 passed`

