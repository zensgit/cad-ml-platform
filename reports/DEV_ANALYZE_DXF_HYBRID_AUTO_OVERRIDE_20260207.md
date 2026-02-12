# DEV_ANALYZE_DXF_HYBRID_AUTO_OVERRIDE_20260207

## Summary

Improved DXF part-type classification quality in `/api/v1/analyze/` by automatically applying the existing `HybridClassifier` decision when the base classifier is clearly a placeholder (rule_version=`v1` bucket types like `simple_plate`).

This makes common production-style filenames (e.g. `J2925001-01人孔v2.dxf`) yield a meaningful part label without requiring any env toggle.

## Changes

### 1) Auto Hybrid Override for Placeholder Rule Outputs

Updated `src/api/v1/analyze.py`:

- Existing behavior kept for fusion/ML outputs.
- For DXF requests where classification is:
  - `confidence_source == "rules"`
  - `rule_version == "v1"`
  - and `part_type` is one of placeholder bucket types
- If `HybridClassifier` produced a label with confidence >= `HYBRID_OVERRIDE_MIN_CONF` (default `0.8`),
  the API now overrides:
  - `part_type`
  - `confidence`
  - `rule_version` -> `HybridClassifier-v1`
  - `confidence_source` -> `hybrid`

Added response metadata:

- `classification.hybrid_override_applied` (mode, min_confidence, previous values)

### 2) Force Override Still Supported

If `HYBRID_CLASSIFIER_OVERRIDE=true` is set, the override is forced (subject to the same `HYBRID_OVERRIDE_MIN_CONF` threshold),
and a failure results in `classification.hybrid_override_skipped`.

New env var:

- `HYBRID_CLASSIFIER_AUTO_OVERRIDE` (default `true`) to disable the auto behavior if needed.

## Tests

Added integration coverage:

- `tests/integration/test_analyze_dxf_hybrid_override.py`
  - Confirms a placeholder DXF classification is overridden to a filename-derived label (`人孔`).

## Validation

Executed:

```bash
.venv/bin/pytest -q \
  tests/integration/test_analyze_dxf_hybrid_override.py \
  tests/integration/test_analyze_dxf_fusion.py::test_analyze_dxf_triggers_l2_fusion
```

Results:

- `2 passed`

## Notes

- This change does not affect the L2/L3 fusion cases (e.g. `Bolt_M6x20.dxf`) because those produce `confidence_source="fusion"` and are not considered placeholder outputs.
