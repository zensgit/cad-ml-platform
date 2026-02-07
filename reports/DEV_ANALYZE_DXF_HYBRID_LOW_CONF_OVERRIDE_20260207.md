# DEV_ANALYZE_DXF_HYBRID_LOW_CONF_OVERRIDE_20260207

## Goal
Make DXF part-type classification surface meaningful fine-grained labels from `HybridClassifier` not only when the base classifier returns placeholder buckets, but also when the base classifier returns a low-confidence result (for example coarse ML categories with near-random confidence).

## Changes
- `src/api/v1/analyze.py`
  - Added env var `HYBRID_OVERRIDE_BASE_MAX_CONF` (default `0.7`).
  - Extended auto-override to apply when:
    - `HYBRID_CLASSIFIER_AUTO_OVERRIDE=true` (default), and
    - `HybridClassifier` has `confidence >= HYBRID_OVERRIDE_MIN_CONF` (default `0.8`), and
    - base DXF classification is either:
      - placeholder rule bucket (`confidence_source=rules`, `rule_version=v1`, `part_type` in `{simple_plate, moderate_component, complex_assembly, unknown, other}`), or
      - low-confidence base result (`confidence_source=rules`, `confidence < HYBRID_OVERRIDE_BASE_MAX_CONF`).
  - Override metadata now includes `base_max_confidence`, and uses `mode=auto_low_conf` for the low-confidence path.

## Verification
- Ran:
  - `.venv/bin/pytest -q tests/integration/test_analyze_dxf_hybrid_override.py`
- Result:
  - `2 passed`

## Notes
- Force override remains supported via `HYBRID_CLASSIFIER_OVERRIDE=true`.
- To disable auto behavior: `HYBRID_CLASSIFIER_AUTO_OVERRIDE=false`.
