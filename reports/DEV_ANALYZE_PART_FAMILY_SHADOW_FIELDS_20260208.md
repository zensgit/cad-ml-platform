# DEV_ANALYZE_PART_FAMILY_SHADOW_FIELDS_20260208

## Goal
Harden the **shadow-only** PartClassifier provider path in `/api/v1/analyze/` by:
- Normalizing coarse labels into stable `part_family*` fields
- Adding timeout + max-file-size guards
- Avoiding cache cross-hits between runs with/without shadow fields
- Emitting Prometheus metrics for visibility

This work is **additive** and does **not** override `classification.part_type`.

## Changes
- `src/core/classification/part_family.py` (+ `src/core/classification/__init__.py`)
  - Added `normalize_part_family_prediction(...)` to convert provider payloads into stable `part_family*` fields.

- `src/api/v1/analyze.py`
  - Added normalized `part_family*` fields alongside `classification.part_classifier_prediction`.
  - Added format gating via `PART_CLASSIFIER_PROVIDER_SHADOW_FORMATS` (default `dxf,dwg`).
  - Added timeout via `PART_CLASSIFIER_PROVIDER_TIMEOUT_SECONDS` (default `2.0`).
  - Added max-size guard via `PART_CLASSIFIER_PROVIDER_MAX_MB` (default `10.0`).
  - Added optional cache-key separation via `PART_CLASSIFIER_PROVIDER_INCLUDE_IN_CACHE_KEY` (default `true`).

- `src/utils/analysis_metrics.py`
  - Added:
    - `analysis_part_classifier_requests_total{status,provider}`
    - `analysis_part_classifier_seconds{provider}`
    - `analysis_part_classifier_skipped_total{reason}`

- `docs/ANALYZE_CLASSIFICATION_FIELDS.md`
  - Documented primary vs additive classification fields and the new shadow env flags.

- `scripts/eval_part_family_shadow.py`
  - Added a lightweight local evaluator (TestClient-based) that writes a CSV for review.

- Tests
  - `tests/integration/test_analyze_dxf_fusion.py`
    - Stub provider case asserts `part_family*` fields are present.
    - Timeout case asserts `part_classifier_prediction.status == "timeout"` and `part_family_error.code == "timeout"`.
  - `tests/unit/test_part_family_normalization.py`
    - Added unit coverage for normalization edge cases.
  - `tests/conftest.py`
    - Isolated new env vars for test determinism.

## Verification
Commands:
```bash
.venv/bin/python -m pytest -q tests/integration/test_analyze_dxf_fusion.py
.venv/bin/python -m pytest -q tests/unit/test_part_family_normalization.py
.venv/bin/python -m pytest -q tests/unit/test_provider_framework.py tests/unit/test_provider_registry_bootstrap.py
```

Result:
- All tests passed locally.

