# CAD ML DecisionService Batch Classify Verification

Date: 2026-05-12

## Scope

Validated the batch classify DecisionService integration, response-model exposure,
and adjacent classification behavior.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  src/core/classification/batch_classify_pipeline.py \
  src/api/v1/analyze_live_models.py \
  tests/unit/test_batch_classify_pipeline.py \
  tests/integration/test_analyze_batch_classify_pipeline.py \
  tests/unit/test_v16_classifier_endpoints.py
```

```bash
.venv311/bin/flake8 \
  src/core/classification/batch_classify_pipeline.py \
  src/api/v1/analyze_live_models.py \
  tests/unit/test_batch_classify_pipeline.py \
  tests/integration/test_analyze_batch_classify_pipeline.py \
  tests/unit/test_v16_classifier_endpoints.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_batch_classify_pipeline.py \
  tests/integration/test_analyze_batch_classify_pipeline.py \
  tests/unit/test_v16_classifier_endpoints.py \
  tests/unit/test_decision_service.py \
  tests/unit/test_classification_pipeline.py
```

```bash
git diff --check
```

## Results

- Python compile passed for touched implementation and tests.
- Flake8 passed for touched implementation and tests.
- Targeted pytest passed: `27 passed, 7 warnings in 2.12s`.
- `git diff --check` passed.

## Notes

- Error rows for unsupported files or unavailable classifiers remain error-only rows.
- Successful batch classification rows now carry the shared decision contract.
- The 7 warnings are existing `ezdxf`/`pyparsing` deprecation warnings from the test
  environment.
