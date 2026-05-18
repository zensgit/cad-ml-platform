# CAD ML Reviewed Manifest Hybrid Consumption Verification

Date: 2026-05-13

## Scope

Validated workflow wiring that lets hybrid blind evaluation prefer the merged
reviewed benchmark manifest produced by the forward scorecard wrapper. The checks
cover same-run step ordering, manifest fallback behavior, and output metadata.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
.venv311/bin/flake8 \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_evaluation_report_workflow_hybrid_superpass_step.py \
  tests/unit/test_hybrid_superpass_workflow_integration.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
git diff --check
```

## Results

- Python compile passed for touched workflow tests.
- Flake8 passed for touched workflow tests.
- Targeted pytest passed: `13 passed, 7 warnings in 3.41s`.
- `git diff --check` passed.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings imported
  through existing DXF evaluation helpers.

## Verified Behavior

- The forward scorecard step runs before hybrid blind evaluation, so merge outputs are
  available in the same workflow run.
- Hybrid blind evaluation checks
  `steps.forward_scorecard.outputs.manufacturing_review_manifest_merge_available`.
- When a merged reviewed benchmark manifest exists, hybrid blind evaluation uses it as
  `MANIFEST_CSV`.
- When the merged manifest is unavailable or missing, the workflow keeps the existing
  `HYBRID_BLIND_MANIFEST_CSV` fallback.
- The hybrid blind step emits `manifest_source=reviewed_benchmark_manifest` when the
  reviewed manifest is selected.
- The hybrid blind step emits `manifest_csv` and `manifest_source` in normal and
  missing-summary paths.
