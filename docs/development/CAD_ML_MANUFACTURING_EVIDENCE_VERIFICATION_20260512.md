# CAD ML Manufacturing Evidence Verification

Date: 2026-05-12

## Scope

Validated Phase 6 manufacturing evidence from the summary builder through analyze
integration and adjacent pipeline tests.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  src/core/process/manufacturing_summary.py \
  src/core/analysis_manufacturing_summary.py \
  src/core/process/__init__.py \
  tests/unit/test_manufacturing_summary.py \
  tests/unit/test_analysis_manufacturing_summary.py \
  tests/integration/test_analyze_manufacturing_summary.py
```

```bash
.venv311/bin/flake8 \
  src/core/process/manufacturing_summary.py \
  src/core/analysis_manufacturing_summary.py \
  src/core/process/__init__.py \
  tests/unit/test_manufacturing_summary.py \
  tests/unit/test_analysis_manufacturing_summary.py \
  tests/integration/test_analyze_manufacturing_summary.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_manufacturing_summary.py \
  tests/unit/test_analysis_manufacturing_summary.py \
  tests/integration/test_analyze_manufacturing_summary.py
```

```bash
.venv311/bin/pytest -q \
  tests/integration/test_analyze_process_pipeline.py \
  tests/integration/test_analyze_quality_pipeline.py \
  tests/unit/test_analysis_parallel_pipeline.py \
  tests/unit/test_process_pipeline.py \
  tests/unit/test_quality_pipeline.py \
  tests/unit/test_decision_service.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py
```

```bash
git diff --check
```

## Results

- Python compile passed for touched implementation and tests.
- Flake8 passed for touched implementation and tests.
- Manufacturing-focused pytest passed: `12 passed, 7 warnings in 3.19s`.
- Adjacent analyze/process/quality/decision pytest passed:
  `21 passed, 7 warnings in 2.51s`.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings.
- `git diff --check` passed.

## Verified Behavior

- `results["manufacturing_evidence"]` is emitted when DFM/process/cost outputs exist.
- With classification enabled, the same rows are appended to
  `classification["evidence"]`.
- `classification["decision_contract"]["evidence"]` stays aligned with
  `classification["evidence"]`.
- Analyze integration covers the four new manufacturing evidence sources:
  `dfm`, `manufacturing_process`, `manufacturing_cost`, and
  `manufacturing_decision`.
