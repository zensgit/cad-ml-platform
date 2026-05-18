# CAD ML Knowledge Rule Metadata Verification

Date: 2026-05-12

## Scope

Validated Phase 6 knowledge rule metadata from the knowledge summary layer through
DecisionService evidence and analyze API integration.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  src/core/knowledge/analysis_summary.py \
  src/core/classification/decision_service.py \
  tests/unit/test_knowledge_analysis_summary.py \
  tests/unit/test_decision_service.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py
```

```bash
.venv311/bin/flake8 \
  src/core/knowledge/analysis_summary.py \
  src/core/classification/decision_service.py \
  tests/unit/test_knowledge_analysis_summary.py \
  tests/unit/test_decision_service.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_knowledge_analysis_summary.py \
  tests/unit/test_decision_service.py \
  tests/unit/test_classification_pipeline.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_batch_analyze_dxf_local_knowledge_context.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_classification_finalization.py \
  tests/unit/test_classification_decision_contract.py \
  tests/unit/test_classification_active_learning_policy.py \
  tests/unit/test_benchmark_engineering_signals.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_export_hybrid_rejection_review_pack_context.py
```

```bash
git diff --check
```

## Results

- Python compile passed for touched implementation and tests.
- Flake8 passed for touched implementation and tests.
- Targeted pytest passed: `18 passed, 7 warnings in 2.12s`.
- Adjacent classification/benchmark pytest passed: `23 passed, 7 warnings in 2.28s`.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings.
- `git diff --check` passed.

## Regression Fixed

The first pytest run exposed an import cycle between
`src.core.knowledge.analysis_summary` and `src.core.classification`. The slice fixed
this by lazily importing coarse-label normalization only when knowledge summaries are
built.
