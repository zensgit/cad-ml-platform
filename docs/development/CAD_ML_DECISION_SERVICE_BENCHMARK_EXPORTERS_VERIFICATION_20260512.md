# CAD ML DecisionService Benchmark Exporters Verification

Date: 2026-05-12

## Scope

Validated DecisionService contract propagation through benchmark exporters,
downstream real-data benchmark summaries, and adjacent DecisionService consumers.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/eval_hybrid_dxf_manifest.py \
  scripts/batch_analyze_dxf_local.py \
  src/core/benchmark/realdata_signals.py \
  src/core/benchmark/realdata_scorecard.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_batch_analyze_dxf_local_knowledge_context.py \
  tests/unit/test_benchmark_realdata_signals.py \
  tests/unit/test_benchmark_realdata_scorecard.py
```

```bash
.venv311/bin/flake8 \
  scripts/eval_hybrid_dxf_manifest.py \
  scripts/batch_analyze_dxf_local.py \
  src/core/benchmark/realdata_signals.py \
  src/core/benchmark/realdata_scorecard.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_batch_analyze_dxf_local_knowledge_context.py \
  tests/unit/test_benchmark_realdata_signals.py \
  tests/unit/test_benchmark_realdata_scorecard.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_batch_analyze_dxf_local_knowledge_context.py \
  tests/unit/test_benchmark_realdata_signals.py \
  tests/unit/test_benchmark_realdata_scorecard.py \
  tests/unit/test_decision_service.py \
  tests/unit/test_classification_pipeline.py \
  tests/unit/test_batch_classify_pipeline.py \
  tests/unit/assistant/test_llm_api.py
```

```bash
git diff --check
```

## Results

- Python compile passed for touched scripts, benchmark helpers, and tests.
- Flake8 passed for touched scripts, benchmark helpers, and tests.
- Targeted pytest passed: `50 passed, 7 warnings in 6.31s`.
- Warnings are existing `ezdxf`/`pyparsing` deprecation warnings from dependencies.
- `git diff --check` passed.

## Coverage Notes

- `eval_hybrid_dxf_manifest.py` verifies row-level DecisionService contract export
  and aggregate decision coverage metrics.
- `batch_analyze_dxf_local.py` verifies local batch exporter contract extraction and
  summary metrics.
- Real-data signals and scorecard tests verify `decision_signals` survive into
  benchmark summary payloads consumed by release/reporting surfaces.
