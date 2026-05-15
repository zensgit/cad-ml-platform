# CAD ML Manufacturing Evidence Benchmark Summary Verification

Date: 2026-05-12

## Scope

Validated row-level manufacturing evidence export, aggregate manufacturing evidence
summary generation, and compatibility with the forward scorecard component.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/batch_analyze_dxf_local.py \
  scripts/eval_hybrid_dxf_manifest.py \
  tests/unit/test_batch_analyze_dxf_local_knowledge_context.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py
```

```bash
.venv311/bin/flake8 \
  scripts/batch_analyze_dxf_local.py \
  scripts/eval_hybrid_dxf_manifest.py \
  tests/unit/test_batch_analyze_dxf_local_knowledge_context.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_batch_analyze_dxf_local_knowledge_context.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_forward_scorecard.py
```

```bash
git diff --check
```

## Results

- Python compile passed for touched exporters and tests.
- Flake8 passed for touched exporters and tests.
- Targeted pytest passed: `21 passed, 7 warnings in 2.63s`.
- `git diff --check` passed.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings.

## Verified Behavior

- Local batch DXF exporter emits `manufacturing_evidence` row fields and aggregate
  summary coverage.
- Labeled DXF manifest exporter emits the same row fields and summary contract.
- Exporters prefer explicit analyze manufacturing evidence and can fall back to
  manufacturing-source rows in DecisionService evidence.
- The generated summary shape matches the forward scorecard manufacturing evidence
  input contract.
