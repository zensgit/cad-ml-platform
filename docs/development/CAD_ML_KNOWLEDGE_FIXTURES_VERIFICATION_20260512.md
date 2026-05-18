# CAD ML Knowledge Fixtures Verification

Date: 2026-05-12

## Scope

Validated the Phase 6 knowledge fixtures from the extraction layer through the
adjacent decision and scorecard tests.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  src/core/knowledge/analysis_summary.py \
  tests/unit/test_knowledge_analysis_summary.py
```

```bash
.venv311/bin/flake8 \
  src/core/knowledge/analysis_summary.py \
  tests/unit/test_knowledge_analysis_summary.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_knowledge_analysis_summary.py \
  tests/unit/test_decision_service.py \
  tests/integration/test_analyze_dxf_coarse_knowledge_outputs.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
git diff --check
```

## Results

- Python compile passed for touched implementation and fixture tests.
- Flake8 passed for touched implementation and fixture tests.
- Targeted and adjacent pytest passed: `23 passed, 7 warnings in 3.08s`.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings.
- `git diff --check` passed.

## Fixture Evidence

- Material substitution: `304 -> 316L` resolves to `S30408 -> S31603` and carries
  material equivalence metadata.
- Fit validation: `H7/g6` resolves through the ISO 286 fit catalog as
  `clearance` / `normal_running`.
- Surface finish recommendation: Ra/N-grade recommendation text emits a
  `surface_finish_recommendation` check.
- Machining process route: route text emits ordered process steps such as turning,
  milling, drilling, and grinding.
- Manufacturability risk: thin-wall, deep-hole, and high-stock-removal signals emit
  both knowledge checks and DFM-rule violations.
