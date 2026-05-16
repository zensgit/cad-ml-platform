# CAD ML Forward Scorecard Knowledge Grounding Verification

Date: 2026-05-12

## Scope

Validated that forward scorecard release readiness now depends on knowledge
grounding coverage, not only knowledge readiness.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  src/core/benchmark/forward_scorecard.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
.venv311/bin/flake8 \
  src/core/benchmark/forward_scorecard.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_benchmark_engineering_signals.py
```

```bash
git diff --check
```

## Results

- Python compile passed for touched implementation and tests.
- Flake8 passed for touched implementation and tests.
- Forward scorecard, release gate, benchmark scorecard, and engineering signal tests
  passed: `22 passed, 7 warnings in 2.68s`.
- `git diff --check` passed.

## Regression Covered

- A knowledge summary with `knowledge_foundation_ready` but no `knowledge_grounding`
  now yields `benchmark_ready_with_gap`.
- CI wrapper ready fixtures now include grounding evidence, so valid ready bundles
  still export `overall_status=release_ready`.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings.
