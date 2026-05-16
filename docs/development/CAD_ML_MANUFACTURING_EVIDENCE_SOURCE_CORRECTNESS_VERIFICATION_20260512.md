# CAD ML Manufacturing Evidence Source Correctness Verification

Date: 2026-05-12

## Scope

Validated reviewed manufacturing evidence source labels, aggregate correctness
metrics, and forward scorecard release gating.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/eval_hybrid_dxf_manifest.py \
  src/core/benchmark/forward_scorecard.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
.venv311/bin/flake8 \
  scripts/eval_hybrid_dxf_manifest.py \
  src/core/benchmark/forward_scorecard.py \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_eval_hybrid_dxf_manifest.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
git diff --check
```

## Results

- Python compile passed for touched exporter, scorecard, and tests.
- Flake8 passed for touched exporter, scorecard, and tests.
- Targeted pytest passed: `26 passed, 7 warnings in 3.96s`.
- `git diff --check` passed.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings.

## Verified Behavior

- Manifest rows can carry reviewed manufacturing evidence source labels.
- `none` marks a reviewed sample with no expected manufacturing evidence.
- Row-level output records true-positive, false-positive, and false-negative sources.
- Summary output includes reviewed sample count, exact-match rate, micro
  precision/recall/F1, and per-source correctness.
- Forward scorecard `manufacturing_evidence` requires reviewed correctness evidence
  for `release_ready`.
- Missing correctness metrics downgrade an otherwise high-coverage manufacturing
  evidence component to `benchmark_ready_with_gap`.
