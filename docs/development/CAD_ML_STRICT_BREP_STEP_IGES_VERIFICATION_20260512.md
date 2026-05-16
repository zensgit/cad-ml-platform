# CAD ML Strict STEP/IGES B-Rep Verification

Date: 2026-05-12

## Scope

Validated the strict STEP/IGES evaluator slice, IGES loader routing, graph QA output,
and downstream benchmark compatibility.

## Commands

```bash
PYTHONPYCACHEPREFIX=/private/tmp/cad-ml-pycache .venv311/bin/python -m py_compile \
  scripts/eval_brep_step_dir.py \
  src/core/geometry/engine.py \
  tests/unit/test_eval_brep_step_dir.py
```

```bash
.venv311/bin/flake8 --max-line-length=100 \
  scripts/eval_brep_step_dir.py \
  src/core/geometry/engine.py \
  tests/unit/test_eval_brep_step_dir.py
```

```bash
.venv311/bin/python -m pytest tests/unit/test_eval_brep_step_dir.py -q
```

```bash
.venv311/bin/python -m pytest \
  tests/unit/test_eval_brep_step_dir.py \
  tests/unit/test_benchmark_realdata_signals.py \
  tests/unit/test_benchmark_realdata_scorecard.py \
  tests/unit/test_forward_scorecard.py -q
```

```bash
git diff --check
```

## Results

- Python compile passed under `.venv311`.
- Flake8 passed for the strict evaluator slice.
- `tests/unit/test_eval_brep_step_dir.py`: `11 passed, 7 warnings in 1.96s`.
- Related downstream suite:
  `23 passed, 7 warnings in 1.93s`.
- `git diff --check` passed.

## Coverage

- Strict mode fails invalid B-Rep features.
- Strict mode rejects synthetic/demo geometry unless `--allow-demo-geometry` is set.
- IGES files use `GeometryEngine.load_iges(...)` when available.
- IGES files fail with `iges_loader_missing` instead of being pushed through the STEP
  parser when no IGES loader exists.
- Summary output tracks parse success, graph validity, topology counts, surface type
  histogram, latency, and stable failure reasons.
- `graph_qa.json` lists invalid graph rows for release review.

## Notes

- The current local validation is unit-level because this environment does not provide
  a real OCC-backed STEP/IGES golden directory for strict end-to-end parsing.
- The next Phase 4 slice should create the real golden manifest and run this evaluator
  against that manifest in an OCC-enabled runtime.
