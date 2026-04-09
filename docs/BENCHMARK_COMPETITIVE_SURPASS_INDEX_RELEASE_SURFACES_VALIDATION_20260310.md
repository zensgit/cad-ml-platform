# Benchmark Competitive Surpass Index Release Surfaces Validation

## Scope

- Extend `benchmark release decision` with `competitive_surpass_index` status,
  gaps, recommendations, and artifact presence.
- Extend `benchmark release runbook` with the same `competitive_surpass_index`
  payload so downstream release guidance can reason about benchmark readiness.

## Changed Files

- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_runbook.py`

## Validation

```bash
python3 -m py_compile \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

git diff --check
```

## Result

- `release decision` now exposes:
  - `competitive_surpass_index_status`
  - `competitive_surpass_index`
  - `competitive_surpass_primary_gaps`
  - `competitive_surpass_recommendations`
- `release runbook` now exposes the same fields and renders a dedicated
  `Competitive Surpass Index` section in markdown.
- CLI coverage includes `--benchmark-competitive-surpass-index`.
