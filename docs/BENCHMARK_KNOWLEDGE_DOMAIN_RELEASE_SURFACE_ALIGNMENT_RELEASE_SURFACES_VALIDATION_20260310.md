# Benchmark Knowledge Domain Release Surface Alignment Release Surfaces Validation

## Scope
- Lift `knowledge_domain_release_surface_alignment` into:
  - `scripts/export_benchmark_release_decision.py`
  - `scripts/export_benchmark_release_runbook.py`
- Cover both blocking and ready paths in release decision/runbook tests.
- Ensure CLI render paths include the new section and artifact presence.

## Changed Files
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_runbook.py`

## Validation Commands
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
- `py_compile`: pass
- `flake8`: pass
- `pytest`: pass
- `git diff --check`: pass

## Notes
- Added aligned fixture coverage for ready/CLI release-runbook paths so the new
  artifact is no longer treated as missing evidence.
- Added release-surface alignment section rendering assertions to keep markdown
  output contract stable.
