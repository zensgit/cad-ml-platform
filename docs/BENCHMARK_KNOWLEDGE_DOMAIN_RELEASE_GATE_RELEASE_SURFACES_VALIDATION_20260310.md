# Benchmark Knowledge Domain Release Gate Release Surfaces Validation

## Scope
- release decision surface
- release runbook surface
- release-gate fields, blockers, review signals, markdown sections

## Key Changes
- `scripts/export_benchmark_release_decision.py`
  - kept `knowledge_domain_release_gate` wired into blockers, review signals, payload,
    artifacts, CLI args, and markdown output
- `scripts/export_benchmark_release_runbook.py`
  - added `knowledge_domain_release_gate` component parsing
  - added release-gate artifact tracking
  - added release-gate blocker and review-signal propagation
  - added release-gate payload fields and markdown section
  - added CLI arg and artifact path wiring
- `tests/unit/test_benchmark_release_decision.py`
  - added blocked release-gate fixture and assertions
- `tests/unit/test_benchmark_release_runbook.py`
  - added blocked release-gate fixture and assertions

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

## Expected Outcome
- blocked `knowledge_domain_release_gate` feeds release blockers
- gate warnings/recommendations feed release review signals
- release decision/runbook payloads expose stable gate fields
- markdown renders a `## Knowledge Domain Release Gate` section
