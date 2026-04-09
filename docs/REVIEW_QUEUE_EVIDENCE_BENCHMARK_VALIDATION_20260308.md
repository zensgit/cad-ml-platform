# Review Queue Evidence Benchmark Validation

## Goal

Teach the benchmark scorecard to consume evidence richness from the
active-learning review queue summary instead of treating review queue health as
backlog-only.

## Delivered

- Extended the `review_queue` benchmark component with:
  - `evidence_count_total`
  - `average_evidence_count`
  - `records_with_evidence_count`
  - `records_with_evidence_ratio`
  - `top_evidence_sources`
- Added `evidence_gap` status when backlog exists but evidence coverage is too low.
- Wired `evidence_gap` into:
  - `overall_status`
  - benchmark recommendations
  - markdown rendering

## Validation

```bash
python3 -m py_compile \
  scripts/generate_benchmark_scorecard.py \
  tests/unit/test_generate_benchmark_scorecard.py

flake8 \
  scripts/generate_benchmark_scorecard.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  --max-line-length=100

pytest -q tests/unit/test_generate_benchmark_scorecard.py
```

## Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `4 passed`

## Notes

- The benchmark remains backward-compatible with older review queue summaries.
- Missing evidence fields default to zero/empty values.
- `evidence_gap` is intentionally softer than `critical_backlog` and only fires
  when review work exists but reviewer evidence is still weak.
