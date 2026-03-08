# Benchmark Scorecard Validation 2026-03-08

## Scope

This change adds a single scorecard generator so benchmark and "beyond" claims
stop depending on scattered validation docs.

The generator consumes existing artifact summaries and emits:

- one machine-readable JSON scorecard
- one Markdown scorecard for review, release notes, and benchmark reporting

## Motivation

The repository already had strong validation primitives:

- `scripts/eval_hybrid_dxf_manifest.py`
- `scripts/eval_history_sequence_classifier.py`
- `scripts/eval_brep_step_dir.py`
- vector migration readiness / plan summaries

But there was still no single artifact answering:

1. which branch is strong right now
2. which branch is still evidence-poor
3. whether governance is production-ready
4. whether the current run is only benchmark-ready or closer to benchmark-beyond

This patch closes that gap.

## Added Files

- `scripts/generate_benchmark_scorecard.py`
- `tests/unit/test_generate_benchmark_scorecard.py`

## Output Contract

The scorecard reports:

- `overall_status`
- `components.hybrid`
- `components.graph2d`
- `components.history_sequence`
- `components.brep`
- `components.migration_governance`
- `recommendations[]`

The current policy encoded by the script is intentionally conservative:

- `Graph2D` becomes `weak_signal_only` when diagnosis accuracy is low or
  low-confidence saturation is high
- `Hybrid` becomes `strong_primary` only when exact accuracy is already strong
- `History Sequence` stays `smoke_only` or `needs_more_evidence` until sample
  count and coverage are both meaningful
- `B-Rep` is only `graph_ready` when real 3D validation and `v2` graph schema
  evidence both exist
- migration governance is only `operationally_ready` when planning is both
  ready and coverage-complete

## Validation

```bash
python3 -m py_compile scripts/generate_benchmark_scorecard.py \
  tests/unit/test_generate_benchmark_scorecard.py

flake8 scripts/generate_benchmark_scorecard.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  --max-line-length=100

pytest -q tests/unit/test_generate_benchmark_scorecard.py
```

## Result

- `py_compile`: passed
- `flake8`: passed
- `pytest`: expected `2 passed`

## Practical Outcome

The benchmark line now has a stable, reproducible total scorecard artifact.

That makes it possible to say, with one file instead of many docs:

- whether the current line is benchmark-ready
- where evidence is still missing
- whether governance is ready for rollout
- what the next engineering recommendation is
