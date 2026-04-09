# Benchmark Real-Data Scorecard Validation

## Goal

Add a reusable benchmark `realdata_scorecard` layer that compares real-data
outcomes across:

- DXF hybrid evaluation
- history-sequence `.h5` evaluation
- STEP smoke validation
- STEP directory evaluation

Then propagate that scorecard into:

- benchmark companion summary
- benchmark artifact bundle
- benchmark release decision
- benchmark release runbook

## Key Changes

- Added [src/core/benchmark/realdata_scorecard.py](../src/core/benchmark/realdata_scorecard.py)
- Added [scripts/export_benchmark_realdata_scorecard.py](../scripts/export_benchmark_realdata_scorecard.py)
- Updated:
  - [scripts/export_benchmark_companion_summary.py](../scripts/export_benchmark_companion_summary.py)
  - [scripts/export_benchmark_artifact_bundle.py](../scripts/export_benchmark_artifact_bundle.py)
  - [scripts/export_benchmark_release_decision.py](../scripts/export_benchmark_release_decision.py)
  - [scripts/export_benchmark_release_runbook.py](../scripts/export_benchmark_release_runbook.py)
- Added tests:
  - [tests/unit/test_benchmark_realdata_scorecard.py](../tests/unit/test_benchmark_realdata_scorecard.py)

## Design

The new `realdata_scorecard` differs from `realdata_signals`:

- `realdata_signals` answers whether benchmark real-data evidence exists
- `realdata_scorecard` answers how strong that evidence is across semantic
  surfaces

Output fields include:

- `status`
- `best_surface`
- `component_statuses`
- `ready_component_count`
- `partial_component_count`
- `environment_blocked_count`
- component-level `coarse_accuracy`, `exact_accuracy`, `sample_size`

Status contract:

- `realdata_scorecard_ready`
- `realdata_scorecard_partial`
- `realdata_scorecard_missing`

## Validation Commands

```bash
python3 -m py_compile \
  src/core/benchmark/realdata_scorecard.py \
  scripts/export_benchmark_realdata_scorecard.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_realdata_scorecard.py
```

```bash
flake8 \
  src/core/benchmark/realdata_scorecard.py \
  scripts/export_benchmark_realdata_scorecard.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_realdata_scorecard.py \
  --max-line-length=100
```

```bash
pytest -q \
  tests/unit/test_benchmark_realdata_scorecard.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

## Results

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `21 passed`

## Outcome

The benchmark stack now has a stable surface for comparing real-data semantic
evidence, instead of only reporting whether a DXF/history/STEP input existed.
