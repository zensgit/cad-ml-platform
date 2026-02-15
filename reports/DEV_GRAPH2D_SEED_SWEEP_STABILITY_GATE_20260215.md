# DEV_GRAPH2D_SEED_SWEEP_STABILITY_GATE_20260215

## Goal

Add a reproducible stability gate on top of Graph2D seed sweeps so local/CI workflows can fail fast when strict-mode quality regresses.

## Implementation

Updated `scripts/sweep_graph2d_profile_seeds.py`:

- Added gate CLI flags:
  - `--min-strict-accuracy-mean`
  - `--min-strict-accuracy-min`
  - `--require-all-ok`
- Added gate evaluation logic via `_evaluate_gate(...)`.
- Extended summary output (`seed_sweep_summary.json`) with:
  - `num_success_runs`, `num_error_runs`
  - `gate.enabled`, `gate.passed`, `gate.failures`
  - configured threshold values.
- Added non-zero failure exit:
  - returns `3` when gate is enabled and not passed.

## Unit Tests

Updated `tests/unit/test_sweep_graph2d_profile_seeds.py`:

- existing seed parser tests retained.
- added gate tests for:
  - disabled gate pass behavior
  - mean-threshold failure
  - `require_all_ok` failure when error rows exist.

Validation:

```bash
.venv/bin/python -m pytest tests/unit/test_run_graph2d_pipeline_local_profile.py tests/unit/test_sweep_graph2d_profile_seeds.py tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py tests/unit/test_run_graph2d_pipeline_local_diagnose_strict_wiring.py -v
```

Result: `11 passed`.

## Runtime Verification

### A) Real 5-seed strict run with gate

Command:

```bash
/usr/bin/time -p .venv/bin/python scripts/sweep_graph2d_profile_seeds.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --training-profile strict_node23_edgesage_v1 \
  --seeds 7,13,21,42,84 \
  --min-strict-accuracy-mean 0.30 \
  --min-strict-accuracy-min 0.25 \
  --require-all-ok
```

Artifact root:

- `/tmp/graph2d_profile_seed_sweep_20260215_220844`

Per-seed strict accuracy:

- seed `7`: `0.381818`
- seed `13`: `0.281818`
- seed `21`: `0.281818`
- seed `42`: `0.354545`
- seed `84`: `0.400000`

Aggregate:

- `strict_accuracy_mean`: `0.340000`
- `strict_accuracy_min`: `0.281818`
- `strict_accuracy_max`: `0.400000`
- `num_error_runs`: `0`
- gate: `passed=true`

### B) Gate failure exit-code smoke

Command:

```bash
.venv/bin/python scripts/sweep_graph2d_profile_seeds.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --seeds 7 \
  --dry-run \
  --min-strict-accuracy-mean 0.1
```

Observed return code:

- `rc=3` (expected gate failure behavior)

## Notes

- During runs, `ezdxf` printed cache write warnings for:
  - `/Users/huazhou/.cache/ezdxf/font_manager_cache.json`
- These warnings did not fail training/eval/diagnosis and did not affect gate pass/fail logic.
