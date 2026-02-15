# DEV_GRAPH2D_CI_SEED_GATE_SUMMARY_STRICT_CHANNEL_20260215

## Scope

Complete the follow-up `1+2` package:

1. Add Graph2D seed-gate markdown summary output in CI step summary.
2. Add a strict-mode CI channel (config + target + optional workflow lane).

## Implementation

### A) CI summary output for seed gate

- Added `scripts/ci/summarize_graph2d_seed_gate.py`.
- Added unit tests:
  - `tests/unit/test_graph2d_seed_gate_summary.py`
- CI integration in `.github/workflows/ci.yml`:
  - append seed-gate summary into `GITHUB_STEP_SUMMARY` after the gate run.

### B) Strict-mode CI channel

- Added strict config:
  - `config/graph2d_seed_gate_strict.yaml`
- Added Make target:
  - `validate-graph2d-seed-gate-strict`
- Added optional CI lane (3.11 only, guarded by repository variable):
  - `vars.GRAPH2D_STRICT_SEED_GATE_ENABLED == 'true'`
  - runs strict gate
  - uploads strict gate log
  - appends strict gate summary to `GITHUB_STEP_SUMMARY`.

### C) Strict non-normalized override path

To support strict profile without label normalization collapse:

- `scripts/run_graph2d_pipeline_local.py`
  - new args:
    - `--force-normalize-labels {auto,true,false}`
    - `--force-clean-min-count`
  - new helper:
    - `_apply_training_profile_overrides(...)`
  - new manifest helper:
    - `_build_manifest_cmd(...)`
  - summary now records:
    - `manifest.label_mode`, `manifest.force_normalize_labels`, `manifest.force_clean_min_count`.

- `scripts/sweep_graph2d_profile_seeds.py`
  - passes through strict override args:
    - `--force-normalize-labels`
    - `--force-clean-min-count`

## Validation

### 1) Unit tests

```bash
.venv/bin/python -m pytest \
  tests/unit/test_sweep_graph2d_profile_seeds.py \
  tests/unit/test_run_graph2d_pipeline_local_profile.py \
  tests/unit/test_run_graph2d_pipeline_local_manifest_wiring.py \
  tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py \
  tests/unit/test_run_graph2d_pipeline_local_diagnose_strict_wiring.py \
  tests/unit/test_graph2d_seed_gate_summary.py -q
```

Result: `18 passed`.

### 2) Standard seed gate (CI default channel)

```bash
make validate-graph2d-seed-gate
```

Summary:

- `/tmp/graph2d-seed-gate/seed_sweep_summary.json`
- profile: `none`, label mode: `parent_dir`
- strict accuracy mean/min/max: `0.3625 / 0.291667 / 0.433333`
- gate: `passed=true`

### 3) Strict seed gate channel

```bash
make validate-graph2d-seed-gate-strict
```

Summary:

- `/tmp/graph2d-seed-gate-strict/seed_sweep_summary.json`
- profile: `strict_node23_edgesage_v1`, overrides:
  - `force_normalize_labels=false`
  - `force_clean_min_count=0`
- strict accuracy mean/min/max: `0.945833 / 0.941667 / 0.950000`
- gate: `passed=true`

### 4) CI summary renderer sanity check

```bash
.venv/bin/python scripts/ci/summarize_graph2d_seed_gate.py \
  --summary-json /tmp/graph2d-seed-gate/seed_sweep_summary.json \
  --title "Graph2D Seed Gate Local Check"
```

Generated markdown table successfully.

## Notes

- `ezdxf` font cache warnings still appear in local runs:
  - `~/.cache/ezdxf/font_manager_cache.json`
- These warnings did not impact gate pass/fail behavior.
