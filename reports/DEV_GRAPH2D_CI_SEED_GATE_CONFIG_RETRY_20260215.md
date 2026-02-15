# DEV_GRAPH2D_CI_SEED_GATE_CONFIG_RETRY_20260215

## Scope

Implemented the requested `1+2+3` package:

1. Integrate Graph2D seed stability gate into CI.
2. Externalize gate profile/seeds/thresholds as config.
3. Add automatic retry for failed seed runs.

## Code Changes

### 1) CI integration

- Added a CI step in `.github/workflows/ci.yml` (Python 3.11 lane):
  - run `make validate-graph2d-seed-gate`
  - upload `/tmp/graph2d-seed-gate-ci.log` artifact.

- Added Make target in `Makefile`:
  - `validate-graph2d-seed-gate`
  - invokes `scripts/sweep_graph2d_profile_seeds.py --config ${GRAPH2D_SEED_GATE_CONFIG:-config/graph2d_seed_gate.yaml}`.

### 2) Config externalization

- Added `config/graph2d_seed_gate.yaml` with section `graph2d_seed_sweep`.
- Added `--config` support to `scripts/sweep_graph2d_profile_seeds.py`:
  - reads YAML defaults from section `graph2d_seed_sweep`
  - supports key normalization (`kebab-case` -> `snake_case`).

Current CI-oriented defaults:

- `dxf_dir: data/synthetic_v2`
- `training_profile: none`
- `manifest_label_mode: parent_dir`
- `seeds: 7,21`
- `diagnose_max_files: 120`
- `max_samples: 120`
- `min_label_confidence: 0.0`
- gate thresholds:
  - `min_strict_accuracy_mean: 0.25`
  - `min_strict_accuracy_min: 0.20`
  - `require_all_ok: true`

### 3) Retry / fallback behavior

- Added `--retry-failures` and `--retry-backoff-seconds` in `scripts/sweep_graph2d_profile_seeds.py`.
- Implemented `_run_with_retries(...)`:
  - retries per seed run up to configured attempts
  - returns status/error/attempt count/return code for traceability.
- Added row fields:
  - `attempts`, `return_code`.
- Added summary fields:
  - `num_retried_runs`, retry settings, and gate metadata.

## Additional robustness fix

- Updated `scripts/run_graph2d_pipeline_local.py`:
  - added `--manifest-label-mode` (`filename` / `parent_dir`)
  - extracted `_build_manifest_cmd(...)` for explicit command wiring
  - recorded manifest label mode in `pipeline_summary.json`.

Rationale:
- Strict profile (`strict_node23_edgesage_v1`) enforces normalization and collapses `synthetic_v2` parent-dir labels into a single class (`other`), producing misleading near-1.0 scores.
- CI gate now uses `training_profile=none` + `parent_dir` labels to keep a meaningful multi-class signal.

## Tests

Added/updated unit tests:

- `tests/unit/test_sweep_graph2d_profile_seeds.py`
  - YAML config defaults parsing
  - retry success after one failure
  - retry final failure after max attempts
- `tests/unit/test_run_graph2d_pipeline_local_manifest_wiring.py`
  - validates `--label-mode parent_dir` manifest command wiring.

Validation:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_sweep_graph2d_profile_seeds.py \
  tests/unit/test_run_graph2d_pipeline_local_manifest_wiring.py \
  tests/unit/test_run_graph2d_pipeline_local_profile.py \
  tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py \
  tests/unit/test_run_graph2d_pipeline_local_diagnose_strict_wiring.py -q
```

Result: `15 passed`.

## Runtime Verification

### A) Retry behavior (failure path)

Command:

```bash
.venv/bin/python scripts/sweep_graph2d_profile_seeds.py \
  --config config/graph2d_seed_gate.yaml
```

During early iteration (before `manifest_label_mode` + `min_label_confidence` fix), each seed failed with `no_rows_after_filter`, retried once, and gate failed with non-zero exit.
This verified retry + gate fail path.

### B) Final gate run via Make target

Command:

```bash
make validate-graph2d-seed-gate
```

Artifact root:

- `/tmp/graph2d_profile_seed_sweep_20260215_223401`

Observed strict accuracies:

- seed `7`: `0.2916667`
- seed `21`: `0.4333333`

Aggregate:

- mean: `0.3625`
- min: `0.2916667`
- gate: `passed=true` (`mean>=0.25`, `min>=0.20`, `require_all_ok=true`).

## Notes

- Repeated `ezdxf` cache warnings were observed:
  - `~/.cache/ezdxf/font_manager_cache.json`
- These warnings did not affect gate completion or pass/fail decisions.
