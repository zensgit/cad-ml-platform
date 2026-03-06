# Parallel Features Signal/Strict Validation (2026-03-06)

## Scope

This continuation implemented three parallel improvements:

1. PR comment red/yellow/green status block for Graph2D signals.
2. Train-sweep best run script export (`run_best_recipe.sh`).
3. `workflow_dispatch` + strict-mode enforcement for review-pack gate.

## Changes

### 1) PR comment signal lights

Updated:
- `.github/workflows/evaluation-report.yml`

Enhancements in `Comment PR with results`:
- Added `Graph2D Signal Lights` section with R/Y/G semantics:
  - Review Pack: `⚪/🟡/🟢`
  - Review Gate: `⚪/🔴/🟢`
  - Train Sweep: `⚪/🟡/🔴/🟢`
- Existing detail lines are preserved and now include best run script path.
- Added `Graph2D Review Gate Strict` row to show strict-mode decision path:
  - `strict`, `should_fail`, `reason`

### 2) Train-sweep best run script export

Updated:
- `scripts/sweep_graph2d_train_recipes.py`

New outputs:
- `recommended_graph2d_train.env` (already available, preserved)
- `run_best_recipe.sh` (new, executable)

New CLI args:
- `--recommended-env-out`
- `--best-run-script-out`

Summary JSON (`train_recipe_sweep_summary.json`) now includes:
- `recommended_env_file`
- `best_run_script`
- `best_args`

### 3) Review gate strict mode + dispatch overrides

Updated:
- `.github/workflows/evaluation-report.yml`

New `workflow_dispatch` input:
- `review_gate_strict`

New env:
- `GRAPH2D_REVIEW_PACK_GATE_STRICT` (default false)

New workflow steps:
- `Evaluate Graph2D review gate strict mode (optional)`
  - Computes `should_fail` and `reason` outputs.
  - Always non-blocking (`continue-on-error: true`) so report/comment steps still run.
- `Fail workflow when Graph2D review gate strict check requires blocking`
  - Runs after PR comment step.
  - Fails the job only when `should_fail == 'true'`.
- Job summary now includes:
  - `headline`
  - `strict_mode`
  - `strict_should_fail`
  - `strict_reason`

Also added optional gate annotation step earlier:
- `Emit Graph2D review gate annotations (optional)`

## Tests Updated/Added

Updated:
- `tests/unit/test_sweep_graph2d_train_recipes.py`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

Existing/new tests executed:
- `tests/unit/test_emit_graph2d_review_pack_gate_annotations.py`
- `tests/unit/test_graph2d_review_pack_gate_check.py`
- `tests/unit/test_graph2d_parallel_make_targets.py`

## Validation

### Pytest

```bash
pytest -q \
  tests/unit/test_emit_graph2d_review_pack_gate_annotations.py \
  tests/unit/test_graph2d_review_pack_gate_check.py \
  tests/unit/test_sweep_graph2d_train_recipes.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_graph2d_parallel_make_targets.py
```

Result: `14 passed`.

Note:
- One warning remained from third-party dependency import deprecation (`python_multipart`),
  not from changed project code paths.

### Flake8

```bash
flake8 \
  scripts/sweep_graph2d_train_recipes.py \
  scripts/ci/check_graph2d_review_pack_gate.py \
  scripts/ci/emit_graph2d_review_pack_gate_annotations.py \
  tests/unit/test_sweep_graph2d_train_recipes.py \
  tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  tests/unit/test_graph2d_review_pack_gate_check.py \
  tests/unit/test_emit_graph2d_review_pack_gate_annotations.py \
  tests/unit/test_graph2d_parallel_make_targets.py \
  --max-line-length=100
```

Result: pass.

### Workflow YAML parse

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
p = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(p.read_text(encoding='utf-8'))
print('ok')
PY
```

Result: `ok`.

## End-to-End Dispatch Validation (Strict Toggle)

To avoid dependency on repository variable configuration, workflow dispatch now supports:
- `review_pack_input_csv`

Fixture used:
- `tests/fixtures/ci/graph2d_review_pack_input.csv`

Dispatch runs on commit `8fe0383`:

1. strict disabled (`review_gate_strict=false`)
   - Run: `22743533017`
   - URL: `https://github.com/zensgit/cad-ml-platform/actions/runs/22743533017`
   - Result: `success`
   - Key step conclusions:
     - `Build hybrid rejection review pack (optional)`: `success`
     - `Check Graph2D review-pack gate (optional)`: `success` (gate script runs with `continue-on-error`)
     - `Evaluate Graph2D review gate strict mode (optional)`: `success`
     - `Fail workflow when Graph2D review gate strict check requires blocking`: `skipped`

2. strict enabled (`review_gate_strict=true`)
   - Run: `22743533942`
   - URL: `https://github.com/zensgit/cad-ml-platform/actions/runs/22743533942`
   - Result: `failure` (expected)
   - Key step conclusions:
     - `Build hybrid rejection review pack (optional)`: `success`
     - `Check Graph2D review-pack gate (optional)`: `success` (non-blocking gate evaluation)
     - `Evaluate Graph2D review gate strict mode (optional)`: `success`
     - `Fail workflow when Graph2D review gate strict check requires blocking`: `failure`
   - Failed step log includes:
     - `gate status is 'failed'`
     - `Failure reason: gate_failed_under_strict_mode`

## Make Target Automation

Added script:
- `scripts/ci/dispatch_graph2d_review_gate_strict_e2e.py`

Added make target:
- `graph2d-review-pack-gate-strict-e2e`

Default behavior:
1. Dispatch `evaluation-report.yml` with `strict=false` and wait for completion.
2. Dispatch `evaluation-report.yml` with `strict=true` and wait for completion.
3. Assert conclusions are `success` then `failure`.
4. Emit JSON summary artifact.

Real run via make target:

```bash
make graph2d-review-pack-gate-strict-e2e
```

Observed runs:
- strict=false: `22744089501` → `success`
- strict=true: `22744124251` → `failure` (expected)

Generated summary artifact:
- `reports/experiments/20260306/graph2d_review_pack_gate_strict_e2e.json`
