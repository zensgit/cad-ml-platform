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
