# CAD ML Forward Scorecard Release Gate Verification

Date: 2026-05-12

## Scope

Validated the new CI wrapper, release gate script, workflow integration, and TODO
closeout for the forward scorecard release-control slice.

## Commands

```bash
PYTHONPYCACHEPREFIX=/private/tmp/cad-ml-pycache .venv311/bin/python -m py_compile \
  scripts/ci/check_forward_scorecard_release_gate.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
bash -n scripts/ci/build_forward_scorecard_optional.sh
```

```bash
.venv311/bin/python -m pytest tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py -q
```

```bash
.venv311/bin/flake8 --max-line-length=100 \
  scripts/ci/check_forward_scorecard_release_gate.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
.venv311/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/evaluation-report.yml', encoding='utf-8'))"
```

```bash
FORWARD_SCORECARD_ENABLE=true \
FORWARD_SCORECARD_OUTPUT_JSON=/private/tmp/cad-forward-scorecard-local.json \
FORWARD_SCORECARD_OUTPUT_MD=/private/tmp/cad-forward-scorecard-local.md \
FORWARD_SCORECARD_RELEASE_GATE_OUTPUT_JSON=/private/tmp/cad-forward-scorecard-local-gate.json \
GITHUB_OUTPUT=/private/tmp/cad-forward-scorecard-local.outputs \
PATH=/Users/chouhua/Downloads/Github/cad-ml-platform/.venv311/bin:$PATH \
bash scripts/ci/build_forward_scorecard_optional.sh
```

```bash
git diff --check
```

## Results

- Python compile passed under `.venv311`.
- Bash syntax validation passed for `scripts/ci/build_forward_scorecard_optional.sh`.
- Targeted pytest passed: `11 passed, 7 warnings in 2.51s`.
- Flake8 passed for the new release gate script and test file.
- Workflow YAML parsed successfully with PyYAML.
- Local wrapper smoke generated `/private/tmp/cad-forward-scorecard-local.json`
  with `overall_status=blocked`; this is expected without real benchmark artifact
  inputs. The disabled gate report wrote `should_fail=false`.
- `git diff --check` passed.

## Test Coverage

- `blocked` plus release label fails.
- `shadow_only` plus release label fails.
- non-release labels do not trigger the gate.
- `benchmark_ready_with_gap` is allowed.
- optional CI wrapper emits ready outputs.
- optional CI wrapper exits non-zero when the gate blocks release.

## Notes

- A direct `python3 -m pytest ...` run used macOS Python 3.9 and failed before the
  tests executed because existing FastAPI router annotations require Python 3.10+.
  The repo-local `.venv311` matches `.python-version` and was used for validation.
- The gate is intentionally status-based. It does not inspect individual component
  thresholds directly; component threshold ownership remains in
  `src/core/benchmark/forward_scorecard.py`.
- The workflow step is not marked `continue-on-error`, so an enabled gate can stop a
  release-labelled run.
