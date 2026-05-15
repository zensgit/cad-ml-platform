# CAD ML B-Rep Golden Manifest Verification

Date: 2026-05-12

## Scope

Validated the B-Rep golden manifest contract, example manifest, CLI release gate,
and manifest-driven strict evaluator wiring.

## Commands

```bash
PYTHONPYCACHEPREFIX=/private/tmp/cad-ml-pycache .venv311/bin/python -m py_compile \
  scripts/validate_brep_golden_manifest.py \
  scripts/eval_brep_step_dir.py \
  tests/unit/test_validate_brep_golden_manifest.py \
  tests/unit/test_eval_brep_step_dir.py
```

```bash
.venv311/bin/flake8 --max-line-length=100 \
  scripts/validate_brep_golden_manifest.py \
  scripts/eval_brep_step_dir.py \
  tests/unit/test_validate_brep_golden_manifest.py \
  tests/unit/test_eval_brep_step_dir.py
```

```bash
.venv311/bin/python -m pytest \
  tests/unit/test_validate_brep_golden_manifest.py \
  tests/unit/test_eval_brep_step_dir.py -q
```

```bash
.venv311/bin/python -m pytest \
  tests/unit/test_validate_brep_golden_manifest.py \
  tests/unit/test_eval_brep_step_dir.py \
  tests/unit/test_benchmark_realdata_signals.py \
  tests/unit/test_benchmark_realdata_scorecard.py \
  tests/unit/test_forward_scorecard.py -q
```

```bash
.venv311/bin/python scripts/validate_brep_golden_manifest.py \
  --manifest config/brep_golden_manifest.example.json \
  --output-json /private/tmp/brep-golden-manifest-example-validation.json
```

```bash
.venv311/bin/python scripts/validate_brep_golden_manifest.py \
  --manifest config/brep_golden_manifest.example.json \
  --output-json /private/tmp/brep-golden-manifest-example-gate.json \
  --fail-on-not-release-ready
```

```bash
git diff --check
```

## Results

- Python compile passed under `.venv311`.
- Flake8 passed for the new manifest validator and touched evaluator tests.
- Manifest/evaluator/downstream compatibility suite passed:
  `31 passed, 7 warnings in 1.97s`.
- Example manifest validation returned `insufficient_release_samples`, as intended.
- Example manifest with `--fail-on-not-release-ready` returned exit code `1`, as
  intended for a non-release fixture manifest.
- `git diff --check` passed.

## Coverage

- 50 real temp-file cases produce `release_ready`.
- Fixture cases do not count toward release readiness.
- Fixture cases marked `release_eligible=true` are invalid.
- Duplicate IDs and missing files are invalid.
- Expected parse failures require `expected_failure_reason`.
- CLI writes JSON validation reports.
- `--fail-on-not-release-ready` can block insufficient manifests.
- `scripts/eval_brep_step_dir.py --manifest` resolves case paths from manifest root.

## Notes

- `config/brep_golden_manifest.example.json` is intentionally not release-ready.
- This slice does not add real proprietary/vendor STEP/IGES files to the repository.
  The remaining blocker is data acquisition and licensing, not evaluator mechanics.
