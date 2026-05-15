# CAD ML B-Rep Golden CI Wiring Verification

Date: 2026-05-12

## Scope

Validated the B-Rep golden manifest CI wrapper, workflow integration, forward
scorecard handoff, and TODO closeout.

## Commands

```bash
bash -n scripts/ci/build_brep_golden_manifest_optional.sh
```

```bash
bash -n scripts/ci/build_forward_scorecard_optional.sh
```

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/ci/check_forward_scorecard_release_gate.py \
  scripts/validate_brep_golden_manifest.py \
  scripts/export_forward_scorecard.py \
  tests/unit/test_brep_golden_manifest_ci_wrapper.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
python3 - <<'PY'
import yaml
from pathlib import Path
payload = yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print(payload['name'])
PY
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_brep_golden_manifest_ci_wrapper.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_validate_brep_golden_manifest.py \
  tests/unit/test_eval_brep_step_dir.py
```

```bash
.venv311/bin/flake8 \
  tests/unit/test_brep_golden_manifest_ci_wrapper.py \
  tests/unit/test_forward_scorecard_release_gate.py
```

```bash
GITHUB_OUTPUT=/private/tmp/cad-ml-brep-golden-ci-smoke-lower.out \
BREP_GOLDEN_MANIFEST_ENABLE=true \
BREP_GOLDEN_MANIFEST_JSON=config/brep_golden_manifest.example.json \
BREP_GOLDEN_MANIFEST_OUTPUT_JSON=/private/tmp/cad-ml-brep-golden-ci-validation-lower.json \
BREP_GOLDEN_MANIFEST_FAIL_ON_NOT_RELEASE_READY=false \
bash scripts/ci/build_brep_golden_manifest_optional.sh
```

```bash
git diff --check
```

## Results

- Bash syntax validation passed for both CI wrappers.
- Python compile passed for touched scripts and tests.
- Workflow YAML parsed successfully and reported `Evaluation Report`.
- Combined targeted pytest passed: `30 passed, 7 warnings in 2.91s`.
- Flake8 passed for the new wrapper tests and touched forward scorecard gate tests.
- Local wrapper smoke wrote:
  - `enabled=true`
  - `validation_status=insufficient_release_samples`
  - `ready_for_release=false`
  - `case_count=1`
  - `release_eligible_count=0`
  - `min_release_samples=50`
- `git diff --check` passed.

## Coverage

- Disabled wrapper mode skips cleanly.
- Example fixture manifest publishes validation evidence but is not release-ready.
- `BREP_GOLDEN_MANIFEST_FAIL_ON_NOT_RELEASE_READY=true` fails an insufficient
  manifest.
- Reduced-floor temp manifest can pass as `release_ready`.
- Forward scorecard wrapper consumes
  `steps.brep_golden_manifest.outputs.eval_summary_json`.

## Notes

- The 7 warnings are existing `ezdxf`/`pyparsing` deprecation warnings from the test
  environment.
- `pytest` is not on the global shell PATH; validation used repo-local `.venv311`.
- Strict B-Rep evaluation itself still requires an OCC-enabled runtime when enabled
  against real STEP/IGES files.
