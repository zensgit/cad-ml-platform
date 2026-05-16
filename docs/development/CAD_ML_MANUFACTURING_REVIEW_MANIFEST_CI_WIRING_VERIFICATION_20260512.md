# CAD ML Manufacturing Review Manifest CI Wiring Verification

Date: 2026-05-12

## Scope

Validated optional CI/release wiring for manufacturing review manifest validation,
including wrapper outputs, workflow artifact upload, and fail-on-blocked behavior.

## Commands

```bash
bash -n scripts/ci/build_forward_scorecard_optional.sh
```

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
.venv311/bin/flake8 \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
git diff --check
```

## Results

- Bash syntax check passed for the forward scorecard wrapper.
- Python compile passed for the review manifest script and touched tests.
- Flake8 passed for touched tests.
- Targeted pytest passed: `12 passed, 7 warnings in 4.04s`.
- `git diff --check` passed.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings imported
  through existing DXF evaluation helpers.

## Verified Behavior

- The wrapper validates a configured manufacturing review manifest.
- The wrapper emits GitHub outputs for review manifest availability, CSV path, summary
  JSON path, and validation status.
- A release-label-ready manifest keeps the scorecard wrapper green.
- A blocked manifest fails the wrapper when
  `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_FAIL_ON_BLOCKED=true`.
- The workflow exposes repository-variable wiring for path, summary path, minimum
  reviewed samples, and fail-on-blocked mode.
- The workflow uploads the validation summary after the forward scorecard and
  manufacturing evidence summary artifacts.
