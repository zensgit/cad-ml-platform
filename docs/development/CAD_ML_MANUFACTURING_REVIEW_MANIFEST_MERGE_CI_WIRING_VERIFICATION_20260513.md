# CAD ML Manufacturing Review Manifest Merge CI Wiring Verification

Date: 2026-05-13

## Scope

Validated optional CI/release wiring for approved-only manufacturing review manifest
merge outputs. The checks cover wrapper execution, GitHub outputs, workflow upload
wiring, merge failure gating, and artifact path behavior.

## Commands

```bash
bash -n scripts/ci/build_forward_scorecard_optional.sh
```

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
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
- Python compile passed for touched tests.
- Flake8 passed for touched tests.
- Targeted pytest passed: `14 passed, 7 warnings in 4.06s`.
- `git diff --check` passed.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings imported
  through existing DXF evaluation helpers.

## Verified Behavior

- The wrapper runs the approved-only merge when review manifest and base benchmark
  manifest paths are both configured.
- The wrapper emits merged benchmark manifest and merge summary outputs only when
  merge status is `merged`.
- The wrapper reports non-ready merge statuses, including `missing_base_manifest`,
  `missing_review_manifest`, and `blocked`.
- `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_FAIL_ON_BLOCKED=true`
  fails the wrapper when merge status is not `merged`.
- The workflow exposes repository-variable wiring for base manifest, merged manifest,
  merge summary, and merge fail-on-blocked mode.
- The workflow uploads the merged reviewed benchmark manifest and merge summary after
  review manifest validation and before downstream operator-adoption artifacts.
