# CAD ML Manufacturing Review Manifest Governance Verification

Date: 2026-05-13

## Scope

Validated review-status gating, optional reviewer metadata enforcement, CI wrapper
wiring, workflow variable wiring, and scorecard propagation of review manifest
governance fields.

## Commands

```bash
bash -n scripts/ci/build_forward_scorecard_optional.sh
```

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/build_manufacturing_review_manifest.py \
  scripts/export_forward_scorecard.py \
  src/core/benchmark/forward_scorecard.py \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
.venv311/bin/flake8 \
  scripts/build_manufacturing_review_manifest.py \
  scripts/export_forward_scorecard.py \
  src/core/benchmark/forward_scorecard.py \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
.venv311/bin/pytest -q \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

```bash
git diff --check
```

## Results

- Bash syntax check passed for the forward scorecard wrapper.
- Python compile passed for touched scripts, scorecard helper, and tests.
- Flake8 passed for touched scripts, scorecard helper, and tests.
- Targeted pytest passed: `28 passed, 7 warnings in 3.83s`.
- `git diff --check` passed.
- Warnings are existing `ezdxf`/`pyparsing` dependency deprecation warnings imported
  through existing DXF evaluation helpers.

## Verified Behavior

- Generated review manifests include `review_status`, `reviewer`, and `reviewed_at`.
- Generated templates default to `needs_human_review`.
- Prefilled suggestions are not counted as release-reviewed labels unless review
  status is approved.
- Approved rows count toward source, payload, and detail reviewed sample thresholds.
- `--require-reviewer-metadata` blocks approved rows that lack reviewer or reviewed
  timestamp metadata.
- The CI wrapper exposes reviewer metadata enforcement through environment variables.
- The workflow exposes the reviewer metadata enforcement repository variable.
- The forward scorecard preserves the governance counts inside
  `review_manifest_validation`.
