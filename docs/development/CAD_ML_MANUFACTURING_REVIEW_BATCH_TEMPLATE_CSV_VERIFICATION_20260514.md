# CAD ML Manufacturing Review Batch Template CSV Verification

Date: 2026-05-14

## Scope

Verified batch reviewer template CSV generation, optional forward scorecard
wrapper wiring, workflow artifact upload wiring, and focused regression
coverage.

## Commands

```bash
bash -n scripts/ci/build_forward_scorecard_optional.sh
```

Result: passed.

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

Result: passed.

```bash
.venv311/bin/flake8 \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

Result: passed.

```bash
.venv311/bin/pytest -q \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py
```

Result: `41 passed, 7 warnings in 6.27s`.

Warnings: existing third-party `ezdxf`/`pyparsing` deprecation warnings surfaced
during manifest tests. No project-code failure.

```bash
git diff --check
```

Result: passed.

```bash
rg -n "[ \t]+$" \
  scripts/build_manufacturing_review_manifest.py \
  scripts/ci/build_forward_scorecard_optional.sh \
  .github/workflows/evaluation-report.yml \
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_MANUFACTURING_REVIEW_BATCH_TEMPLATE_CSV_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_MANUFACTURING_REVIEW_BATCH_TEMPLATE_CSV_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Batch template CSV uses the same label-balanced selection as the batch CSV.
- Batch metadata is preserved with editable reviewer template columns.
- Suggested source/payload context and gap reasons remain available in the
  template.
- Build mode can write the batch template next to the other review artifacts.
- Forward scorecard wrapper emits `manufacturing_review_batch_template_csv`.
- Evaluation workflow uploads the batch template with review-manifest validation
  artifacts.

## Residual Risk

- The batch template is a reviewer input surface, not a source of truth.
- Release readiness still depends on qualified reviewers approving source,
  payload, and `details.*` labels, followed by preflight and approved-only apply.
