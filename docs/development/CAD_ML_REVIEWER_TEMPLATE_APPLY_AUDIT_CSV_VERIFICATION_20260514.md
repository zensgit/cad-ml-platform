# CAD ML Reviewer Template Apply Audit CSV Verification

Date: 2026-05-14

## Scope

Verified reviewer-template apply audit CSV generation, optional forward
scorecard wrapper wiring, workflow artifact upload wiring, and targeted
regression coverage.

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

Result: `37 passed, 7 warnings in 5.07s`.

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
  docs/development/CAD_ML_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Apply audit CSV lists every filled-template row.
- Applied, skipped, and unmatched outcomes are represented explicitly.
- CLI apply mode writes audit CSV when requested.
- Wrapper exposes the audit CSV as
  `manufacturing_reviewer_template_apply_audit_csv`.
- Workflow uploads the audit CSV with apply artifacts.

## Residual Risk

- The audit CSV records apply mechanics, not domain correctness.
- Real release readiness still depends on approved source, payload, and
  `details.*` labels from qualified review.
