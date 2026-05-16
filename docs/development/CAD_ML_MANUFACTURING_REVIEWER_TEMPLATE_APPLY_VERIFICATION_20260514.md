# CAD ML Manufacturing Reviewer Template Apply Verification

Date: 2026-05-14

## Scope

Verified the reviewer template apply path, CLI integration, post-apply validation,
and targeted regression tests.

## Commands

```bash
bash -n scripts/ci/build_forward_scorecard_optional.sh
```

Result: passed.

```bash
/usr/bin/env PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache \
  python3 -m py_compile \
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

Result: `27 passed, 7 warnings in 3.74s`.

Warnings: existing third-party `ezdxf`/`pyparsing` deprecation warnings surfaced
during manifest tests. No project-code failure.

```bash
git diff --check
```

Result: passed.

```bash
rg -n "[ \t]+$" \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_build_manufacturing_review_manifest.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- `apply_reviewer_template_rows` applies only approved template rows with reviewed
  content.
- Optional reviewer metadata is enforced before a template row can be applied.
- Unmatched template rows are counted and skipped.
- CLI `--apply-reviewer-template` writes a full updated review manifest.
- Summary JSON includes `post_apply_validation`.
- `--fail-under-minimum` works against post-apply release readiness.

## Residual Risk

- This slice does not create domain-approved labels.
- The release benchmark still depends on reviewers filling the template with real
  source, payload, and `details.*` labels.
