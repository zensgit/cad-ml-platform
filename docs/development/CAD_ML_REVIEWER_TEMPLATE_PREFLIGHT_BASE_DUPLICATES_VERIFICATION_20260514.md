# CAD ML Reviewer Template Preflight Base Duplicates Verification

Date: 2026-05-14

## Scope

Verified duplicate-base-manifest diagnostics in reviewer-template preflight,
Markdown reporting, Claude Code read-only review, and focused regression
coverage.

## Commands

```bash
<targeted snippets> | claude -p --tools "" --max-budget-usd 0.50
```

Result: Claude Code found no behavior regression. It confirmed the no-base
default path remains unchanged and duplicate supplied base manifests block with
`base_manifest_duplicate_rows`.

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

Result: `50 passed, 7 warnings in 6.26s`.

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
  tests/unit/test_build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PREFLIGHT_BASE_DUPLICATES_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PREFLIGHT_BASE_DUPLICATES_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Duplicate supplied base manifest row identities block with
  `base_manifest_duplicate_rows`.
- Summary JSON exposes duplicate base identity count and identifiers.
- Preflight Markdown exposes duplicate base identity count and duplicate IDs.
- No-base preflight remains duplicate count `0`.
- Existing manifest-match, empty-base, wrapper, and workflow tests remain in the
  focused regression set.

## Residual Risk

- This detects ambiguous base manifests but does not choose or repair the
  canonical row.
- Release readiness still depends on qualified reviewer labels and later
  threshold tuning.
