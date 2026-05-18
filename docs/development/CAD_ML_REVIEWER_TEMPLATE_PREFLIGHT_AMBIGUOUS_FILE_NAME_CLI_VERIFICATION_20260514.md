# CAD ML Reviewer Template Preflight Ambiguous File Name CLI Verification

Date: 2026-05-14

## Scope

Verified the CLI artifact contract for reviewer-template preflight rows that
would otherwise fall back ambiguously by duplicate `file_name`: summary JSON,
Markdown, gap CSV schema/content, fail-under-minimum behavior, Claude Code
read-only review, and focused regression coverage.

## Commands

```bash
<targeted snippets> | claude -p --tools "" --max-budget-usd 0.50
```

Result: Claude Code confirmed the test covered the core summary, Markdown, gap
CSV, and fail-under-minimum contract. It recommended stricter assertions for
summary echo fields, exact blocking reasons, Markdown structure, CSV schema,
empty `relative_path`, and the no-fail flag path; those were added.

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

Result: `62 passed, 7 warnings in 5.28s`.

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
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PREFLIGHT_AMBIGUOUS_FILE_NAME_CLI_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PREFLIGHT_AMBIGUOUS_FILE_NAME_CLI_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- CLI preflight with duplicate base `file_name` values and blank template
  `relative_path` returns `1` under `--fail-under-minimum`.
- The same blocked preflight returns `0` when `--fail-under-minimum` is omitted.
- Summary JSON preserves input path echoes and reports blocked status.
- Summary JSON reports the row as ambiguous rather than unmatched.
- Blocking reasons are pinned to ready-count minimum plus
  `ambiguous_file_name_match_rows`.
- Markdown reports the preflight title, ready count, unmatched count, ambiguous
  fallback count, affected row, and fix guidance.
- Gap CSV preserves the new `ambiguous_file_name_match` column, empty
  `relative_path`, `matched_manifest_row=false`, and reviewer fix guidance.

## Residual Risk

- This is a CLI artifact contract test; it does not create real domain-approved
  labels.
- Release readiness still depends on qualified reviewer labels and later
  threshold tuning.
