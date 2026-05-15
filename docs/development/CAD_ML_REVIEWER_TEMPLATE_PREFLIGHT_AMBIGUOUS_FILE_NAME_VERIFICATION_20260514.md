# CAD ML Reviewer Template Preflight Ambiguous File Name Verification

Date: 2026-05-14

## Scope

Verified reviewer-template preflight blocking for ambiguous file-name fallback
rows, Markdown and gap CSV diagnostics, precise `relative_path` regression
coverage, Claude Code read-only review, and focused release-scorecard
regression coverage.

## Commands

```bash
<targeted snippets> | claude -p --tools "" --max-budget-usd 0.50
```

Result: Claude Code found no focused regression in duplicate-base precedence or
counter accounting. It confirmed ambiguous rows are not double-counted as
unmatched rows and called out one redundant ready-row guard; the redundancy was
removed.

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

Result: `61 passed, 7 warnings in 5.52s`.

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
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PREFLIGHT_AMBIGUOUS_FILE_NAME_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PREFLIGHT_AMBIGUOUS_FILE_NAME_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Clean base manifests still preflight reviewer-template rows normally.
- Duplicate base review manifest file names block file-name fallback when a
  returned template omits `relative_path`.
- Duplicate base review manifest file names block file-name fallback when a
  returned template carries a non-matching `relative_path`.
- Precise `relative_path` matches still count as ready even when another base
  row has the same `file_name`.
- Ambiguous rows are counted separately from unmatched manifest rows.
- Duplicate base row identities keep precedence over ambiguous file-name
  fallback reporting.
- Preflight summary exposes `ambiguous_file_name_match_row_count`.
- Preflight Markdown reports ambiguous fallback row counts and fixes.
- Preflight gap CSV marks ambiguous fallback rows with
  `ambiguous_file_name_match=true`.

## Residual Risk

- This detects ambiguous preflight rows but does not choose or repair the
  canonical base review manifest row.
- Release readiness still depends on qualified reviewer labels and later
  threshold tuning.
