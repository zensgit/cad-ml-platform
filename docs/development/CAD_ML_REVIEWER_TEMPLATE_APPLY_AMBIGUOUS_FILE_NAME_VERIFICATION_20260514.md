# CAD ML Reviewer Template Apply Ambiguous File Name Verification

Date: 2026-05-14

## Scope

Verified direct reviewer-template apply blocking for ambiguous file-name
fallback rows, apply audit diagnostics, precise `relative_path` regression
coverage, Claude Code read-only review, and focused release-scorecard
regression coverage.

## Commands

```bash
<targeted snippets> | claude -p --tools "" --max-budget-usd 0.50
```

Result: Claude Code found no focused apply-path regression. It confirmed precise
`relative_path` matches are preserved, ambiguous fallback rows are blocked,
mixed batches report blocked status when any ambiguous row is present, and audit
status mirrors apply behavior. It suggested pinning the mistyped
`relative_path` case; that regression assertion was added.

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

Result: `58 passed, 7 warnings in 5.72s`.

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
  docs/development/CAD_ML_REVIEWER_TEMPLATE_APPLY_AMBIGUOUS_FILE_NAME_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_REVIEWER_TEMPLATE_APPLY_AMBIGUOUS_FILE_NAME_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Clean base manifests still apply reviewer-template rows normally.
- Duplicate base review manifest file names block file-name fallback when a
  returned template omits `relative_path`.
- Duplicate base review manifest file names block file-name fallback when a
  returned template carries a non-matching `relative_path`.
- Precise `relative_path` matches still apply even when another base row has the
  same `file_name`.
- Mixed batches stay blocked when they contain any ambiguous file-name fallback
  row, even if other precise rows apply.
- Blocked ambiguous rows leave base rows unchanged.
- Apply summary exposes `ambiguous_file_name_match_row_count`.
- Apply audit marks ambiguous fallback rows as `ambiguous_file_name_match`.
- Existing reviewer-template apply/preflight, CI wrapper, and workflow tests
  remain in the focused regression set.

## Residual Risk

- This detects ambiguous direct apply rows but does not choose or repair the
  canonical base review manifest row.
- Release readiness still depends on qualified reviewer labels and later
  threshold tuning.
