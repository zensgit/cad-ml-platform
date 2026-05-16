# CAD ML Reviewer Template Apply Duplicate Base Block Verification

Date: 2026-05-14

## Scope

Verified direct reviewer-template apply blocking for duplicate base review
manifest identities, apply audit diagnostics, Claude Code read-only review, and
focused regression coverage.

## Commands

```bash
<targeted snippets> | claude -p --tools "" --max-budget-usd 0.50
```

Result: Claude Code found no clean-base behavior regression and confirmed that
duplicate-base direct apply is blocked before any writeback.

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

Result: `52 passed, 7 warnings in 6.80s`.

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
  docs/development/CAD_ML_REVIEWER_TEMPLATE_APPLY_DUPLICATE_BASE_BLOCK_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_REVIEWER_TEMPLATE_APPLY_DUPLICATE_BASE_BLOCK_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Clean base manifests still apply reviewer-template rows normally.
- Duplicate base review manifest identities block direct apply.
- Blocked direct apply leaves base rows unchanged.
- Apply summary exposes duplicate base identity count and identifiers.
- Apply audit marks eligible rows as `blocked_duplicate_base_manifest`.
- Existing preflight, CI wrapper, and workflow tests remain in the focused
  regression set.

## Residual Risk

- This detects ambiguous base manifests but does not choose or repair the
  canonical row.
- Release readiness still depends on qualified reviewer labels and later
  threshold tuning.
