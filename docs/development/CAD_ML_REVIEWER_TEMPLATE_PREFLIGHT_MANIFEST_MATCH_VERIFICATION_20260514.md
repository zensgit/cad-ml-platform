# CAD ML Reviewer Template Preflight Manifest Match Verification

Date: 2026-05-14

## Scope

Verified optional base-manifest matching for reviewer-template preflight,
handoff command parity, CI wrapper preflight wiring, Claude Code read-only
review, and focused regression coverage.

## Commands

```bash
<targeted snippets> | claude -p --tools "" --max-budget-usd 0.50
```

Result: Claude Code found no default-path behavior regression and confirmed the
wrapper passes the current review manifest into preflight.

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

Result: `46 passed, 7 warnings in 5.83s`.

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
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PREFLIGHT_MANIFEST_MATCH_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PREFLIGHT_MANIFEST_MATCH_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Preflight remains backward-compatible when no base manifest is provided.
- No-base preflight gap rows mark manifest matching as `not_checked`.
- Preflight blocks otherwise valid rows that do not match the supplied review
  manifest.
- Preflight Markdown and gap CSV expose unmatched row diagnostics.
- CLI preflight accepts `--base-manifest`.
- CI wrapper passes the active review manifest into preflight before apply.
- Generated handoff preflight commands include `--base-manifest` when available.

## Residual Risk

- A header-only or truncated base manifest will intentionally block returned
  reviewer rows as unmatched.
- Release readiness still depends on real approved labels from qualified
  reviewers; this slice only makes row identity validation earlier.
