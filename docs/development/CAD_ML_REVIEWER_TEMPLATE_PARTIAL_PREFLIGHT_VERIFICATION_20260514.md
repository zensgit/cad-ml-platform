# CAD ML Reviewer Template Partial Preflight Verification

Date: 2026-05-14

## Scope

Verified reviewer-template partial preflight threshold wiring, workflow env
coverage, and targeted regression behavior for incremental manufacturing review
batches.

## Commands

```bash
claude --version
```

Result: `2.1.141 (Claude Code)`.

```bash
bash -n scripts/ci/build_forward_scorecard_optional.sh
```

Result: passed.

```bash
PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py \
  tests/unit/test_build_manufacturing_review_manifest.py
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

Result: `42 passed, 7 warnings in 6.01s`.

Warnings: existing third-party `ezdxf`/`pyparsing` deprecation warnings surfaced
during manifest tests. No project-code failure.

```bash
git diff --check
```

Result: passed.

```bash
rg -n "[ \t]+$" \
  scripts/ci/build_forward_scorecard_optional.sh \
  .github/workflows/evaluation-report.yml \
  tests/unit/test_forward_scorecard_release_gate.py \
  tests/unit/test_evaluation_report_workflow_forward_scorecard.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PARTIAL_PREFLIGHT_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PARTIAL_PREFLIGHT_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Reviewer-template preflight can use a lower min-ready-row threshold than the
  final release manifest validation.
- A one-row approved reviewer template can pass preflight and apply.
- The final review manifest remains blocked when release minimums are still not
  met.
- Workflow env exposes the partial preflight threshold.
- Existing reviewer-template apply, audit, and manifest validation behavior still
  passes targeted regression tests.

## Residual Risk

- Partial preflight only checks returned template correctness; it is not a
  release readiness signal.
- Release readiness still depends on enough qualified reviewer-approved source,
  payload, and `details.*` labels.
