# CAD ML Manufacturing Reviewer Template Preflight CI Verification

Date: 2026-05-14

## Scope

Verified optional forward scorecard CI wiring for reviewer-template preflight,
artifact upload wiring, and targeted regression tests.

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

Result: `33 passed, 7 warnings in 5.07s`.

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
  docs/development/CAD_ML_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_CI_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_CI_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Wrapper reports no preflight when no filled template is configured.
- Wrapper runs preflight before template apply.
- Template apply only runs after ready preflight.
- Blocked preflight can fail the wrapper through the fail-on-blocked flag.
- Preflight summary path and status are emitted as GitHub outputs.
- Workflow uploads preflight summary only when available.

## Residual Risk

- Preflight checks structure and completeness, not domain correctness.
- Release readiness still depends on real domain-approved source, payload, and
  `details.*` labels.
