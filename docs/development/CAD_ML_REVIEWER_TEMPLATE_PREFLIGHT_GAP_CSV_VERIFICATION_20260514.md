# CAD ML Reviewer Template Preflight Gap CSV Verification

Date: 2026-05-14

## Scope

Verified reviewer-template preflight gap CSV generation, optional forward
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

Result: `36 passed, 7 warnings in 4.42s`.

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
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_REVIEWER_TEMPLATE_PREFLIGHT_GAP_CSV_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Preflight gap CSV lists blocking filled-template rows.
- Duplicate row IDs are represented as machine-readable blockers.
- Ready preflight writes an empty gap CSV with headers.
- Blocked preflight writes actionable gap rows before fail-on-blocked exits.
- Wrapper exposes the gap CSV path as
  `manufacturing_reviewer_template_preflight_gap_csv`.
- Workflow uploads the gap CSV with preflight artifacts.

## Residual Risk

- The CSV validates structure and completeness, not domain correctness.
- Real release readiness still depends on approved source, payload, and
  `details.*` labels from qualified review.
