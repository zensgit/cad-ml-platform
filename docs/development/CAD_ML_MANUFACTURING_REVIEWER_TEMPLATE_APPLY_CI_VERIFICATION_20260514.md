# CAD ML Manufacturing Reviewer Template Apply CI Verification

Date: 2026-05-14

## Scope

Verified optional forward scorecard CI wiring for filled reviewer-template apply,
workflow artifact wiring, and targeted regression tests.

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

Result: `29 passed, 7 warnings in 4.27s`.

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
  docs/development/CAD_ML_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CI_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CI_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Wrapper keeps apply outputs disabled when no filled template is configured.
- Wrapper applies a filled reviewer template before review manifest validation.
- Applied manifest becomes the `manufacturing_review_manifest_csv` output.
- Apply summary and applied manifest paths are emitted as GitHub outputs.
- Workflow wires apply path repository variables.
- Workflow uploads apply artifacts only when apply is available.

## Residual Risk

- CI wiring is ready, but it still needs a real filled reviewer template from
  domain review.
- The path intentionally does not auto-approve suggestions.
