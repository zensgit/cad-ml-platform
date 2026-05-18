# CAD ML Manufacturing Reviewer Template Preflight Verification

Date: 2026-05-14

## Scope

Verified reviewer-template preflight validation, CLI summary output, return-code
behavior, and targeted regression tests.

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

Result: `31 passed, 7 warnings in 4.85s`.

Warnings: existing third-party `ezdxf`/`pyparsing` deprecation warnings surfaced
during manifest tests. No project-code failure.

```bash
git diff --check
```

Result: passed.

```bash
rg -n "[ \t]+$" \
  scripts/build_manufacturing_review_manifest.py \
  tests/unit/test_build_manufacturing_review_manifest.py \
  docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md \
  docs/development/CAD_ML_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_MANUFACTURING_REVIEWER_TEMPLATE_PREFLIGHT_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Preflight validates ready rows, duplicate row identities, status approval,
  reviewer metadata, source labels, payload labels, and detail labels.
- CLI `--validate-reviewer-template` writes summary JSON.
- `--fail-under-minimum` returns success for a ready template that meets threshold.
- Existing apply, validation, wrapper, and workflow tests still pass.

## Residual Risk

- Preflight verifies structure and review completeness, not domain correctness.
- Domain reviewers still need to provide real approved source, payload, and
  `details.*` labels.
