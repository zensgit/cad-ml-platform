# CAD ML Manufacturing Review Batch CSV Verification

Date: 2026-05-14

## Scope

Verified manufacturing review batch CSV generation, optional forward scorecard
wrapper wiring, workflow artifact upload wiring, and focused regression
coverage.

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

Result: `40 passed, 7 warnings in 5.02s`.

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
  docs/development/CAD_ML_MANUFACTURING_REVIEW_BATCH_CSV_DEVELOPMENT_20260514.md \
  docs/development/CAD_ML_MANUFACTURING_REVIEW_BATCH_CSV_VERIFICATION_20260514.md
```

Result: no trailing whitespace matches.

## Coverage

- Batch CSV emits only rows with outstanding review gaps.
- Rows are grouped into label-balanced batches.
- Per-label row limits are honored.
- Batch rows expose source, payload, detail, approval, and metadata gap flags.
- Build mode can write batch CSVs next to the existing manifest review artifacts.
- Forward scorecard wrapper emits `manufacturing_review_batch_csv`.
- Evaluation workflow uploads the batch CSV with review-manifest validation
  artifacts.

## Residual Risk

- The batch CSV is an operational reviewer worklist, not a domain approval.
- Release readiness still depends on qualified reviewers approving source,
  payload, and `details.*` labels.
