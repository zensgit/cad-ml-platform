# CAD ML Manufacturing Reviewer Template Preflight Development

Date: 2026-05-14

## Goal

Catch reviewer-template issues before applying them into the full manufacturing
review manifest. The CI path can already apply a filled reviewer template, but
reviewers need a preflight check that reports missing review content, unapproved
rows, missing metadata, missing source/payload/detail labels, and duplicate rows.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Added `validate_reviewer_template_rows`.
  - Added `--validate-reviewer-template`.
  - Added explicit review-status approval for template preflight checks.
  - Counts ready template rows only when a row has:
    - reviewed source labels
    - reviewed payload JSON
    - `details.*` payload labels
    - approved review status
    - reviewer metadata when required
    - no duplicate row identity
  - Reports blocking reasons for:
    - ready template rows below threshold
    - rows with no review content
    - unapproved rows
    - missing reviewer metadata
    - missing source labels
    - missing payload labels
    - missing payload detail labels
    - duplicate row identities
- Updated `tests/unit/test_build_manufacturing_review_manifest.py`.
  - Added direct preflight summary coverage.
  - Added CLI summary and return-code coverage.
- Updated Phase 6 TODO.

## CLI Usage

```bash
python scripts/build_manufacturing_review_manifest.py \
  --validate-reviewer-template reports/benchmark/forward_scorecard/manufacturing_reviewer_template.filled.csv \
  --summary-json reports/benchmark/forward_scorecard/manufacturing_reviewer_template_preflight.json \
  --min-reviewed-samples 30 \
  --require-reviewer-metadata \
  --fail-under-minimum
```

## Release Impact

Reviewer-template quality can now be checked before the apply step. This reduces
CI churn and keeps the release gate focused on approved human labels rather than
CSV formatting or incomplete review rows.

## Remaining Work

- Produce the first real filled reviewer template from domain review.
- Run preflight, then apply, validate, and merge through the existing release flow.
- Tune source, payload, and detail thresholds after the reviewed release set is
  stable.
