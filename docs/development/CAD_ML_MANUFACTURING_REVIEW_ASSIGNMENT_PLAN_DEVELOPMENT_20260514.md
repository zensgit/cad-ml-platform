# CAD ML Manufacturing Review Assignment Plan Development

Date: 2026-05-14

## Goal

Make the manufacturing review backlog easier to assign. The gap CSV is filterable,
but release reviewers still need a compact Markdown plan that groups remaining
work by part label and gap type before assigning rows.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Added `build_review_assignment_markdown`.
  - Added `--assignment-md`.
  - Added `--max-assignment-rows-per-label`.
  - Build and validate modes can now emit a label-grouped assignment plan.
  - The assignment plan includes:
    - validation status
    - total gap row count
    - remaining source, payload, and detail label counts
    - per-label gap buckets
    - per-label review batches with row id, status, gaps, and suggested sources
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds `FORWARD_SCORECARD_MANUFACTURING_REVIEW_ASSIGNMENT_MD`.
  - Passes `--assignment-md` to review manifest validation.
  - Emits `manufacturing_review_assignment_md` as a GitHub output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the assignment Markdown path.
  - Uploads the assignment Markdown with the manufacturing review validation
    artifact bundle.
- Updated tests for:
  - assignment Markdown grouping by label and gap type
  - CLI assignment Markdown generation
  - forward scorecard wrapper output wiring
  - workflow artifact upload wiring
- Updated Phase 6 TODO.

## CLI Usage

```bash
python scripts/build_manufacturing_review_manifest.py \
  --validate-manifest reports/experiments/<run>/manufacturing_review_manifest.csv \
  --summary-json reports/benchmark/forward_scorecard/manufacturing_review_manifest_validation.json \
  --progress-md reports/benchmark/forward_scorecard/manufacturing_review_manifest_progress.md \
  --gap-csv reports/benchmark/forward_scorecard/manufacturing_review_manifest_gaps.csv \
  --assignment-md reports/benchmark/forward_scorecard/manufacturing_review_assignment.md \
  --min-reviewed-samples 30 \
  --require-reviewer-metadata
```

## Release Impact

The review closeout now has three complementary outputs:

- Summary JSON for gates and scorecard ingestion.
- Gap CSV for filtering and batch tracking.
- Assignment Markdown for label-level reviewer batching.

This improves operational readiness without making any unsupported claim that
machine suggestions are domain-approved labels.

## Remaining Work

- Populate real reviewed source, payload, and detail labels for the release
  benchmark set.
- Use the assignment Markdown to split rows by `label_cn` and gap type.
- Re-run the validator until the assignment plan reports zero gap rows.
