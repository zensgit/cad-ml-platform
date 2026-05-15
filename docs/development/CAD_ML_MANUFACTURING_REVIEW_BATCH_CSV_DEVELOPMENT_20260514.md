# CAD ML Manufacturing Review Batch CSV Development

Date: 2026-05-14

## Goal

Move the manufacturing review closeout closer to real label population without
fabricating domain labels. The new batch CSV gives reviewers a label-balanced,
machine-readable worklist built from rows that still have source, payload,
detail, approval, or metadata gaps.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Adds `REVIEW_BATCH_COLUMNS`.
  - Adds `build_review_batch_rows`.
  - Adds `--review-batch-csv`.
  - Adds `--max-batch-rows-per-label`.
  - Writes batch CSVs in both build and validate modes.
- Updated review handoff Markdown generation.
  - Adds batch CSV to the artifact map.
  - Directs reviewers to use assignment, gap, context, and batch artifacts
    together.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds `FORWARD_SCORECARD_MANUFACTURING_REVIEW_BATCH_CSV`.
  - Passes `--review-batch-csv` during review-manifest validation.
  - Emits `manufacturing_review_batch_csv` as a GitHub Actions output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the batch CSV path.
  - Uploads the batch CSV with review-manifest validation artifacts.
- Updated targeted tests for:
  - balanced batch row construction
  - CLI batch CSV writing
  - optional forward scorecard wrapper output wiring
  - workflow env and artifact upload wiring
- Updated Phase 6 TODO.

## Batch Semantics

The batch CSV contains only rows with outstanding review gaps.

Rows are grouped by `label_cn`; labels with more gap rows are emitted first.
Within each label group, rows with more gap reasons are prioritized first, then
sorted by stable `row_id`. Each label receives at most
`--max-batch-rows-per-label` rows in the current batch.

## Default CI Path

```text
reports/benchmark/forward_scorecard/manufacturing_review_batch.csv
```

Override with:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEW_BATCH_CSV=<path>
```

## Release Impact

The release blocker remains real reviewed source, payload, and detail labels.
This change makes the first reviewer pass easier to operate: reviewers can pick
a bounded, label-balanced batch while still using the context CSV and reviewer
template for evidence lookup and final label entry.

## Remaining Work

- Run the batch CSV against the real release review manifest.
- Assign batches to qualified manufacturing reviewers.
- Apply the returned reviewer template through preflight and approved-only apply.
- Tune quality thresholds only after the reviewed release set is stable.
