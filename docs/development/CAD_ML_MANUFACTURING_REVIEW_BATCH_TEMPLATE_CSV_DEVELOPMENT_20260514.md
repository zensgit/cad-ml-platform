# CAD ML Manufacturing Review Batch Template CSV Development

Date: 2026-05-14

## Goal

Make the label-balanced review batch directly editable by reviewers. The batch
CSV identifies bounded work, but reviewers still need a fillable template with
the approved-label fields. This slice emits a batch reviewer template CSV that
combines batch metadata with the existing reviewer template columns.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Adds `REVIEW_BATCH_TEMPLATE_COLUMNS`.
  - Adds `build_review_batch_template_rows`.
  - Adds `--review-batch-template-csv`.
  - Reuses the same label-balanced row selection as `--review-batch-csv`.
  - Preserves editable reviewer fields:
    - `review_status`
    - `reviewer`
    - `reviewed_at`
    - `reviewed_manufacturing_evidence_sources`
    - `reviewed_manufacturing_evidence_payload_json`
    - `review_notes`
  - Preserves suggested source/payload context and `gap_reasons`.
- Updated review handoff Markdown generation.
  - Adds the batch reviewer template to the artifact map.
  - Directs reviewers to use either the batch template or the full template.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds `FORWARD_SCORECARD_MANUFACTURING_REVIEW_BATCH_TEMPLATE_CSV`.
  - Passes `--review-batch-template-csv` during review-manifest validation.
  - Emits `manufacturing_review_batch_template_csv` as a GitHub Actions output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the batch template path.
  - Uploads the batch template with review-manifest validation artifacts.
- Updated targeted tests for:
  - batch template row construction
  - CLI batch template CSV writing
  - forward scorecard wrapper output wiring
  - workflow env and artifact upload wiring
- Updated Phase 6 TODO.

## Default CI Path

```text
reports/benchmark/forward_scorecard/manufacturing_review_batch_template.csv
```

Override with:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEW_BATCH_TEMPLATE_CSV=<path>
```

## Release Impact

This does not approve labels or weaken gates. It removes manual filtering from
the reviewer path: reviewers can receive a bounded batch template, fill only
domain-approved labels, run preflight, and apply approved rows back into the
review manifest.

## Remaining Work

- Assign the generated batch template to qualified manufacturing reviewers.
- Run reviewer-template preflight on returned files.
- Apply approved rows back into the review manifest.
- Tune quality thresholds only after the reviewed release set is stable.
