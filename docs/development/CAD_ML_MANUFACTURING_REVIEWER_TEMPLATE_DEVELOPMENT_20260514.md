# CAD ML Manufacturing Reviewer Template Development

Date: 2026-05-14

## Goal

Move the manufacturing review closeout from reporting into actionable patch input.
The assignment Markdown explains what to review, but reviewers also need a compact
CSV they can edit and return without touching the full benchmark manifest.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Added `REVIEWER_TEMPLATE_COLUMNS`.
  - Added `build_reviewer_template_rows`.
  - Added `--reviewer-template-csv`.
  - Build and validate modes can now emit a reviewer fill-template CSV.
  - Template rows are generated only for rows with remaining review gaps.
  - Existing partial reviewed values are preserved so reviewers can continue work.
  - Blank review status is normalized to `needs_human_review` in the template.
  - Suggested source and payload fields remain separate from reviewed fields, so
    suggestions are not treated as approved labels.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_CSV`.
  - Passes `--reviewer-template-csv` during review manifest validation.
  - Emits `manufacturing_reviewer_template_csv` as a GitHub output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the reviewer template path.
  - Uploads the template CSV with the review validation artifact bundle.
- Updated tests for:
  - template row generation
  - CLI template CSV generation
  - optional forward scorecard output wiring
  - workflow artifact upload wiring
- Updated Phase 6 TODO.

## Template Columns

The template includes reviewer-editable fields first:

- `review_status`
- `reviewer`
- `reviewed_at`
- `reviewed_manufacturing_evidence_sources`
- `reviewed_manufacturing_evidence_payload_json`
- `review_notes`

It also includes row identity, suggestions, and `gap_reasons` for context.

## CLI Usage

```bash
python scripts/build_manufacturing_review_manifest.py \
  --validate-manifest reports/experiments/<run>/manufacturing_review_manifest.csv \
  --summary-json reports/benchmark/forward_scorecard/manufacturing_review_manifest_validation.json \
  --progress-md reports/benchmark/forward_scorecard/manufacturing_review_manifest_progress.md \
  --gap-csv reports/benchmark/forward_scorecard/manufacturing_review_manifest_gaps.csv \
  --assignment-md reports/benchmark/forward_scorecard/manufacturing_review_assignment.md \
  --reviewer-template-csv reports/benchmark/forward_scorecard/manufacturing_reviewer_template.csv \
  --min-reviewed-samples 30 \
  --require-reviewer-metadata
```

## Release Impact

Release closeout now has a full review loop:

- JSON summary for gates.
- Progress Markdown for status.
- Gap CSV for filtering.
- Assignment Markdown for batching.
- Reviewer template CSV for patch input.

The remaining release work is now a domain-review task, not a tooling task.

## Remaining Work

- Have domain reviewers fill the template with real approved source, payload, and
  `details.*` labels.
- Merge approved template rows into the benchmark manifest.
- Re-run validation until the gap CSV and reviewer template contain only headers.
