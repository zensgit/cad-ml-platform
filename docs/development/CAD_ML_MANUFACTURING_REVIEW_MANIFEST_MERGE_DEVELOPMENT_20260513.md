# CAD ML Manufacturing Review Manifest Merge Development

Date: 2026-05-13

## Goal

Close the handoff between human manufacturing review and benchmark evaluation. The
review manifest can now be merged back into a base benchmark manifest only after rows
carry approved reviewed labels.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Added `--merge-approved-review-manifest`.
  - Added required `--base-manifest` input for merge mode.
  - Requires `--output-csv` for the merged benchmark manifest.
  - Matches review rows to base manifest rows by `relative_path` first, then
    `file_name`.
  - Merges only rows that contain reviewed manufacturing source or payload labels.
  - Skips rows whose `review_status` is not approved.
  - Supports `--require-reviewer-metadata` so approved rows without reviewer and
    reviewed timestamp are skipped.
  - Copies reviewed labels and governance metadata:
    - `reviewed_manufacturing_evidence_sources`
    - `reviewed_manufacturing_evidence_payload_json`
    - `review_status`
    - `reviewer`
    - `reviewed_at`
    - `review_notes`
  - Preserves all base manifest columns.
  - Does not copy `suggested_*` columns or `actual_manufacturing_evidence` into the
    release benchmark manifest.
  - Emits merge summary JSON with merged, skipped, missing metadata, and unmatched
    counts.
- Updated `tests/unit/test_build_manufacturing_review_manifest.py`.
  - Covers metadata-required merge blocking.
  - Covers approved-only CLI merge behavior.
  - Covers unapproved row skipping.
  - Covers unmatched approved review rows.
- Updated Phase 6 TODO and review manifest development docs.
- Follow-up CI wiring now publishes the merged reviewed benchmark manifest as a
  scorecard artifact when the optional base manifest path is configured.

## CLI Usage

```bash
python scripts/build_manufacturing_review_manifest.py \
  --merge-approved-review-manifest reports/experiments/<run>/manufacturing_review_manifest.csv \
  --base-manifest data/release/benchmark_manifest.csv \
  --output-csv reports/experiments/<run>/benchmark_manifest.reviewed.csv \
  --summary-json reports/experiments/<run>/manufacturing_review_manifest_merge_summary.json \
  --require-reviewer-metadata \
  --fail-under-minimum
```

## Merge Summary

The merge mode emits a summary shaped for release evidence:

```json
{
  "mode": "merge",
  "status": "merged",
  "base_row_count": 80,
  "review_row_count": 80,
  "approved_review_row_count": 80,
  "merged_row_count": 80,
  "skipped_no_review_content_row_count": 0,
  "skipped_unapproved_review_row_count": 0,
  "skipped_missing_metadata_row_count": 0,
  "unmatched_review_row_count": 0,
  "require_reviewer_metadata": true,
  "blocking_reasons": []
}
```

If no approved reviewed rows are merged, status becomes `blocked` and
`blocking_reasons` includes `no_approved_review_rows_merged`.

## Release Flow

1. Generate a manufacturing review manifest from benchmark results.
2. Reviewers confirm or edit source, payload, and detail labels.
3. Reviewers set `review_status` to an approved value.
4. Reviewers fill `reviewer` and `reviewed_at` when metadata enforcement is enabled.
5. Merge approved reviewed rows into the benchmark manifest.
6. Run the benchmark and forward scorecard against the reviewed benchmark manifest.

## Remaining Work

- Populate the real release benchmark review manifest with domain-approved labels.
- Configure release-labelled evaluation jobs to consume the merged reviewed benchmark
  manifest path after the first real review set is approved.
- Tune source, payload, and detail quality thresholds after the reviewed set is
  stable.
