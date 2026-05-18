# CAD ML Manufacturing Reviewer Template Apply Development

Date: 2026-05-14

## Goal

Close the loop between reviewer fill-template CSVs and the full manufacturing
review manifest. Reviewers can now return a compact template, and the tooling can
apply approved rows back into the complete manifest without treating suggestions as
approved labels.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Added `apply_reviewer_template_rows`.
  - Added `--apply-reviewer-template`.
  - Reuses existing row matching by `relative_path` and `file_name`.
  - Reuses approved review status governance.
  - Reuses optional reviewer metadata enforcement.
  - Skips template rows with no reviewed content, unapproved status, missing
    reviewer metadata, or no matching manifest row.
  - Writes a full updated review manifest, not just the template rows.
  - Adds `post_apply_validation` to the summary JSON so the caller can see whether
    the updated manifest is release-label-ready.
- Updated tests for:
  - direct template application behavior
  - CLI `--apply-reviewer-template` behavior
  - post-apply validation summary
- Updated Phase 6 TODO.

## CLI Usage

```bash
python scripts/build_manufacturing_review_manifest.py \
  --apply-reviewer-template reports/benchmark/forward_scorecard/manufacturing_reviewer_template.filled.csv \
  --base-manifest reports/experiments/<run>/manufacturing_review_manifest.csv \
  --output-csv reports/experiments/<run>/manufacturing_review_manifest.updated.csv \
  --summary-json reports/benchmark/forward_scorecard/manufacturing_reviewer_template_apply.json \
  --min-reviewed-samples 30 \
  --require-reviewer-metadata
```

Use `--fail-under-minimum` when the apply step should fail until the updated full
review manifest reaches release thresholds.

## Release Impact

The manufacturing review workflow now supports this loop:

1. Generate review manifest.
2. Generate gap CSV, assignment Markdown, and reviewer template CSV.
3. Reviewer fills and approves template rows.
4. Apply the approved template rows into the full review manifest.
5. Validate and merge approved rows into the benchmark manifest.

This keeps machine suggestions, human-reviewed labels, and release-ready benchmark
labels separate and auditable.

## Remaining Work

- Fill the template with real domain-approved source, payload, and `details.*`
  labels.
- Apply the filled template into the full review manifest.
- Run the existing approved-only merge and hybrid consumption gates.
