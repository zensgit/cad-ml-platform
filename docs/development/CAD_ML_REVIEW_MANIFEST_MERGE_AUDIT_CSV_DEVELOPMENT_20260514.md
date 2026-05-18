# CAD ML Review Manifest Merge Audit CSV Development

Date: 2026-05-14

## Goal

Make approved-review-manifest merge outcomes auditable per row. The merge
summary gives aggregate counts, but release closeout needs a machine-readable
CSV showing which reviewed rows merged into the benchmark manifest, which were
skipped, and which did not match a benchmark row.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Adds `REVIEW_MANIFEST_MERGE_AUDIT_COLUMNS`.
  - Adds `build_review_manifest_merge_audit_rows`.
  - Adds `--review-manifest-merge-audit-csv`.
  - Reuses the same approved-only merge eligibility rules.
  - Emits one audit row per reviewed manifest row with:
    - `merge_status`
    - `merge_reasons`
    - `matched_base_row`
    - reviewer metadata
    - source/payload/detail readiness
    - reviewed fields and review notes
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds
    `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_AUDIT_CSV`.
  - Writes the merge audit CSV when approved-review-manifest merge runs.
  - Emits `manufacturing_review_manifest_merge_audit_csv` as a GitHub output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the merge audit path.
  - Uploads the audit CSV with reviewed benchmark manifest artifacts.
- Updated targeted tests for:
  - merge audit row construction
  - CLI merge audit CSV output
  - optional forward scorecard wrapper output wiring
  - workflow env and upload artifact wiring
- Updated Phase 6 TODO.

## Merge Statuses

The audit CSV uses explicit statuses:

- `merged`
- `skipped_no_review_content`
- `skipped_unapproved_review`
- `skipped_missing_metadata`
- `unmatched_review_row`
- `skipped_empty_updates`

These statuses expose merge mechanics without changing approval semantics.

## Default CI Path

```text
reports/benchmark/forward_scorecard/manufacturing_review_manifest_merge_audit.csv
```

Override with:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_AUDIT_CSV=<path>
```

## Release Impact

Reviewed benchmark manifest artifacts now include row-level merge evidence.
Skipped or unmatched review rows can be traced directly, so release reviewers do
not need to infer merge issues from aggregate counts alone.

## Remaining Work

- Run the merge audit against the first real approved review set.
- Resolve skipped and unmatched reviewed rows.
- Re-run the forward scorecard with the merged benchmark manifest.
- Tune thresholds after the reviewed release set is stable.
