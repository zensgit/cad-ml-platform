# CAD ML Reviewer Template Apply Audit CSV Development

Date: 2026-05-14

## Goal

Make reviewer-template apply outcomes auditable per row. The apply summary gives
counts, but reviewer closeout needs a machine-readable CSV showing which filled
template rows were applied, skipped, or unmatched after preflight succeeds.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Adds `REVIEWER_TEMPLATE_APPLY_AUDIT_COLUMNS`.
  - Adds `build_reviewer_template_apply_audit_rows`.
  - Adds `--reviewer-template-apply-audit-csv`.
  - Reuses the same apply eligibility rules as the apply path.
  - Emits one audit row per filled-template row with:
    - `apply_status`
    - `apply_reasons`
    - `matched_manifest_row`
    - reviewer metadata
    - source/payload/detail readiness
    - reviewed fields and review notes
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds
    `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV`.
  - Writes the audit CSV when template apply runs.
  - Emits `manufacturing_reviewer_template_apply_audit_csv` as a GitHub output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the apply audit path.
  - Uploads the audit CSV with reviewer-template apply artifacts.
- Updated targeted tests for:
  - apply audit row construction
  - CLI audit CSV output
  - optional forward scorecard wrapper output wiring
  - workflow env and upload artifact wiring
- Updated Phase 6 TODO.

## Apply Statuses

The audit CSV uses explicit statuses:

- `applied`
- `skipped_no_review_content`
- `skipped_unapproved_template`
- `skipped_missing_metadata`
- `unmatched_template_row`
- `skipped_empty_updates`

These statuses help reviewer closeout without changing how labels are approved.

## Default CI Path

```text
reports/benchmark/forward_scorecard/manufacturing_reviewer_template_apply_audit.csv
```

Override with:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_AUDIT_CSV=<path>
```

## Release Impact

After a filled template passes preflight and is applied, CI now preserves a
row-level audit artifact. Any skipped or unmatched row can be traced directly
back to the reviewer-template source row.

## Remaining Work

- Run apply audit on the first real filled reviewer template.
- Feed skipped and unmatched rows back into reviewer closeout.
- Merge only approved applied rows into the benchmark manifest.
- Tune thresholds after the reviewed release set is stable.
