# CAD ML Manufacturing Reviewer Template Apply CI Development

Date: 2026-05-14

## Goal

Wire the reviewer-template apply path into the optional forward scorecard CI flow.
The CLI can already apply a filled reviewer template into the full review manifest;
this slice lets release CI consume that filled template before validation, scorecard
ingestion, and approved-only benchmark-manifest merge.

## Changes

- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV`.
  - Adds `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLIED_MANIFEST_CSV`.
  - Adds `FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_SUMMARY_JSON`.
  - When a filled template CSV exists, runs:
    `scripts/build_manufacturing_review_manifest.py --apply-reviewer-template`.
  - Validates the applied review manifest instead of the original manifest.
  - Emits apply outputs:
    - `manufacturing_reviewer_template_apply_available`
    - `manufacturing_reviewer_template_apply_csv`
    - `manufacturing_reviewer_template_applied_manifest_csv`
    - `manufacturing_reviewer_template_apply_summary_json`
    - `manufacturing_reviewer_template_apply_status`
  - Keeps the existing behavior unchanged when no filled template is configured.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the filled template, applied manifest, and
    apply summary paths.
  - Adds a dedicated upload step for reviewer-template apply artifacts.
- Updated tests for:
  - wrapper behavior without a filled template
  - wrapper behavior with a filled template before validation
  - workflow environment wiring
  - workflow upload artifact wiring
- Updated Phase 6 TODO.

## CI Usage

Configure these variables for a release job that should consume a filled reviewer
template:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV=reports/experiments/<run>/manufacturing_review_manifest.csv
FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_CSV=reports/benchmark/forward_scorecard/manufacturing_reviewer_template.filled.csv
FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLIED_MANIFEST_CSV=reports/benchmark/forward_scorecard/manufacturing_review_manifest.applied.csv
FORWARD_SCORECARD_MANUFACTURING_REVIEWER_TEMPLATE_APPLY_SUMMARY_JSON=reports/benchmark/forward_scorecard/manufacturing_reviewer_template_apply.json
```

The wrapper then validates the applied manifest and exposes it as
`manufacturing_review_manifest_csv` for downstream merge and scorecard steps.

## Release Impact

The release workflow can now run the complete human-review path in CI:

1. Start from the generated review manifest.
2. Apply a filled reviewer template.
3. Validate the applied full review manifest.
4. Feed validation into the forward scorecard.
5. Merge approved labels into the benchmark manifest when configured.

This keeps the release gate tied to approved reviewer output, not machine
suggestions.

## Remaining Work

- Produce the first real filled reviewer template from domain review.
- Run the CI path with the filled template configured.
- Tune source, payload, and detail thresholds after the reviewed release set is
  stable.
