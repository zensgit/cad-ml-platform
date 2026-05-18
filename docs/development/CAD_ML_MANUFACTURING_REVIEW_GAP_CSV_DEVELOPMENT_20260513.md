# CAD ML Manufacturing Review Gap CSV Development

Date: 2026-05-13

## Goal

Turn the manufacturing review progress report into an assignable backlog. The
Markdown report is useful for release review, but reviewers also need a stable CSV
that can be filtered by label, row, status, missing action, and readiness state.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Added `REVIEW_GAP_COLUMNS`.
  - Added `build_review_gap_rows`.
  - Added `--gap-csv` for build and validate modes.
  - Gap rows are emitted only for rows with remaining review work.
  - Each gap row includes:
    - row identity fields
    - reviewer and review timestamp fields
    - semicolon-delimited gap reasons
    - source, payload, and detail readiness booleans
    - suggested manufacturing evidence source and payload columns
    - reviewed manufacturing evidence source and payload columns
    - review notes
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds
    `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_GAP_CSV`.
  - Passes `--gap-csv` when validating the review manifest.
  - Emits `manufacturing_review_manifest_gap_csv` as a GitHub output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the gap CSV path.
  - Uploads the gap CSV with the manufacturing review manifest validation artifact.
- Updated tests for:
  - gap-row content and actionable reasons
  - CLI gap CSV generation
  - forward scorecard wrapper output wiring
  - workflow artifact upload wiring
- Updated Phase 6 TODO.

## Gap Reasons

The CSV reuses the same blocker taxonomy as the progress Markdown:

- `fill reviewed_manufacturing_evidence_sources`
- `fill reviewed_manufacturing_evidence_payload_json`
- `add details.* payload labels`
- `set approved review_status`
- `fill reviewer and reviewed_at`

## CLI Usage

```bash
python scripts/build_manufacturing_review_manifest.py \
  --validate-manifest reports/experiments/<run>/manufacturing_review_manifest.csv \
  --summary-json reports/benchmark/forward_scorecard/manufacturing_review_manifest_validation.json \
  --progress-md reports/benchmark/forward_scorecard/manufacturing_review_manifest_progress.md \
  --gap-csv reports/benchmark/forward_scorecard/manufacturing_review_manifest_gaps.csv \
  --min-reviewed-samples 30 \
  --require-reviewer-metadata
```

## Release Impact

The release closeout now has two complementary artifacts:

- Markdown progress report for human status review.
- CSV gap backlog for filtering, assignment, and batch review tracking.

This keeps the next remaining work focused on domain review labels rather than
tooling interpretation.

## Remaining Work

- Populate real reviewed source, payload, and detail labels for the release
  benchmark set.
- Use the gap CSV to assign rows by `label_cn`, `gap_reasons`, and readiness state.
- Re-run validation until the gap CSV contains only the header and the validation
  summary is `release_label_ready`.
