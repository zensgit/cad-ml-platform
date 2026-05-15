# CAD ML Manufacturing Review Context CSV Development

Date: 2026-05-14

## Goal

Reduce manufacturing label closeout cost by emitting a machine-readable context
CSV for every review-manifest row that still has release-label gaps.

The CSV is reviewer evidence, not an approval shortcut. It summarizes suggested
manufacturing evidence, raw actual evidence, readiness flags, and current review
fields so reviewers can decide what to fill in the reviewer template.

## Changes

- Updated `scripts/build_manufacturing_review_manifest.py`.
  - Adds `REVIEW_CONTEXT_COLUMNS`.
  - Adds `build_review_context_rows`.
  - Adds `--review-context-csv` for build and validate modes.
  - Summarizes suggested sources and payload fields.
  - Summarizes actual evidence sources, payload fields, and `details.*` keys.
  - Keeps existing review gap semantics unchanged.
- Updated review handoff Markdown generation.
  - Adds the context CSV to the artifact map.
  - Directs reviewers to use assignment, gap, and context artifacts together.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds `FORWARD_SCORECARD_MANUFACTURING_REVIEW_CONTEXT_CSV`.
  - Passes `--review-context-csv` during review-manifest validation.
  - Emits `manufacturing_review_context_csv` as a GitHub Actions output.
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the context CSV path.
  - Uploads the context CSV with review-manifest validation artifacts.
- Updated targeted tests for:
  - row-level context generation
  - CLI context CSV writing
  - forward scorecard wrapper output wiring
  - workflow env and artifact upload wiring
- Updated Phase 6 TODO.

## CSV Columns

Key columns include:

- `row_id`, `file_name`, `label_cn`, `relative_path`, `source_dir`
- `gap_reasons`, `source_ready`, `payload_ready`, `detail_ready`
- `suggested_manufacturing_evidence_sources`
- `suggested_payload_fields`
- `suggested_manufacturing_evidence_payload_json`
- `actual_evidence_sources`
- `actual_evidence_summary`
- `actual_evidence_detail_keys`
- `actual_manufacturing_evidence`
- reviewed source, payload, status, reviewer, timestamp, and notes fields

## Default CI Path

```text
reports/benchmark/forward_scorecard/manufacturing_review_context.csv
```

Override with:

```text
FORWARD_SCORECARD_MANUFACTURING_REVIEW_CONTEXT_CSV=<path>
```

## Release Impact

Reviewers can now use one CSV to inspect why a row is blocked, what the system
suggested, and what actual manufacturing evidence was produced. This should make
manual source, payload, and detail-label closeout faster without relaxing
release-readiness gates.

## Remaining Work

- Populate real reviewed source, payload, and detail labels for the release
  benchmark set.
- Use the context CSV during the first qualified manufacturing review pass.
- Tune source, payload, and detail quality thresholds after the reviewed release
  set is stable.
