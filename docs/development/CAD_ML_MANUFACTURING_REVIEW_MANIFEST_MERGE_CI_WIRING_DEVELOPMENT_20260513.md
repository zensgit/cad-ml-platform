# CAD ML Manufacturing Review Manifest Merge CI Wiring Development

Date: 2026-05-13

## Goal

Make the approved-only reviewed benchmark manifest merge usable in CI/release runs.
The forward scorecard wrapper can now produce and publish a merged benchmark manifest
artifact when a reviewed manifest and a base benchmark manifest are configured.

## Changes

- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Adds optional base manifest input:
    `FORWARD_SCORECARD_MANUFACTURING_REVIEW_BASE_MANIFEST_CSV`.
  - Adds optional merge output paths:
    - `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MERGED_MANIFEST_CSV`
    - `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_SUMMARY_JSON`
  - Adds optional blocking mode:
    `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_FAIL_ON_BLOCKED=true`.
  - Runs `scripts/build_manufacturing_review_manifest.py
    --merge-approved-review-manifest` after the review manifest validation step when
    both review and base manifests are available.
  - Reuses
    `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA` for
    merge metadata enforcement.
  - Emits GitHub outputs:
    - `manufacturing_review_manifest_merge_available`
    - `manufacturing_review_manifest_base_csv`
    - `manufacturing_review_manifest_merged_csv`
    - `manufacturing_review_manifest_merge_summary_json`
    - `manufacturing_review_manifest_merge_status`
- Updated `.github/workflows/evaluation-report.yml`.
  - Adds repository-variable wiring for the merge inputs and outputs.
  - Uploads the merged reviewed benchmark manifest and merge summary as a dedicated
    artifact when merge output is available.
- Updated tests.
  - Wrapper ready-output test now verifies merged CSV output, merge summary output,
    and GitHub outputs.
  - Wrapper failure test verifies
    `FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_FAIL_ON_BLOCKED=true`.
  - Workflow wiring tests verify environment variables, upload ordering, and artifact
    paths.
- Updated Phase 6 TODO to mark CI/release merge artifact wiring complete.
- Follow-up workflow wiring now lets hybrid blind evaluation prefer the merged
  reviewed benchmark manifest in the same workflow run when that merge output is
  available.

## Configuration

```bash
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_CSV=reports/experiments/<run>/manufacturing_review_manifest.csv
FORWARD_SCORECARD_MANUFACTURING_REVIEW_BASE_MANIFEST_CSV=data/release/benchmark_manifest.csv
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MERGED_MANIFEST_CSV=reports/benchmark/forward_scorecard/manufacturing_review_manifest_merged.csv
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_SUMMARY_JSON=reports/benchmark/forward_scorecard/manufacturing_review_manifest_merge.json
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_REQUIRE_REVIEWER_METADATA=true
```

For release-labelled jobs that must have a merged reviewed manifest:

```bash
FORWARD_SCORECARD_MANUFACTURING_REVIEW_MANIFEST_MERGE_FAIL_ON_BLOCKED=true
```

## Release Impact

This keeps the release benchmark path auditable:

- Review manifest validation proves enough approved labels exist.
- Merge summary proves approved labels were copied into the base benchmark manifest.
- The merged benchmark manifest becomes a CI artifact that downstream release
  evaluation can consume without rebuilding or manually editing files in CI.

## Remaining Work

- Populate the real release benchmark review manifest with domain-approved source,
  payload, and detail labels.
- Run the first release-labelled evaluation job with a real approved review manifest
  and confirm the hybrid benchmark uses `manifest_source=reviewed_benchmark_manifest`.
- Tune source, payload, and detail quality thresholds after the reviewed set is
  stable.
