# CAD ML Manufacturing Review Manifest Scorecard Evidence Development

Date: 2026-05-13

## Goal

Make manufacturing review manifest validation visible inside the forward scorecard
itself. CI already validates and uploads the manifest validation summary, but release
claims cite the scorecard artifact; the scorecard now records whether the reviewed
source, payload, and nested detail labels are sufficiently populated.

## Changes

- Updated `scripts/export_forward_scorecard.py`.
  - Adds `--manufacturing-review-manifest-validation-summary`.
  - Loads the validation summary JSON and passes it into the scorecard builder.
  - Records the validation summary path under
    `artifacts.manufacturing_review_manifest_validation_summary`.
- Updated `src/core/benchmark/forward_scorecard.py`.
  - Adds review manifest validation fields under
    `components.manufacturing_evidence.review_manifest_validation`.
  - Preserves reviewed sample counts, payload/detail field totals, status, and
    blocking reasons.
  - Preserves review governance fields such as approved/unapproved review counts,
    approved statuses, reviewer metadata requirement, and missing reviewer metadata
    count.
  - Downgrades `manufacturing_evidence` from `release_ready` to
    `benchmark_ready_with_gap` when a provided validation summary is blocked.
  - Adds `manufacturing_review_manifest_validation_blocked` to evidence gaps when the
    validation summary is below threshold.
  - Renders review manifest validation status in the manufacturing evidence Markdown
    evidence cell.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - After validating a configured manufacturing review manifest, passes the generated
    validation summary into `scripts/export_forward_scorecard.py`.
- Updated tests.
  - Direct scorecard test covers blocked validation downgrade.
  - Exporter test covers artifact path and component status preservation.
  - CI wrapper test covers ready validation evidence and blocked validation downgrade.

## Scorecard Shape

When a validation summary is provided:

```json
{
  "components": {
    "manufacturing_evidence": {
      "status": "benchmark_ready_with_gap",
      "review_manifest_validation": {
        "status": "blocked",
        "row_count": 80,
        "min_reviewed_samples": 30,
        "source_reviewed_sample_count": 80,
        "payload_reviewed_sample_count": 12,
        "payload_detail_reviewed_sample_count": 8,
        "approved_review_sample_count": 12,
        "unapproved_review_sample_count": 3,
        "require_reviewer_metadata": true,
        "reviewer_metadata_missing_sample_count": 1,
        "payload_expected_field_total": 64,
        "payload_detail_expected_field_total": 24,
        "blocking_reasons": [
          "payload_reviewed_sample_count_below_minimum",
          "payload_detail_reviewed_sample_count_below_minimum"
        ]
      },
      "evidence_gaps": [
        "manufacturing_review_manifest_validation_blocked"
      ]
    }
  },
  "artifacts": {
    "manufacturing_review_manifest_validation_summary": "reports/benchmark/forward_scorecard/manufacturing_review_manifest_validation.json"
  }
}
```

## Release Impact

This closes the evidence gap between CI outputs and the scorecard artifact. A release
claim that cites the forward scorecard can now see whether manufacturing reviewed
labels were actually validated. If the configured validation summary is blocked, the
scorecard no longer reports manufacturing evidence as `release_ready`.

## Remaining Work

- Populate the real release benchmark manifest with human-reviewed manufacturing
  source, payload, and detail labels.
- Enable fail-on-blocked mode for release-labelled jobs after the release manifest
  path is stable.
- Tune thresholds after the real reviewed label set has enough samples.
