# CAD ML Reviewed Manifest Hybrid Consumption Development

Date: 2026-05-13

## Goal

Make the reviewed manufacturing benchmark manifest useful inside the same
evaluation workflow run. After the forward scorecard wrapper produces a merged
reviewed benchmark manifest, the hybrid blind benchmark now prefers that manifest
over the manually configured `HYBRID_BLIND_MANIFEST_CSV`.

## Changes

- Updated `.github/workflows/evaluation-report.yml`.
  - The `Run Hybrid blind benchmark (optional)` step now reads:
    - `steps.forward_scorecard.outputs.manufacturing_review_manifest_merge_available`
    - `steps.forward_scorecard.outputs.manufacturing_review_manifest_merged_csv`
  - If the merge output is available and the CSV exists, `MANIFEST_CSV` is set to
    the merged reviewed benchmark manifest.
  - If the merge output is unavailable or missing on disk, the step falls back to
    `HYBRID_BLIND_MANIFEST_CSV`.
  - The step emits `manifest_source` so downstream reports can distinguish:
    - `reviewed_benchmark_manifest`
    - `configured`
  - `manifest_csv` and `manifest_source` are emitted for both normal and
    missing-summary outcomes.
- Updated workflow regression tests.
  - Verified the hybrid blind step consumes the forward scorecard merge outputs.
  - Verified the forward scorecard step runs before hybrid blind evaluation, which is
    required for same-run reviewed manifest consumption.
  - Verified existing hybrid superpass integration tests see the reviewed manifest
    path and output metadata.
- Updated Phase 6 TODO.

## Runtime Behavior

Preferred path when reviewed labels are available:

```text
review manifest -> validate -> merge into base manifest -> hybrid blind benchmark
```

Fallback path when reviewed labels are not yet available:

```text
HYBRID_BLIND_MANIFEST_CSV -> hybrid blind benchmark
```

If neither path is available, the benchmark keeps the previous no-manifest behavior.

## Release Impact

This connects the reviewed manufacturing labels to the benchmark that produces
release evidence. Once the release review manifest is populated with real approved
labels, a single workflow run can merge the reviewed labels and evaluate hybrid
outputs against that reviewed benchmark manifest.

## Remaining Work

- Populate the real release benchmark review manifest with domain-approved source,
  payload, and detail labels.
- Run a release-labelled evaluation and confirm the hybrid blind step reports
  `manifest_source=reviewed_benchmark_manifest`.
- Tune source, payload, and detail quality thresholds after the reviewed set is
  stable.
