# CAD ML Manufacturing Evidence Source Correctness Development

Date: 2026-05-12

## Goal

Move manufacturing evidence evaluation beyond coverage. Labeled DXF benchmark runs can
now compare predicted manufacturing evidence sources against reviewed expected
sources and report precision, recall, F1, and exact-match metrics.

## Changes

- Updated `scripts/eval_hybrid_dxf_manifest.py`.
  - Added optional reviewed manifest columns:
    - `expected_manufacturing_evidence_sources`
    - `expected_manufacturing_sources`
    - `reviewed_manufacturing_evidence_sources`
    - `reviewed_manufacturing_sources`
  - Accepts semicolon, comma, pipe, or JSON-list source values.
  - Supports aliases such as `process`, `cost`, and `decision`.
  - Supports `none`/`no_evidence` to mark a reviewed sample with no expected
    manufacturing evidence.
  - Adds row-level correctness fields:
    - `expected_manufacturing_evidence_sources`
    - `manufacturing_evidence_reviewed`
    - `manufacturing_evidence_true_positive_sources`
    - `manufacturing_evidence_false_positive_sources`
    - `manufacturing_evidence_false_negative_sources`
    - `manufacturing_evidence_source_exact_match`
    - `manufacturing_evidence_source_precision`
    - `manufacturing_evidence_source_recall`
    - `manufacturing_evidence_source_f1`
  - Extends `summary.json.manufacturing_evidence` with reviewed sample count,
    micro precision/recall/F1, exact-match rate, and per-source correctness.
- Updated `src/core/benchmark/forward_scorecard.py`.
  - Reads the new correctness metrics from manufacturing evidence summaries.
  - Requires at least 30 reviewed samples for `release_ready`.
  - Requires source precision and recall at or above 0.9 for `release_ready`.
  - Renders reviewed count, precision, and recall in the scorecard component table.
- Updated unit tests for manifest parsing, row-level source correctness, aggregate
  metrics, and scorecard downgrade behavior.
- Updated the Phase TODO and manufacturing evidence scorecard notes.

## Manifest Example

```csv
file_name,label_cn,expected_manufacturing_evidence_sources
shaft.dxf,轴类,dfm;process;cost;decision
cover.dxf,壳体类,dfm;cost
unknown.dxf,其他,none
```

Canonical sources:

- `dfm`
- `manufacturing_process`
- `manufacturing_cost`
- `manufacturing_decision`

## Summary Additions

`summary.json.manufacturing_evidence` now includes:

```json
{
  "source_correctness_available": true,
  "reviewed_sample_count": 80,
  "source_exact_match_rate": 0.95,
  "source_precision": 0.97,
  "source_recall": 0.96,
  "source_f1": 0.964974,
  "source_correctness": {
    "dfm": {
      "expected_count": 80,
      "true_positive": 78,
      "false_positive": 1,
      "false_negative": 2,
      "precision": 0.987342,
      "recall": 0.975,
      "f1": 0.981132
    }
  }
}
```

## Release Impact

Manufacturing evidence can no longer be `release_ready` from coverage alone. If
reviewed correctness is missing or below threshold, the forward scorecard downgrades
the component to `benchmark_ready_with_gap` or lower.

## Remaining Work

- Populate real reviewed source labels for the release benchmark set.
- Tune source correctness thresholds once the reviewed set has enough production
  samples.
- Top-level payload quality labels are now supported for `kind`, `label`, and
  `status`.
- Extend payload quality labels into nested source-specific details, such as DFM
  issue validity or process-route correctness.
