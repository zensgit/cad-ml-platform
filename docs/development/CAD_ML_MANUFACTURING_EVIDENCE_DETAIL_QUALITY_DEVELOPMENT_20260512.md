# CAD ML Manufacturing Evidence Detail Quality Development

Date: 2026-05-12

## Goal

Extend reviewed manufacturing evidence payload quality from top-level
`kind`/`label`/`status` fields into source-specific nested details. Release readiness
can now distinguish "the right source and label were emitted" from "the detailed DFM,
process, cost, or decision evidence was also correct."

## Changes

- Updated `scripts/eval_hybrid_dxf_manifest.py`.
  - Manifest JSON payload expectations now accept nested `details` objects.
  - Manifest JSON payload expectations also accept explicit `details.*` fields.
  - Convenience columns now support source-specific detail labels:
    - `expected_dfm_detail_mode`
    - `expected_process_detail_rule_version`
    - `expected_cost_detail_cost_range__low`
    - `expected_decision_detail_risks_count`
  - Detail convenience column suffixes use `__` to represent nested dots, so
    `expected_cost_detail_cost_range__low` maps to `details.cost_range.low`.
  - Row output now includes detail-specific payload quality fields:
    - `manufacturing_evidence_payload_detail_quality_reviewed`
    - `manufacturing_evidence_payload_detail_expected_fields`
    - `manufacturing_evidence_payload_detail_matched_fields`
    - `manufacturing_evidence_payload_detail_mismatched_fields`
    - `manufacturing_evidence_payload_detail_missing_fields`
    - `manufacturing_evidence_payload_detail_quality_accuracy`
  - `summary.json.manufacturing_evidence` now includes aggregate detail quality
    reviewed sample count, expected/matched/mismatched/missing totals, and detail
    accuracy.
  - Per-source payload quality now includes `detail_accuracy` and detail field
    counters.
  - Payload value normalization preserves valid falsy values such as numeric `0`.
- Updated `src/core/benchmark/forward_scorecard.py`.
  - Manufacturing evidence `release_ready` now requires reviewed detail quality.
  - Detail-reviewed sample count must be at least 30.
  - Detail quality accuracy must be at least 90%.
  - Component output includes `payload_detail_quality_available`,
    `payload_detail_quality_reviewed_sample_count`, and
    `payload_detail_quality_accuracy`.
  - Markdown component summary renders `detail_acc`.
- Updated unit tests for manifest parsing, row-level detail matching, aggregate detail
  accuracy, and release-gate downgrade behavior.
- Updated development TODO and manufacturing evidence design notes.

## Manifest Examples

Nested JSON payload expectations:

```csv
file_name,label_cn,expected_manufacturing_evidence_payload_json
shaft.dxf,轴类,"{""dfm"":{""status"":""manufacturable"",""details"":{""mode"":""rule""}},""cost"":{""label"":""CNY"",""details"":{""cost_range"":{""low"":""90.0""}}}}"
```

Explicit detail fields:

```json
{
  "dfm": {
    "status": "manufacturable",
    "details.mode": "rule"
  },
  "manufacturing_cost": {
    "label": "CNY",
    "details.cost_range.low": "90.0"
  }
}
```

Convenience columns:

```csv
file_name,label_cn,expected_dfm_status,expected_dfm_detail_mode,expected_cost_label,expected_cost_detail_cost_range__low
shaft.dxf,轴类,manufacturable,rule,CNY,90.0
```

## Summary Additions

`summary.json.manufacturing_evidence` now includes:

```json
{
  "payload_detail_quality_available": true,
  "payload_detail_quality_reviewed_sample_count": 80,
  "payload_detail_quality_expected_field_total": 160,
  "payload_detail_quality_matched_field_total": 148,
  "payload_detail_quality_mismatched_field_total": 8,
  "payload_detail_quality_missing_field_total": 4,
  "payload_detail_quality_accuracy": 0.925,
  "payload_quality": {
    "manufacturing_cost": {
      "expected_field_count": 240,
      "matched_field_count": 220,
      "accuracy": 0.916667,
      "detail_expected_field_count": 80,
      "detail_matched_field_count": 74,
      "detail_mismatched_field_count": 4,
      "detail_missing_field_count": 2,
      "detail_accuracy": 0.925
    }
  }
}
```

## Release Impact

Manufacturing evidence `release_ready` now requires:

- manufacturing evidence coverage
- reviewed source precision and recall
- reviewed top-level payload quality
- reviewed nested detail payload quality

This closes the previous gap where a benchmark could pass with the correct evidence
sources and top-level labels while still emitting incorrect nested DFM modes, process
rule versions, cost ranges, or decision risk counts.

## Remaining Work

- Populate real reviewed source, payload, and detail labels for the release benchmark
  set.
- Tune source, payload, and detail quality thresholds after the release review set is
  stable.
- Add domain-specific detail validators for deeper semantics, such as process-route
  step order and DFM issue validity, after enough reviewed samples are available.
