# CAD ML Manufacturing Evidence Payload Quality Development

Date: 2026-05-12

## Goal

Extend manufacturing evidence correctness from source presence to source payload
quality. Labeled DXF benchmark manifests can now review whether each manufacturing
evidence source has the expected `kind`, `label`, `status`, and nested
`details.*` values.

## Changes

- Updated `scripts/eval_hybrid_dxf_manifest.py`.
  - Added optional JSON payload review columns:
    - `expected_manufacturing_evidence_payload_json`
    - `expected_manufacturing_payload_json`
    - `reviewed_manufacturing_evidence_payload_json`
    - `reviewed_manufacturing_payload_json`
  - Added optional per-source convenience columns:
    - `expected_dfm_kind`
    - `expected_dfm_label`
    - `expected_dfm_status`
    - `expected_process_kind`
    - `expected_process_label`
    - `expected_process_status`
    - `expected_cost_kind`
    - `expected_cost_label`
    - `expected_cost_status`
    - `expected_decision_kind`
    - `expected_decision_label`
    - `expected_decision_status`
  - Added optional per-source detail convenience columns with
    `expected_<source>_detail_<path>`, where `__` represents a nested dot path.
    Example: `expected_cost_detail_cost_range__low` maps to
    `details.cost_range.low`.
  - Added nested JSON payload support for `details` objects and explicit
    `details.*` fields.
  - Adds row-level payload quality fields:
    - `expected_manufacturing_evidence_payloads`
    - `manufacturing_evidence_payload_quality_reviewed`
    - `manufacturing_evidence_payload_quality`
    - `manufacturing_evidence_payload_expected_fields`
    - `manufacturing_evidence_payload_matched_fields`
    - `manufacturing_evidence_payload_mismatched_fields`
    - `manufacturing_evidence_payload_missing_fields`
    - `manufacturing_evidence_payload_quality_accuracy`
    - `manufacturing_evidence_payload_detail_quality_reviewed`
    - `manufacturing_evidence_payload_detail_expected_fields`
    - `manufacturing_evidence_payload_detail_matched_fields`
    - `manufacturing_evidence_payload_detail_mismatched_fields`
    - `manufacturing_evidence_payload_detail_missing_fields`
    - `manufacturing_evidence_payload_detail_quality_accuracy`
  - Extends `summary.json.manufacturing_evidence` with payload quality reviewed
    sample count, expected/matched/mismatched/missing field totals, accuracy, and
    per-source payload quality including detail accuracy.
- Updated `src/core/benchmark/forward_scorecard.py`.
  - Requires payload quality and detail review evidence for manufacturing evidence
    `release_ready`.
  - Requires at least 30 payload-reviewed samples.
  - Requires payload quality accuracy at or above 90%.
  - Requires at least 30 detail-reviewed samples and detail quality accuracy at or
    above 90%.
  - Renders payload quality accuracy in the scorecard component table.
- Updated unit tests for manifest parsing, row-level payload quality, aggregate
  payload accuracy, and scorecard downgrade behavior.
- Updated Phase TODO and manufacturing evidence scorecard notes.

## Manifest Examples

JSON column:

```csv
file_name,label_cn,expected_manufacturing_evidence_payload_json
shaft.dxf,轴类,"{""dfm"":{""status"":""manufacturable""},""process"":{""label"":""milling""}}"
```

Convenience columns:

```csv
file_name,label_cn,expected_dfm_status,expected_process_label,expected_cost_label
shaft.dxf,轴类,manufacturable,milling,CNY
```

Nested detail convenience columns:

```csv
file_name,label_cn,expected_dfm_status,expected_dfm_detail_mode,expected_cost_label,expected_cost_detail_cost_range__low
shaft.dxf,轴类,manufacturable,rule,CNY,90.0
```

## Summary Additions

`summary.json.manufacturing_evidence` now includes:

```json
{
  "payload_quality_available": true,
  "payload_quality_reviewed_sample_count": 80,
  "payload_quality_expected_field_total": 240,
  "payload_quality_matched_field_total": 228,
  "payload_quality_mismatched_field_total": 8,
  "payload_quality_missing_field_total": 4,
  "payload_quality_accuracy": 0.95,
  "payload_detail_quality_available": true,
  "payload_detail_quality_reviewed_sample_count": 80,
  "payload_detail_quality_expected_field_total": 80,
  "payload_detail_quality_matched_field_total": 74,
  "payload_detail_quality_mismatched_field_total": 4,
  "payload_detail_quality_missing_field_total": 2,
  "payload_detail_quality_accuracy": 0.925,
  "payload_quality": {
    "dfm": {
      "expected_field_count": 80,
      "matched_field_count": 76,
      "mismatched_field_count": 2,
      "missing_field_count": 2,
      "accuracy": 0.95,
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

- source coverage
- reviewed source precision/recall
- reviewed payload quality accuracy
- reviewed nested detail quality accuracy

This prevents a release-ready claim when the system emits the right source names but
incorrect statuses, process labels, cost labels, evidence kinds, or nested evidence
details.

## Remaining Work

- Populate real reviewed source, payload, and detail labels for the release benchmark
  set.
- Tune payload and detail quality thresholds after enough production samples are
  reviewed.
- Add deeper domain-specific validators for detail semantics such as DFM issue
  validity, process-route steps, and cost range accuracy.
