# CAD ML Manufacturing Evidence Benchmark Summary Development

Date: 2026-05-12

## Goal

Close the gap between manufacturing evidence emitted by analyze and manufacturing
evidence consumed by the forward scorecard. Real DXF benchmark exporters now emit
row-level manufacturing evidence fields plus a `summary.json.manufacturing_evidence`
object that can be passed directly to `scripts/export_forward_scorecard.py`.

## Changes

- Updated `scripts/batch_analyze_dxf_local.py`.
  - Adds row-level fields for:
    - `manufacturing_evidence`
    - `manufacturing_evidence_count`
    - `manufacturing_evidence_sources`
    - `manufacturing_evidence_required_sources_present`
    - `manufacturing_evidence_has_dfm`
    - `manufacturing_evidence_has_process`
    - `manufacturing_evidence_has_cost`
    - `manufacturing_evidence_has_decision`
  - Adds `summary.json.manufacturing_evidence` for local batch DXF runs.
- Updated `scripts/eval_hybrid_dxf_manifest.py`.
  - Exports the same row-level manufacturing evidence fields for labeled DXF
    manifest runs.
  - Adds the same forward-scorecard-compatible manufacturing evidence summary.
  - Adds optional reviewed source labels and correctness metrics for labeled
    benchmark runs.
  - Adds optional reviewed payload quality labels and aggregate payload accuracy.
  - Adds optional nested detail payload quality labels and aggregate detail accuracy.
- Updated exporter unit tests for direct analyze output, DecisionService fallback
  evidence, and aggregate summary coverage.
- Updated the Phase TODO and forward-scorecard manufacturing evidence notes.

## Summary Contract

Both exporters now write this object under `summary.json.manufacturing_evidence`:

```json
{
  "sample_size": 80,
  "records_with_manufacturing_evidence": 80,
  "manufacturing_evidence_coverage_rate": 1.0,
  "manufacturing_evidence_total_count": 320,
  "source_counts": {
    "dfm": 80,
    "manufacturing_process": 80,
    "manufacturing_cost": 80,
    "manufacturing_decision": 80
  },
  "source_coverage_rates": {
    "dfm": 1.0,
    "manufacturing_process": 1.0,
    "manufacturing_cost": 1.0,
    "manufacturing_decision": 1.0
  },
  "sources": [
    "dfm",
    "manufacturing_process",
    "manufacturing_cost",
    "manufacturing_decision"
  ],
  "required_sources": [
    "dfm",
    "manufacturing_process",
    "manufacturing_cost",
    "manufacturing_decision"
  ]
}
```

## Extraction Order

The exporters prefer explicit manufacturing evidence when present:

1. `results.manufacturing_evidence`
2. `classification.manufacturing_evidence`
3. Manufacturing-source rows filtered from `classification.evidence`
4. Manufacturing-source rows filtered from `classification.decision_contract.evidence`

This keeps current analyze output first, while preserving compatibility with older
DecisionService-only payloads.

## Forward Scorecard Usage

The generated `summary.json` can be passed directly:

```bash
python scripts/export_forward_scorecard.py \
  --manufacturing-evidence-summary reports/experiments/<run>/summary.json
```

The scorecard already accepts either the nested `manufacturing_evidence` object or
the object itself.

## Reviewed Correctness Columns

Labeled DXF manifests can add any one of these columns:

- `expected_manufacturing_evidence_sources`
- `expected_manufacturing_sources`
- `reviewed_manufacturing_evidence_sources`
- `reviewed_manufacturing_sources`

Values can be semicolon, comma, pipe, or JSON-list encoded. Use `none` to mark a
reviewed sample with no expected manufacturing evidence.

## Payload Quality Columns

Labeled DXF manifests can add JSON payload expectations:

- `expected_manufacturing_evidence_payload_json`
- `expected_manufacturing_payload_json`
- `reviewed_manufacturing_evidence_payload_json`
- `reviewed_manufacturing_payload_json`

They can also use convenience columns such as `expected_dfm_status`,
`expected_process_label`, `expected_cost_label`, and `expected_decision_status`.
The quality comparison covers top-level `kind`, `label`, and `status`, plus nested
`details.*` fields.

Nested detail expectations can be provided as nested JSON `details` objects, explicit
`details.*` JSON fields, or convenience columns such as:

- `expected_dfm_detail_mode`
- `expected_process_detail_rule_version`
- `expected_cost_detail_cost_range__low`
- `expected_decision_detail_risks_count`

In convenience columns, `__` maps to a nested dot path. For example,
`expected_cost_detail_cost_range__low` maps to `details.cost_range.low`.

## Remaining Work

- CI/release workflow now uploads the consumed manufacturing evidence summary as a
  separate artifact when the configured path exists.
- Labeled DXF benchmark exporter now emits source correctness metrics from reviewed
  manufacturing evidence source labels.
- Labeled DXF benchmark exporter now emits payload quality accuracy from reviewed
  manufacturing evidence payload labels.
- Labeled DXF benchmark exporter now emits nested detail quality accuracy from
  reviewed manufacturing evidence payload labels.
- Populate real reviewed source, payload, and detail labels for the release benchmark
  set.
- Tune source, payload, and detail quality thresholds after the release review set is
  stable.
- Extend the same summary contract to OCR-only benchmark runs if they use a separate
  exporter from the DXF analyze path.
