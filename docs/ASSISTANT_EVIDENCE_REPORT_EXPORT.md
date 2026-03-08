# Assistant Evidence Report Export

`scripts/export_assistant_evidence_report.py` exports offline explainability
artifacts from assistant or analyze outputs into:

- one flat CSV row per record
- one aggregate JSON summary for coverage and gaps

## Supported Inputs

- single `JSON` response file
- `JSONL` file with one record per line
- directory tree containing `*.json` and `*.jsonl`

The script normalizes these common shapes:

- assistant responses with `answer` / `sources` / `evidence`
- analyze outputs with `results.classification.*`
- review or active-learning exports with `score_breakdown.*`

## Usage

```bash
python3 scripts/export_assistant_evidence_report.py \
  --input-path path/to/assistant_outputs \
  --input-path path/to/more_records.jsonl \
  --output-csv artifacts/assistant_evidence_report.csv \
  --summary-json artifacts/assistant_evidence_report.summary.json
```

## CSV Columns

- `record_id`, `record_kind`, `record_locator`
- `query`, `answer_preview`
- `evidence_count`, `evidence_types`, `evidence_sources`
- `sources_count`, `source_contributions_count`
- `decision_path_count`, `decision_path`
- `explanation_summary`
- `missing_fields`, `evidence_missing_fields`

## Summary Metrics

The JSON summary includes:

- evidence / source / decision-path coverage at record level
- evidence item coverage for `source`, `summary`, `match_type`,
  and `reference_id`
- `top_evidence_types`
- `top_structured_sources`
- `top_decision_steps`
- `top_missing_fields`
- `average_evidence_count`

`top_missing_fields` mixes record-level gaps such as `evidence` or
`decision_path` with evidence item gaps such as `evidence[].summary`.
