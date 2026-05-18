# CAD ML DecisionService Benchmark Exporters Development

Date: 2026-05-12

## Goal

Close the Phase 5 DecisionService boundary by making benchmark exporters preserve
the shared decision contract and evidence emitted by analyze classification.

## Changes

- Updated `scripts/eval_hybrid_dxf_manifest.py`.
  - Adds additive `decision_*` CSV columns for:
    - `decision_contract_present`
    - `decision_contract_version`
    - `decision_source`
    - `decision_contract`
    - `decision_evidence`
    - `decision_evidence_count`
    - `decision_evidence_sources`
    - `decision_fallback_flags`
    - `decision_review_reasons`
    - `decision_branch_conflicts`
  - Adds `summary.json.decision_signals` with contract coverage, evidence coverage,
    evidence source counts, fallback flag counts, review reason counts, and branch
    conflict count.
- Updated `scripts/batch_analyze_dxf_local.py`.
  - Exports the same `decision_*` row fields for local DXF batch benchmark runs.
  - Adds `summary.json.decision_signals` so local benchmark reports can be compared
    with manifest-based hybrid evaluation.
- Updated real-data benchmark surfaces.
  - `src/core/benchmark/realdata_signals.py` now carries hybrid DXF
    `decision_signals` into the real-data signals component.
  - `src/core/benchmark/realdata_scorecard.py` now carries the same coverage into
    the real-data scorecard component.
- Updated tests for exporter row fields, summary coverage, and downstream real-data
  benchmark propagation.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility

All new fields are additive. When an older classification payload does not include
`DecisionService` output, exporters keep existing columns and emit empty strings or
zero coverage counts for the new decision fields.

## Phase 5 Status

The Phase 5 acceptance target is now covered across analyze, batch classify,
assistant explainability, and benchmark exporters. New model branches can add
evidence through `DecisionService` without requiring benchmark exporter schema
rewrites.
