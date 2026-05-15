# CAD ML DecisionService Assistant Explanation Development

Date: 2026-05-12

## Goal

Continue Phase 5 by making assistant explainability consume the shared
`DecisionService` decision contract and evidence when they are present in assistant
response metadata.

## Changes

- Updated `src/api/v1/assistant.py`.
  - `QueryExplainability` now exposes:
    - `contract_version`
    - `decision_contract`
    - `decision_evidence`
    - `fallback_flags`
    - `review_reasons`
  - `_build_query_explainability` reads `metadata.decision_contract`.
  - It reads decision evidence from `metadata.decision_evidence` or
    `metadata.decision_contract.evidence`.
  - Decision evidence source contributions now override retrieval-only source
    contributions when present.
  - The explainability decision path records:
    - `decision_contract_loaded`
    - `decision_evidence_grounded`
  - Review reasons, branch conflicts, and fallback flags are reflected in uncertainty.
- Updated `tests/unit/assistant/test_llm_api.py`.
  - Added a focused test proving assistant explainability consumes
    `classification_decision.v1`.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility

Existing retrieval-grounded assistant responses continue to work. The new fields are
additive and remain empty or `null` when no DecisionService contract is present.

## Remaining Phase 5 Work

- Route benchmark exporters through the shared decision evidence contract.
