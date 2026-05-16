# CAD ML Assistant Knowledge Citations Development

Date: 2026-05-12

## Goal

Continue Phase 6 by making assistant responses cite the same structured knowledge
evidence and rule metadata emitted by analyze/DecisionService.

## Changes

- Updated `src/api/v1/assistant.py`.
  - Added `QueryKnowledgeCitation`.
  - Extended `QueryExplainability` with:
    - `knowledge_citations`
    - `rule_sources`
    - `rule_versions`
  - Added extraction of rule citations from DecisionService `decision_evidence`.
  - Added `knowledge_rule_citations_grounded` to the assistant decision path when
    analyze knowledge evidence includes rule metadata.
  - Added rule version to the explainability summary when available.
  - Added answer-level citation note appending for responses that carry knowledge
    citations, for example `materials_catalog@knowledge_grounding.v1`.
- Updated `tests/unit/assistant/test_llm_api.py`.
  - The DecisionService metadata test now includes knowledge evidence with
    `rule_sources` and `rule_versions`.
  - The test verifies stable assistant citation fields and answer-level citation
    rendering.
- Updated `config/openapi_schema_snapshot.json`.
  - The response schema now includes the new assistant citation model.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility

The new fields are additive. Assistant responses without DecisionService knowledge
evidence keep the previous answer text and emit empty citation arrays. Existing
clients that read `answer`, `evidence`, or `decision_evidence` are not required to
change.

## Phase 6 Status

Analyze and assistant outputs now share the same rule source/version identifiers when
DecisionService knowledge evidence is present.
