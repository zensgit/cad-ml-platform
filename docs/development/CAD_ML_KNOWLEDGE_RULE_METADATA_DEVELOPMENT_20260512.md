# CAD ML Knowledge Rule Metadata Development

Date: 2026-05-12

## Goal

Start Phase 6 by making manufacturing knowledge outputs auditable. Every knowledge
check, standards candidate, violation, and hint now carries stable rule source and
rule version metadata.

## Changes

- Updated `src/core/knowledge/analysis_summary.py`.
  - Added `KNOWLEDGE_RULE_VERSION = "knowledge_grounding.v1"`.
  - Added source mapping for:
    - thread standards
    - general tolerances
    - IT grades
    - surface finish
    - GD&T
    - materials catalog
    - knowledge-manager hints
  - Added metadata enrichment helpers so all returned knowledge rows include:
    - `rule_source`
    - `rule_version`
  - Moved coarse-label normalization behind a lazy helper to avoid an import cycle
    between knowledge summary and classification package initialization.
- Updated `src/core/classification/decision_service.py`.
  - Knowledge evidence details now include:
    - `knowledge_hints_count`
    - `check_categories`
    - `standards_candidate_types`
    - `rule_sources`
    - `rule_versions`
- Updated tests.
  - Unit coverage verifies rule metadata on checks, standards candidates,
    violations, and hints.
  - DecisionService tests verify rule metadata survives into shared evidence.
  - Analyze integration coverage verifies API classification payloads expose the
    metadata and the knowledge evidence includes `knowledge_grounding.v1`.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility

The new fields are additive. Existing consumers that read `category`, `type`,
`designation`, `label`, or `score` continue to work. Exporters that serialize the
knowledge rows automatically preserve the new metadata without schema rewrites.

## Remaining Phase 6 Work

- Connect process, cost, and DFM checks into the same analyze evidence path.
- Make assistant answers cite the same structured evidence used by analyze.
- Add knowledge grounding coverage to the forward scorecard.
