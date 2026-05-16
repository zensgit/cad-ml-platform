# CAD ML Knowledge Fixtures Development

Date: 2026-05-12

## Goal

Close the Phase 6 fixture gap for knowledge-grounded manufacturing intelligence.
The knowledge summary now emits structured, rule-versioned checks for five
manufacturing cases that must be explainable through analyze, assistant, and
scorecard evidence paths.

## Scope

Updated `src/core/knowledge/analysis_summary.py` with additive extraction for:

- material substitution, backed by material classification and equivalence lookup
- H7/g6 style ISO 286 fit validation, backed by the existing fit catalog
- surface finish recommendation context for Ra/N-grade drawing notes
- machining process route extraction from drawing/process text
- manufacturability risk extraction from text and geometry signals

Updated `tests/unit/test_knowledge_analysis_summary.py` with dedicated fixtures for:

- `material_substitution`
- `fit_validation`
- `surface_finish_recommendation`
- `machining_process_route`
- `manufacturability_risk`

Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md` to mark the
five Phase 6 fixtures complete.

## Contract

All new rows keep the existing knowledge output contract:

- `category`
- `item`
- `value`
- `confidence`
- `source`
- `status`
- `rule_source`
- `rule_version`

New rule sources:

- `iso286_fit_catalog`
- `machining_process_knowledge_base`
- `dfm_manufacturability_rules`

Existing source mappings are reused for:

- `materials_catalog`
- `iso1302_surface_finish_catalog`

## Notes

This slice is intentionally lightweight. It does not replace full DFM/process/cost
engines. It gives the shared evidence contract stable fixture coverage so future
analyze and assistant improvements can rely on typed knowledge rows instead of
free text only.

## Remaining Phase 6 Work

- Connect full process, cost, and DFM engine outputs into analyze evidence.
- Expand fixture samples from text-only cases to real DXF/OCR drawings.
- Add benchmark metrics for per-category knowledge extraction precision.
