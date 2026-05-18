# CAD ML Manufacturing Evidence Development

Date: 2026-05-12

## Goal

Close the remaining Phase 6 analyze-evidence gap by connecting manufacturing
outputs into the same structured evidence surface used by classification decisions.

## Changes

- Updated `src/core/process/manufacturing_summary.py`.
  - Added `build_manufacturing_evidence`.
  - Converts DFM, process, cost, and manufacturing summary payloads into
    DecisionService-compatible evidence rows.
  - Emits sources:
    - `dfm`
    - `manufacturing_process`
    - `manufacturing_cost`
    - `manufacturing_decision`
- Updated `src/core/analysis_manufacturing_summary.py`.
  - Writes `results["manufacturing_evidence"]` whenever manufacturing outputs exist.
  - Appends the same rows to `results["classification"]["evidence"]` when
    classification is enabled.
  - Keeps `classification["decision_contract"]["evidence"]` aligned with the
    top-level classification evidence list.
- Updated `src/core/process/__init__.py` to export `build_manufacturing_evidence`.
- Updated tests for unit and analyze integration coverage.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md` to mark
  the Phase 6 analyze-evidence item complete.

## Evidence Contract

The new manufacturing rows use the existing evidence row shape:

- `source`
- `kind`
- `label`
- `confidence`
- `status`
- `details`

Mapping:

- DFM quality results become `source=dfm`, `kind=manufacturability_check`.
- Process recommendations become `source=manufacturing_process`,
  `kind=process_recommendation`.
- Cost estimates become `source=manufacturing_cost`, `kind=cost_estimate`.
- The combined manufacturing decision summary becomes
  `source=manufacturing_decision`, `kind=manufacturing_summary`.

## Compatibility

The change is additive:

- Existing `results["quality"]`, `results["process"]`, `results["cost_estimation"]`,
  and `results["manufacturing_decision"]` payloads are preserved.
- Analyze calls with classification disabled still receive
  `results["manufacturing_evidence"]`.
- Analyze calls with classification enabled additionally expose the same rows through
  `classification.evidence` and `classification.decision_contract.evidence`.

## Remaining Work

- Add real DXF/OCR fixtures for manufacturing evidence, not only stubbed integration
  probes.
- Add category-level benchmark precision/recall for manufacturing evidence extraction.
- Feed manufacturing evidence quality into the forward scorecard with real fixture
  counts. The scorecard input is now wired; the remaining work is to replace synthetic
  ready fixtures with real benchmark summaries.
