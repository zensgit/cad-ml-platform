# CAD ML DecisionService Development

Date: 2026-05-12

## Goal

Start Phase 5 by adding a stable `DecisionService` boundary for final CAD
classification decisions.

## Changes

- Added `src/core/classification/decision_service.py`.
  - Defines `DECISION_CONTRACT_VERSION=classification_decision.v1`.
  - Wraps the existing `finalize_classification_payload` behavior.
  - Emits `decision_contract`.
  - Emits normalized `evidence` rows.
  - Emits stable `fallback_flags`.
- Updated `src/core/classification/classification_pipeline.py`.
  - Replaced direct finalize call with `DecisionService.decide`.
  - Preserved the existing finalize function as the injected implementation, so the
    current monkeypatch/testing surface remains narrow.
  - Passes B-Rep `features_3d` into the decision boundary.
- Updated `src/core/classification/__init__.py`.
  - Exports `DecisionService` and `DECISION_CONTRACT_VERSION`.
- Added `tests/unit/test_decision_service.py`.
  - Covers stable contract output.
  - Covers normalized evidence for baseline, filename, Graph2D, Hybrid, B-Rep,
    knowledge, vector neighbors, and active-learning history.
  - Covers fallback flags.
- Updated the Phase 5 TODO.
- Added this design/development/verification documentation set.

## Current Scope

Integrated now:

- analyze classification pipeline.

Explicit follow-up:

- batch classify.
- assistant explanation.
- benchmark exporters.

## Reasoning

The existing code already had finalization, review governance, and a label decision
contract. This slice keeps those semantics and adds a service boundary around them
instead of rewriting branch priority. That makes the first Phase 5 change testable
without changing production decision behavior.
