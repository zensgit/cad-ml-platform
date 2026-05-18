# CAD ML DecisionService Batch Classify Development

Date: 2026-05-12

## Goal

Continue Phase 5 by routing batch classification success results through the shared
`DecisionService` contract while preserving existing batch response fields.

## Changes

- Updated `src/core/classification/batch_classify_pipeline.py`.
  - `build_batch_classify_item` now builds a DecisionService payload for successful
    batch classifications.
  - Existing fields remain stable:
    - `category`
    - `fine_category`
    - `coarse_category`
    - `is_coarse_label`
    - `confidence`
    - `classifier`
  - New decision fields are attached:
    - `fine_part_type`
    - `coarse_part_type`
    - `decision_source`
    - `branch_conflicts`
    - `evidence`
    - `review_reasons`
    - `fallback_flags`
    - `contract_version`
    - `decision_contract`
  - Classifier-provided `needs_review` and `review_reason` are preserved and merged
    with DecisionService review governance.
- Updated `src/api/v1/analyze_live_models.py`.
  - `BatchClassifyResultItem` now exposes the stable decision contract fields.
- Updated tests.
  - Unit tests verify the DecisionService contract on direct batch item creation.
  - Batch pipeline tests verify V6 fallback and V16 batch paths emit the contract.
  - API tests verify response-model exposure.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Compatibility

The batch API remains backward-compatible for existing consumers that read
`category`, `fine_category`, or `coarse_category`. The new fields are additive and
mirror the analyze decision contract shape.

## Remaining Phase 5 Work

- Route assistant explanation through the shared decision evidence contract.
- Route benchmark exporters through the shared decision evidence contract.
