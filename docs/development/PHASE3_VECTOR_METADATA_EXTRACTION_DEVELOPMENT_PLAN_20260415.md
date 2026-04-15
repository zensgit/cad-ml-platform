# Phase 3 Vector Metadata Extraction Development Plan

## Goal
- Extract the vector registration metadata assembly from `src/api/v1/analyze.py` into a shared helper.

## Scope
- Move the classification-aware vector metadata construction into `src/core/classification/vector_metadata.py`.
- Keep `analyze.py` responsible for feature-vector assembly and backend registration only.
- Preserve the existing vector metadata schema used by memory and Qdrant backends.

## Planned Changes
- Add `build_vector_registration_metadata(...)` for vector registration payload assembly.
- Add a shared `extract_vector_label_contract(...)` helper for vector metadata reads.
- Update `src/api/v1/analyze.py` to use the new builder.
- Keep `src/core/similarity.extract_vector_label_contract(...)` as a compatibility wrapper.
- Add unit coverage for helper output and compatibility behavior.

## Risk Controls
- Preserve existing metadata keys:
  - `part_type`
  - `fine_part_type`
  - `coarse_part_type`
  - `final_decision_source`
  - `is_coarse_label`
- Preserve base vector metadata keys and L3 dimension handling.
- Avoid changing API- or store-facing import paths by keeping the `similarity.py` wrapper.
