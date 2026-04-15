# Phase 3 Active Learning Review Extraction Development Plan

Date: 2026-04-15
Owner: Codex
Scope: Extract active-learning review dispatch from `src/api/v1/analyze.py`

## Objective

Reduce route-local side-effect logic in `analyze.py` by moving the finalized
classification review dispatch into a reusable helper.

## Why This Slice

After baseline policy, override policy, and finalization were extracted, the
remaining route-local decision tail still included one non-trivial side effect:

- gate on `ACTIVE_LEARNING_ENABLED`
- consume finalized `needs_review` contract
- derive `sample_type`
- derive `uncertainty_reason`
- assemble `score_breakdown`
- call `get_active_learner().flag_for_review(...)`

This is a stable downstream policy boundary and no longer belongs in the route.

## Planned Changes

1. Add `src/core/classification/active_learning_policy.py`
2. Export the helper from `src/core/classification/__init__.py`
3. Replace the inline review-dispatch block in `src/api/v1/analyze.py`
4. Add focused unit tests for dispatch gating and sample-type priority
5. Re-run the existing fusion integration tests to confirm no behavioral drift

## Expected Result

- `analyze.py` becomes more orchestration-only
- active-learning dispatch semantics become independently testable
- finalized review fields remain the single source of truth
- route code stops rebuilding `sample_type` and `score_breakdown` inline
