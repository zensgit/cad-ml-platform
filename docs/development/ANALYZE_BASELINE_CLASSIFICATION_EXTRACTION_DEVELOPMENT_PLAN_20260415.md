# Analyze Baseline Classification Extraction Development Plan

Date: 2026-04-15
Owner: Codex
Scope: Extract the remaining baseline L1/L2/L3 classification policy from `src/api/v1/analyze.py`

## Objective

Reduce route-level decision sprawl in `analyze.py` by moving the initial baseline classification policy into a shared classification helper.

## Why This Slice

Phase 3 already extracted:

- final decision contract shaping
- finalization and governance
- Fusion/Hybrid override policy

The biggest remaining route-local classification block was still the baseline path:

- text signal building
- entity count aggregation
- L3 fusion classification
- L2 fusion upgrade over L1 fallback
- initial `cls_payload` creation

This logic belongs in `src/core/classification`, not inside the API route.

## Planned Changes

1. Add `src/core/classification/baseline_policy.py`
2. Export the helper from `src/core/classification/__init__.py`
3. Replace the inlined baseline classification block in `src/api/v1/analyze.py`
4. Add focused unit tests for L1/L2/L3 baseline behavior
5. Re-run the existing fusion integration tests to confirm no behavioral drift

## Expected Result

- `analyze.py` becomes smaller and more orchestration-only
- baseline classification policy becomes reusable and independently testable
- existing Fusion/Hybrid/finalization helpers continue to receive the same initial payload shape
