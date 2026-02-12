# DEV_CUSTOM_CLASSIFIER_PROVIDER_WORKFLOW_20260210

## Goal
Provide a stable integration workflow for “another window” classifier/model development by
plugging models into the existing **core provider framework** (`src/core/providers/`) instead
of wiring them directly into endpoints.

This reduces merge risk and makes rollout/rollback controllable via env flags.

## Current State (Already In Main)
The repo already supports a provider-based classifier integration path:
- Provider framework:
  - `src/core/providers/`
- Classifier providers (domain=`classifier`):
  - `src/core/providers/classifier.py`
    - `classifier/hybrid`
    - `classifier/graph2d`
    - `classifier/graph2d_ensemble`
    - `classifier/v16`
    - `classifier/v6`

Analyze endpoint supports shadow-only invocation:
- `PART_CLASSIFIER_PROVIDER_ENABLED`
- `PART_CLASSIFIER_PROVIDER_NAME`
Documented in:
- `docs/ANALYZE_CLASSIFICATION_FIELDS.md`
- `.env.example`

## Recommended Workflow For A New Model
1. Implement your model in an isolated module (keep heavy deps optional):
   - Example: `src/ml/part_classifier_v17.py` (lazy import torch inside functions)
2. Add a provider adapter under `src/core/providers/classifier.py`:
   - Implement `BaseProvider[ClassifierProviderConfig, Dict[str, Any]]`
   - Register it:
     - `@ProviderRegistry.register("classifier", "<new_name>")`
3. Roll out behind env flags:
   - `PART_CLASSIFIER_PROVIDER_ENABLED=true`
   - `PART_CLASSIFIER_PROVIDER_NAME=<new_name>`
4. Keep it shadow-only until validated:
   - Do not override `results.classification.part_type`
   - Emit additive fields under `classification.part_classifier_prediction` and normalized `part_family*`
5. Add 2 tests:
   - Provider bootstrap/registry test:
     - ensures the provider appears in registry snapshot
   - Analyze integration test (TestClient):
     - ensures the additive output fields exist and the endpoint remains backward compatible

## Verification Hooks
Provider registry snapshot:
- `/api/v1/health/providers/registry`
  - confirms which providers are registered + their class paths

Readiness checks:
- `/ready`
  - validates provider health without forcing heavyweight model loads

## Why This Matters
- Avoids endpoint-level coupling and merge conflicts.
- Makes multi-model experiments safe (shadow + flags).
- Keeps production truth stable while iterating on ML.

