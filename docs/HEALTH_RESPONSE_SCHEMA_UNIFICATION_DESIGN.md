# Health Response Schema Unification Design

## Scope
- Define explicit Pydantic response models for `/health` and `/health/extended`.
- Align `/health/extended` with the base `/health` payload while keeping vector/Faiss details.
- Keep resilience payload included when available.

## Problem Statement
- Health endpoints returned untyped dicts and `/health/extended` diverged from the base schema.
- Downstream tooling could not rely on a consistent response contract.

## Design
- Introduce `src/api/health_models.py` with `HealthResponse` and `ExtendedHealthResponse` schemas.
- Update `/health` and `/api/v1/health` to use `HealthResponse` as the response model.
- Update `/health/extended` to return the base health payload plus vector store/Faiss fields.
- Keep resilience data as an optional field on the base payload.

## Impact
- Health responses are now schema-validated while preserving existing fields.
- `/health/extended` gains the base health payload for consistency.

## Validation
- `python3 -m pytest tests/unit/test_main_coverage.py -k "readiness or extended_health" -v`
- `python3 -m pytest tests/unit/test_health_extended_endpoint.py -v`
