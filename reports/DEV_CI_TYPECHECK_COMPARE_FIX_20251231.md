# CI Type-Check Fix for Compare Endpoint (2025-12-31)

## Scope

- Resolve CI mypy failures in `src/api/v1/compare.py`.

## Changes

- Added return type annotation to `compare_features`.
- Added explicit `__init__` return types for vector store classes to satisfy `no-untyped-call`.

## Root Cause

- Mypy disallowed calling untyped constructors (`InMemoryVectorStore`) and missing return type on the compare handler.
