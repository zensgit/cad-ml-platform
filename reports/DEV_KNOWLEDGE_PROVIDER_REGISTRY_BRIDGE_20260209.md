# DEV_KNOWLEDGE_PROVIDER_REGISTRY_BRIDGE_20260209

Date: 2026-02-09

## Summary

Added a minimal `knowledge/*` domain to the core provider registry so built-in knowledge modules participate in:
- provider registry snapshots (`/api/v1/*/providers/registry`)
- best-effort health checks (`/api/v1/*/providers/health`)
- readiness selection (`/ready` with `READINESS_*_PROVIDERS`)

This is intentionally lightweight: query surfaces remain in dedicated HTTP APIs under `src/api/v1/` (for example `/api/v1/tolerance/*`, `/api/v1/standards/*`).

## Changes

- Added knowledge providers:
  - `src/core/providers/knowledge.py`
    - `knowledge/tolerance` (probes ISO286 tables via deterministic lookups)
    - `knowledge/standards` (probes thread database via deterministic lookup)
- Bootstrapped knowledge providers in the core registry:
  - `src/core/providers/bootstrap.py`
- Exported knowledge bootstrap in provider package:
  - `src/core/providers/__init__.py`
- Updated provider docs to list the new built-in providers:
  - `docs/PROVIDER_FRAMEWORK.md`
- Updated bootstrap expectations:
  - `tests/unit/test_provider_registry_bootstrap.py`
- Added unit coverage for the knowledge bridges:
  - `tests/unit/test_provider_framework_knowledge_bridge.py`

## Validation

- `pytest tests/unit/test_provider_framework_knowledge_bridge.py -q` (passed)
- `pytest tests/unit/test_provider_registry_bootstrap.py -q` (passed)

