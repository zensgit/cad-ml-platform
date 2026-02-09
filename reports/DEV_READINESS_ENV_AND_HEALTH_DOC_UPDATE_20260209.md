# DEV_READINESS_ENV_AND_HEALTH_DOC_UPDATE_20260209

Date: 2026-02-09

## Summary

Aligned operational documentation and environment templates with the current provider registry footprint (including the `knowledge/*` domain) and documented readiness-related environment variables.

## Changes

- Updated `/health` configuration visibility doc to reflect the current core provider registry snapshot shape (domains now include `classifier`, `knowledge`, `ocr`, `vision`):
  - `docs/HEALTH_ENDPOINT_CONFIG.md`
- Added provider registry and readiness env variables to the environment example template:
  - `PROVIDER_REGISTRY_CACHE_ENABLED`
  - `READINESS_CHECK_TIMEOUT_SECONDS`
  - `READINESS_REQUIRED_PROVIDERS`
  - `READINESS_OPTIONAL_PROVIDERS`
  - `\.env.example`

## Validation

Documentation-only change; existing provider registry bootstrap and knowledge bridge unit suites remain the source of truth for behavior:

- `pytest tests/unit/test_provider_registry_bootstrap.py -q`
- `pytest tests/unit/test_provider_framework_knowledge_bridge.py -q`

