# DEV_PROVIDER_REGISTRY_SNAPSHOT_CLASS_METADATA_20260208

## Goal
Make the core provider framework easier to debug by exposing **which concrete provider
classes** were registered for each `domain/provider` in the registry snapshot.

This is useful when multiple providers exist for the same domain (e.g. `classifier/*`)
and you want to quickly confirm what implementation is active in a given runtime.

## Changes
- `src/core/providers/bootstrap.py`
  - Added `provider_classes` to the registry snapshot:
    - `provider_classes[domain][provider] = "<module>.<qualname>"`
  - Best-effort only: snapshot never fails if a class cannot be resolved.

- `tests/unit/test_provider_registry_bootstrap.py`
  - Added assertions that `provider_classes` exists and includes `classifier/hybrid`.

## Verification
Command:
```bash
python3 -m pytest -q tests/unit/test_provider_registry_bootstrap.py
```

Result: `3 passed`

