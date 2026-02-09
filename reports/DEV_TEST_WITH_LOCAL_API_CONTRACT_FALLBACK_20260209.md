# DEV_TEST_WITH_LOCAL_API_CONTRACT_FALLBACK_20260209

## Goal
Make the canonical tiered runner `scripts/test_with_local_api.sh --suite contract`
usable in environments where binding a local TCP port is not permitted (e.g. sandboxed
dev shells), while keeping strict behavior in CI.

## Problem
In some environments, uvicorn fails to bind to `127.0.0.1:8000` with:
- `operation not permitted`

This prevented `--suite contract` from running, even though the contract tests
can execute in-process via `fastapi.testclient.TestClient` (manual contract assertions)
and already include a live-server probe.

## Changes
File: `scripts/test_with_local_api.sh`
- Added `CONTRACT_INPROCESS_FALLBACK` env (default `auto`):
  - `auto`: allow contract-only fallback when not in CI
  - `true`: always allow fallback for contract suite
  - `false`: never allow fallback
- When starting the local API fails or times out and the failure indicates
  a port-binding restriction, the runner:
  - prints the last uvicorn logs
  - continues to run `tests/contract` without a live server
  - relies on `tests/contract/test_api_contract.py` TestClient fallback for the
    manual contract assertions

Safety:
- In CI (`CI` env set), fallback is disabled by default.
- Fallback is only triggered on known bind-permission errors (or empty logs).

## Validation
Command run:
```bash
bash scripts/test_with_local_api.sh --suite contract --wait-seconds 5
```

Observed behavior:
- uvicorn startup reached application startup completion
- bind failed with `operation not permitted`
- runner emitted fallback warnings and executed contract tests
- result: `13 passed, 4 skipped`

