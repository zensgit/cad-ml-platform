# DEV_DEDUP2D_E2E_WEBHOOK_20251224

## Scope
- Run Dedup2D webhook E2E smoke test (Redis + ARQ + local fake vision + callback).

## Validation
- Command: `.venv/bin/python scripts/e2e_dedup2d_webhook.py`
  - Result: `OK: job completed + callback received + signature verified`.

## Notes
- Script spins up local Redis, API, fake vision, and callback servers with isolated temp storage.
