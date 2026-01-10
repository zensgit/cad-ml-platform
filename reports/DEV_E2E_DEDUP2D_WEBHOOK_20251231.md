# Dedup2D Webhook E2E Smoke (2025-12-31)

## Scope

- Run the end-to-end Dedup2D webhook smoke test with Redis + ARQ + local callback.

## Command

- `./.venv/bin/python scripts/e2e_dedup2d_webhook.py`

## Results

- OK: job completed, callback received, signature verified.

## Notes

- Script uses ephemeral ports and temporary directories; all resources cleaned on exit.
