# Eval Reporting Phase 1 Zero-Consumer Surface Removal — Design

日期：2026-04-01

## Scope

Batch 17A: remove 3 zero-consumer surfaces identified in Batch 16 inventory. No merges.

## Removed Artifacts

| Artifact | Helper/Consumer | Tests | Workflow Steps |
|---|---|---|---|
| `eval_reporting_webhook_signature_policy` | `generate_eval_reporting_webhook_signature_policy.py` | `test_generate_eval_reporting_webhook_signature_policy.py` | 3 (generate + append + upload) + sparse-checkout entry |
| `eval_reporting_webhook_retry_plan` | `generate_eval_reporting_webhook_retry_plan.py` | `test_generate_eval_reporting_webhook_retry_plan.py` | 3 + sparse-checkout entry |
| `eval_reporting_release_draft_dry_run` | `post_eval_reporting_release_draft_dry_run.js` | `test_post_eval_reporting_release_draft_dry_run_js.py` | 3 + sparse-checkout entry |

## What Was NOT Changed

- `eval_reporting_webhook_delivery_result` — kept (action result with real retry fields)
- `eval_reporting_release_draft_publish_result` — kept (action result)
- `eval_reporting_webhook_delivery_request` — kept (delivery surface, still reads webhook_export)
- No merge operations performed
- No public surface contracts changed
