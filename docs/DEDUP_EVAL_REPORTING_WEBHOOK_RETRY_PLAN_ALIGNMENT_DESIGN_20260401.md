# Eval Reporting Webhook Retry Plan Alignment — Design

日期：2026-04-01

## Scope

Batch 15A: create `generate_eval_reporting_webhook_retry_plan.py` that reads only the delivery result and produces a retry / dead-letter plan artifact.

## Design Decisions

### 1. Helper

Reads only `eval_reporting_webhook_delivery_result.json`. Outputs JSON + MD.

Retry logic derivation:
- Succeeded → `no_retry`, no dead-letter
- Attempted + retry_recommended → `manual_or_future_queue`, retry_after=300s
- Attempted + not retry_recommended → dead_letter_recommended=true
- Not attempted → `no_retry`, no dead-letter

### 2. Fields

All required fields per plan: `status`, `surface_kind`, `generated_at`, `release_readiness`, `delivery_mode`, `delivery_attempted`, `delivery_succeeded`, `http_status`, `delivery_error`, `retry_recommended`, `retry_policy`, `retry_after_seconds`, `retry_reason`, `dead_letter_recommended`, `dead_letter_reason`, `retry_queue_target_kind`, `retry_requires_explicit_enable`, `source_delivery_result_surface_kind`.

### 3. Owner Boundaries

Does NOT re-deliver, create queues, or read upstream artifacts.
