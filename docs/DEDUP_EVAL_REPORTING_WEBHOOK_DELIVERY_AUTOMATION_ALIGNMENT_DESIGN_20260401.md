# Eval Reporting Webhook Delivery Automation Alignment — Design

日期：2026-04-01

## Scope

Batch 14B: create `post_eval_reporting_webhook_delivery.js` that reads only the delivery request and optionally sends to an external webhook. Default is disabled.

## Design Decisions

### 1. JS Module

Reads only `eval_reporting_webhook_delivery_request.json`. Produces result JSON + MD.

Key functions:
- `buildDeliveryResult({request, deliveryEnabled, webhookUrl})` — pure result builder
- `postWebhookDelivery({...})` — workflow entry, optionally sends HTTP POST

### 2. Delivery Gating

| deliveryEnabled | delivery_allowed | webhookUrl | delivery_mode |
|---|---|---|---|
| false (default) | any | any | `disabled` |
| true | true | set | `deliver` |
| true | false | any | `blocked` |

Default: `deliveryEnabled: false` + `webhookUrl: ''` hardcoded in workflow.

### 3. Fail-Soft & Retry Fields

HTTP errors and timeouts are caught and recorded in `delivery_error`, `retry_recommended`, `retry_hint`. The module never retries — it only produces stable fields for a future retry surface.

### 4. Workflow Steps (deploy-pages)

After delivery request upload: generate result (github-script), append to STEP_SUMMARY, upload. All always-run + continue-on-error. JS in sparse-checkout.

### 5. Owner Boundaries

Does NOT read webhook export or upstream. Does NOT implement retry loops or queues.
