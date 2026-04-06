# Eval Reporting Webhook Signature Policy Alignment — Design

日期：2026-04-01

## Scope

Batch 15B: create `generate_eval_reporting_webhook_signature_policy.py` that reads only the delivery request and produces a signature policy artifact for future signed webhook delivery.

## Design Decisions

### 1. Helper

Reads only `eval_reporting_webhook_delivery_request.json`. Outputs JSON + MD.

All signing defaults are disabled: `signature_policy=disabled_by_default`, `signature_required=false`, `signing_enabled=false`, `signature_requires_explicit_secret=true`.

Declared contract: `signature_algorithm=hmac-sha256`, `signature_header_name=X-Eval-Reporting-Signature`, `signature_canonical_fields` lists the webhook export fields a future signer should use.

### 2. Workflow Steps (deploy-pages)

After delivery request upload, before delivery result: generate policy, append to STEP_SUMMARY, upload. All always-run. Script in sparse-checkout.

### 3. Owner Boundaries

Does NOT perform HMAC signing, read secrets, or modify the sender. Pure policy declaration.
