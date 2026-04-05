# Eval Reporting Webhook Signature Policy Alignment — Validation

日期：2026-04-01

## Validation Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | Signature policy only consumes delivery request | PASS |
| 2 | signature_policy=disabled_by_default | PASS |
| 3 | signature_required=false, signing_enabled=false | PASS |
| 4 | signature_requires_explicit_secret=true | PASS |
| 5 | signature_algorithm=hmac-sha256, header=X-Eval-Reporting-Signature | PASS |
| 6 | signature_canonical_fields is a list including webhook_event_type | PASS |
| 7 | Script does not import upstream generators or signing modules | PASS |
| 8 | MD contains Signature Policy / Required / Algorithm / Header / Delivery Method | PASS |
| 9 | Workflow has always-run generate step after delivery request | PASS |
| 10 | Workflow has always-run STEP_SUMMARY append | PASS |
| 11 | Workflow has always-run upload step | PASS |
| 12 | Script in sparse-checkout | PASS |
| 13 | All py_compile checks pass | PASS |
| 14 | All Batch 15B tests pass (73/73) | PASS |
| 15 | Full regression passes (90/90) | PASS |

## Test Runs

```
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/generate_eval_reporting_webhook_signature_policy.py \
  tests/unit/test_generate_eval_reporting_webhook_signature_policy.py
# success

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_webhook_signature_policy.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
73 passed in 9.19s

# Full regression
90 passed in 8.95s
```
