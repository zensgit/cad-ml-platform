# Eval Reporting Webhook Delivery Request Alignment — Validation

日期：2026-04-01

## Validation Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | Delivery request only consumes webhook export | PASS |
| 2 | JSON has all required fields incl. delivery_policy, delivery_allowed, request_body_json | PASS |
| 3 | request_body_json is valid JSON containing webhook export fields | PASS |
| 4 | delivery_policy=disabled_by_default | PASS |
| 5 | delivery_requires_explicit_enable=true | PASS |
| 6 | delivery_allowed=true when export exists, false when missing | PASS |
| 7 | MD contains Webhook Event / Delivery Policy / Release readiness / Landing Page / Static / Interactive | PASS |
| 8 | Script does not import upstream generators | PASS |
| 9 | Workflow has always-run generate step after webhook export | PASS |
| 10 | Workflow has always-run STEP_SUMMARY append step | PASS |
| 11 | Workflow has always-run upload step | PASS |
| 12 | Script in sparse-checkout | PASS |
| 13 | All py_compile checks pass | PASS |
| 14 | All tests pass (58/58) | PASS |

## Test Runs

```
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/generate_eval_reporting_webhook_delivery_request.py \
  tests/unit/test_generate_eval_reporting_webhook_delivery_request.py
# success

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_webhook_delivery_request.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
58 passed in 11.30s
```
