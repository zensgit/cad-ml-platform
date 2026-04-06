# Eval Reporting Release Draft Publish Payload Alignment — Validation

日期：2026-04-01

## Validation Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | Publish payload only consumes release-draft payload | PASS |
| 2 | publish_policy=disabled_by_default | PASS |
| 3 | publish_allowed=true only when readiness=ready | PASS |
| 4 | publish_requires_explicit_enable=true always | PASS |
| 5 | Unavailable status when no input | PASS |
| 6 | Script does not import upstream generators | PASS |
| 7 | MD contains Eval Reporting / Release readiness / Publish Policy / Landing / Static / Interactive | PASS |
| 8 | Workflow has always-run generate step after dry-run | PASS |
| 9 | Workflow has always-run STEP_SUMMARY append step | PASS |
| 10 | Workflow has always-run upload step | PASS |
| 11 | Script in sparse-checkout | PASS |
| 12 | All py_compile checks pass | PASS |
| 13 | All tests pass (49/49) | PASS |

## Test Runs

```
PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m py_compile \
  scripts/ci/generate_eval_reporting_release_draft_publish_payload.py \
  tests/unit/test_generate_eval_reporting_release_draft_publish_payload.py
# success

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_generate_eval_reporting_release_draft_publish_payload.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
49 passed in 22.45s
```
