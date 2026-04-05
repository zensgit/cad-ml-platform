# Eval Reporting Release Draft Dry Run Surface Alignment — Validation

日期：2026-04-01

## Validation Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | Dry-run only consumes release-draft payload | PASS |
| 2 | Default mode is dry_run (publishEnabled=false) | PASS |
| 3 | publish_allowed=true only when enabled AND ready | PASS |
| 4 | publish_mode=blocked when enabled but degraded | PASS |
| 5 | JSON has all required fields | PASS |
| 6 | MD contains Dry Run / Publish Enabled / Release readiness / Landing / Static / Interactive | PASS |
| 7 | JS module does not reference upstream generators | PASS |
| 8 | Workflow has always-run generate step with publishEnabled: false | PASS |
| 9 | Workflow has always-run STEP_SUMMARY append | PASS |
| 10 | Workflow has always-run upload step | PASS |
| 11 | JS file in sparse-checkout | PASS |
| 12 | Steps after release draft payload | PASS |
| 13 | node --check passes | PASS |
| 14 | All Batch 12B tests pass (45/45) | PASS |
| 15 | Full regression passes (57/57) | PASS |

## Test Runs

```
node --check scripts/ci/post_eval_reporting_release_draft_dry_run.js
# success

PYTHONPYCACHEPREFIX=/tmp/pycache python3 -m pytest \
  tests/unit/test_post_eval_reporting_release_draft_dry_run_js.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py -q
45 passed in 6.39s

# Full regression
57 passed in 5.58s
```
