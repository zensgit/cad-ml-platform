# Eval Reporting Phase 5 Release Draft Payload Merge — Validation

日期：2026-04-03

## Validation Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | publish_result now reads dashboard_payload directly | PASS |
| 2 | draft_payload helper + test deleted | PASS |
| 3 | Workflow no longer materializes draft_payload | PASS |
| 4 | sparse-checkout no longer includes draft_payload script | PASS |
| 5 | 3 draft_payload workflow steps removed | PASS |
| 6 | publish_result workflow step uses dashboardPayloadPath | PASS |
| 7 | publish_result output schema unchanged | PASS |
| 8 | dashboard_payload contract unchanged | PASS |
| 9 | node --check passes | PASS |
| 10 | All tests pass (55/55) | PASS |
| 11 | Zero stale draft_payload/draftPayloadPath references | PASS |

## Test Runs

```
node --check scripts/ci/post_eval_reporting_release_draft_publish.js
# success

python3 -m pytest \
  tests/unit/test_post_eval_reporting_release_draft_publish_js.py \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  tests/unit/test_generate_eval_reporting_dashboard_payload.py -q
55 passed in 10.20s

rg -n "release_draft_payload|draftPayloadPath" \
  .github/workflows/evaluation-report.yml \
  tests/unit/test_evaluation_report_workflow_pages_deploy.py \
  scripts/ci/post_eval_reporting_release_draft_publish.js
# (no output — clean)
```
