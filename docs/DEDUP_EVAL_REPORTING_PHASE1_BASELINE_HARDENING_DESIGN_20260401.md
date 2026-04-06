# Eval Reporting Phase 1 Baseline Hardening — Design

日期：2026-04-01

## Scope

Batch 17B: establish new workflow/artifact baseline after Phase 1 removal, with regression guards preventing removed surfaces from reappearing.

## New Baseline

### Remaining deploy-pages eval reporting surfaces (10 artifacts)

1. `eval_reporting_public_index`
2. `eval_reporting_dashboard_payload`
3. `eval_reporting_release_note_snippet`
4. `eval_reporting_release_draft_prefill`
5. `eval_reporting_webhook_export`
6. `eval_reporting_webhook_delivery_request`
7. `eval_reporting_webhook_delivery_result`
8. `eval_reporting_release_draft_payload`
9. `eval_reporting_release_draft_publish_payload`
10. `eval_reporting_release_draft_publish_result`

### Removed (must not reappear)

- `eval_reporting_webhook_signature_policy`
- `eval_reporting_webhook_retry_plan`
- `eval_reporting_release_draft_dry_run`

### Hardening Tests Added

5 new regression guard tests in `test_evaluation_report_workflow_pages_deploy.py`:

1. `test_removed_signature_policy_not_in_workflow` — asserts no step name or sparse-checkout contains `signature_policy`
2. `test_removed_retry_plan_not_in_workflow` — asserts no step name or sparse-checkout contains `retry_plan`
3. `test_removed_dry_run_not_in_workflow` — asserts no step name or sparse-checkout contains `dry_run`
4. `test_kept_delivery_result_still_present` — asserts delivery result generate + upload steps exist
5. `test_kept_publish_result_still_present` — asserts publish result generate + upload steps exist

### What Was NOT Changed

- No new artifacts added
- No merge operations performed
- No public surface contracts changed
