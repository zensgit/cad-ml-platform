# Eval Reporting Phase 2 Webhook Baseline Hardening — Design

日期：2026-04-03

## Scope

Batch 18B: establish new webhook baseline after Phase 2 merge, with regression guards preventing `webhook_export` from reappearing and protecting kept surfaces.

## New Webhook Baseline

### Remaining webhook chain (post-merge)

```
dashboard_payload → delivery_request → delivery_result
```

`webhook_export` has been absorbed into `delivery_request`. The chain depth is now 3 (was 5 pre-Phase 1, 4 post-Phase 1).

### Hardening Tests Added

4 new regression guard tests:

1. `test_merged_webhook_export_not_in_workflow` — asserts no step name or sparse-checkout contains `webhook_export`
2. `test_kept_delivery_request_still_present_after_merge` — asserts delivery_request generate + upload steps exist
3. `test_kept_delivery_result_still_present_after_merge` — asserts delivery_result generate + upload steps exist
4. `test_delivery_request_reads_dashboard_payload_not_webhook_export` — asserts `--dashboard-payload-json` in step, `--webhook-export-json` not in step

### What Was NOT Changed

- No new artifacts added
- No release chain merge performed
- No delivery_result contract changed
