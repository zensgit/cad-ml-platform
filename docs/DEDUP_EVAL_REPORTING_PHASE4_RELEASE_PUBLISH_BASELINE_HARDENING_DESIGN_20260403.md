# Eval Reporting Phase 4 Release Publish Baseline Hardening — Design

日期：2026-04-03

## Scope

Batch 20B: establish new release baseline after Phase 4 merge, with regression guards preventing `publish_payload` from reappearing and protecting kept surfaces.

## New Release Chain (post-merge)

```
dashboard_payload → draft_payload → publish_result
```

`publish_payload` has been absorbed into `publish_result`. The release chain depth is now 3.

## Hardening Tests Added

4 new regression guard tests:

1. `test_merged_publish_payload_not_in_workflow` — negative guard (step names + sparse-checkout)
2. `test_kept_draft_payload_still_present_after_publish_merge` — positive guard
3. `test_kept_publish_result_still_present_after_publish_merge` — positive guard
4. `test_publish_result_reads_draft_payload_not_publish_payload` — input guard (draftPayloadPath, release_draft_payload.json; no publishPayloadPath, no release_draft_publish_payload.json)

## What Was NOT Changed

- No new artifacts added
- No draft_payload → publish_result deeper merge
- No publish_result output contract changed
