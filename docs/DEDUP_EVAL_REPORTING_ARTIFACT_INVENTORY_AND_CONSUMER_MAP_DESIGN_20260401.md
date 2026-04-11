# Eval Reporting Artifact Inventory and Consumer Map — Design

日期：2026-04-01

## Scope

Batch 16A: complete inventory of all eval reporting artifacts with owner, consumer, classification, and recommended action. No code changes.

---

## Artifact Inventory

### evaluate job artifacts

| # | Artifact | Produced By | Primary Input | Current Consumer(s) | Classification | Recommended Action |
|---|---|---|---|---|---|---|
| 1 | `eval_reporting_bundle.json` | `generate_eval_reporting_bundle.py` | sub-bundles + reports | health checker, index, helpers, landing page | **owner** | **keep** |
| 2 | `eval_reporting_bundle_health_report.json` | `check_eval_reporting_bundle_health.py` | bundle manifest | index, stack summary, landing page | **owner** | **keep** |
| 3 | `eval_reporting_index.json` | `generate_eval_reporting_index.py` | bundle manifest | stack summary, landing page, public index, PR comment | **public_surface** | **keep** |
| 4 | `eval_reporting_stack_summary.json` | `summarize_eval_reporting_stack_status.py` | bundle + health + index | landing page, notify, release summary, PR comment, public index, dashboard payload | **public_surface** | **keep** |
| 5 | `eval_reporting_release_summary.json` | `generate_eval_reporting_release_summary.py` | index + stack summary | dashboard payload | **public_surface** | **keep** |
| 6 | `index.html` (landing page) | `generate_eval_reporting_landing_page.py` | index + stack summary + health | Pages root, human entry point | **public_surface** | **keep** |
| 7 | `report_static/index.html` | `generate_eval_report.py` | eval signal summary | human consumer | **owner** | **keep** |
| 8 | `report_interactive/index.html` | `generate_eval_report_v2.py` | eval signal summary | human consumer | **owner** | **keep** |

### deploy-pages job artifacts

| # | Artifact | Produced By | Primary Input | Current Consumer(s) | Classification | Recommended Action |
|---|---|---|---|---|---|---|
| 9 | `eval_reporting_public_index.json` | `generate_eval_reporting_public_index.py` | index + stack summary + page_url | dashboard payload | **public_surface** | **keep** |
| 10 | `eval_reporting_dashboard_payload.json` | `generate_eval_reporting_dashboard_payload.py` | release summary + public index | release note snippet, webhook export | **delivery_surface** | **keep** |
| 11 | `eval_reporting_release_note_snippet.json` | `generate_eval_reporting_release_note_snippet.py` | dashboard payload | release draft prefill only | **thin_pass_through** | **merge** into dashboard payload |
| 12 | `eval_reporting_release_draft_prefill.json` | `generate_eval_reporting_release_draft_prefill.py` | release note snippet | release draft payload only | **thin_pass_through** | **merge** into dashboard payload → draft payload |
| 13 | `eval_reporting_webhook_export.json` | `generate_eval_reporting_webhook_export.py` | dashboard payload | delivery request only | **thin_pass_through** | **merge** into dashboard payload |
| 14 | `eval_reporting_webhook_delivery_request.json` | `generate_eval_reporting_webhook_delivery_request.py` | webhook export | delivery sender, signature policy | **delivery_surface** | **keep** (but absorb webhook export) |
| 15 | `eval_reporting_webhook_signature_policy.json` | `generate_eval_reporting_webhook_signature_policy.py` | delivery request | **no real consumer** (future intent only) | **thin_pass_through** | **remove** (or defer to when signer exists) |
| 16 | `eval_reporting_webhook_delivery_result.json` | `post_eval_reporting_webhook_delivery.js` | delivery request | retry plan only | **action_result** | **keep** |
| 17 | `eval_reporting_webhook_retry_plan.json` | `generate_eval_reporting_webhook_retry_plan.py` | delivery result | **no real consumer** (future intent only) | **thin_pass_through** | **remove** (or defer to when retry queue exists) |
| 18 | `eval_reporting_release_draft_payload.json` | `generate_eval_reporting_release_draft_payload.py` | release draft prefill | dry run, publish payload | **thin_pass_through** | **merge** with prefill into single draft payload |
| 19 | `eval_reporting_release_draft_dry_run.json` | `post_eval_reporting_release_draft_dry_run.js` | release draft payload | **no real consumer** (future intent only) | **thin_pass_through** | **remove** (publish result already covers this) |
| 20 | `eval_reporting_release_draft_publish_payload.json` | `generate_eval_reporting_release_draft_publish_payload.py` | release draft payload | publish automation only | **thin_pass_through** | **merge** into publish result |
| 21 | `eval_reporting_release_draft_publish_result.json` | `post_eval_reporting_release_draft_publish.js` | publish payload | **no real consumer** | **action_result** | **keep** (but absorb publish payload) |

---

## High-Duplication Chains

### Release chain (4 intermediate layers)

```
dashboard_payload → release_note_snippet → release_draft_prefill → release_draft_payload → release_draft_publish_payload → publish_result
```

**Finding:** Between dashboard_payload and publish_result there are 4 intermediate artifacts (snippet, prefill, draft_payload, publish_payload), each a single-file-in → single-file-out transform copying the same readiness/title/body/URLs. Additionally, dry_run sits as a side branch off draft_payload with no downstream consumer. A single `release_draft_publish_result` consuming `dashboard_payload` directly would eliminate these intermediaries.

### Webhook chain (2 intermediate layers + 3 terminal branches)

```
dashboard_payload → webhook_export → delivery_request → {signature_policy, delivery_result → retry_plan}
```

Between dashboard_payload and the terminal artifacts there are 2 intermediate layers (webhook_export, delivery_request), then 3 terminal branches (signature_policy, delivery_result, retry_plan).

**Finding:** webhook_export is a near-identical copy of dashboard_payload. delivery_request adds only policy fields. signature_policy and retry_plan have zero current runtime consumers.

---

## Summary Statistics

| Classification | Count | Artifacts |
|---|---|---|
| **owner** | 4 | bundle, health_report, static report, interactive report |
| **public_surface** | 5 | index, stack summary, release summary, landing page, public index |
| **delivery_surface** | 2 | dashboard payload, delivery request |
| **action_result** | 2 | delivery result, publish result |
| **thin_pass_through** | 8 | snippet, prefill, webhook export, draft payload, dry run, publish payload, signature policy, retry plan |

8 out of 21 artifacts (38%) are thin pass-through with at most 1 downstream consumer.

4 artifacts have zero real runtime consumers (future intent only): signature_policy (#15), retry_plan (#17), dry_run (#19), publish_result (#21).

---

## Evidence Sources

- Workflow step names: `rg -n 'name: (Generate|Append|Upload|Post) eval reporting' .github/workflows/evaluation-report.yml`
- Consumer tracing: `rg -l <artifact_name> scripts/ tests/ .github/` for each artifact
- All evidence gathered from current repository state, not from design docs or assumptions
