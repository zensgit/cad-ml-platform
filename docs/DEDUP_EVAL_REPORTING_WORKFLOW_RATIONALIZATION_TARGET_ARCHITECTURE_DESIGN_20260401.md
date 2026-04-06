# Eval Reporting Workflow Rationalization Target Architecture — Design

日期：2026-04-01

## Scope

Batch 16B: based on the Batch 16A inventory, produce keep/merge/remove lists, a simplified target workflow structure, and a migration order. No code changes.

---

## 1. Keep List (13 artifacts)

These artifacts have real runtime consumers, meaningful independent value, or are structural owners.

| # | Artifact | Reason |
|---|---|---|
| 1 | `eval_reporting_bundle.json` | Owner: root materialization target, consumed by health/index/helpers |
| 2 | `eval_reporting_bundle_health_report.json` | Owner: health/freshness/pointer guard, consumed by index/summary/landing |
| 3 | `eval_reporting_index.json` | Public surface: discovery root, consumed by summary/landing/public_index/PR comment/notify |
| 4 | `eval_reporting_stack_summary.json` | Public surface: workflow-friendly status, consumed by landing/notify/release_summary/PR comment |
| 5 | `eval_reporting_release_summary.json` | Public surface: release readiness signal, consumed by dashboard_payload |
| 6 | `index.html` (landing page) | Public surface: human entry point, Pages root |
| 7 | `report_static/index.html` | Owner: static HTML report |
| 8 | `report_interactive/index.html` | Owner: interactive HTML report |
| 9 | `eval_reporting_public_index.json` | Public surface: public URL discovery, consumed by dashboard_payload |
| 10 | `eval_reporting_dashboard_payload.json` | Delivery surface: canonical external payload (after merge: direct input to publish_result and delivery_request) |
| 11 | `eval_reporting_webhook_delivery_request.json` | Delivery surface: canonical delivery request with policy, consumed by sender |
| 12 | `eval_reporting_webhook_delivery_result.json` | Action result: actual delivery outcome with retry fields |
| 13 | `eval_reporting_release_draft_publish_result.json` | Action result: gated release draft publish outcome (after merge: absorbs publish_payload and reads dashboard_payload directly) |

---

## 2. Merge Candidates (5 artifacts → absorbed into kept artifacts)

| Artifact to Remove | Absorb Into | Rationale |
|---|---|---|
| `eval_reporting_release_note_snippet.json` (#11) | `dashboard_payload` → `publish_result` | Single-field copy between dashboard_payload and prefill; snippet_markdown can be generated directly by publish_result consumer |
| `eval_reporting_release_draft_prefill.json` (#12) | `dashboard_payload` → `publish_result` | draft_title/draft_body can be derived directly from dashboard_payload by the publish consumer |
| `eval_reporting_webhook_export.json` (#13) | `dashboard_payload` → `delivery_request` | Near-identical to dashboard_payload; delivery_request can consume dashboard_payload directly |
| `eval_reporting_release_draft_payload.json` (#18) | `dashboard_payload` → `publish_result` | Intermediate between prefill and publish_payload; publish consumer can derive from dashboard_payload directly |
| `eval_reporting_release_draft_publish_payload.json` (#20) | `publish_result` | Policy fields can be inlined into publish_result consumer |

**Net effect:** Release chain collapses from `dashboard_payload → snippet → prefill → draft_payload → publish_payload → publish_result` to `dashboard_payload → publish_result`. Webhook chain collapses from `dashboard_payload → webhook_export → delivery_request` to `dashboard_payload → delivery_request`.

---

## 3. Remove Candidates (3 artifacts)

| Artifact | Reason |
|---|---|
| `eval_reporting_webhook_signature_policy.json` (#15) | Zero runtime consumers; declares future intent only; can be re-introduced when a signer is implemented |
| `eval_reporting_webhook_retry_plan.json` (#17) | Zero runtime consumers; declares future intent only; can be re-introduced when a retry queue is implemented |
| `eval_reporting_release_draft_dry_run.json` (#19) | Zero runtime consumers; publish_result already covers the same gating logic; redundant side branch |

---

## 4. Move-Out-of-deploy-pages Candidates

| Artifact | Current Location | Recommended Location |
|---|---|---|
| `eval_reporting_release_summary.json` (#5) | evaluate job | Keep in evaluate job (already there) |
| `eval_reporting_dashboard_payload.json` (#10) | deploy-pages job | Consider moving to evaluate job if public_index dependency can be resolved |

**Note:** Most deploy-pages artifacts depend on `page_url` from the deployment step, so they must stay in deploy-pages. However, after merges, the deploy-pages job would shrink significantly.

---

## 5. Target Workflow Shape

### evaluate job (unchanged)

```
refresh_eval_reporting_stack
  → bundle → health → index → landing page
  → stack summary → release summary → status check
  → assemble pages root
  → upload artifacts (static, interactive, stack, landing, pages-ready)
  → fail step
```

### deploy-pages job (simplified)

**Current:** 13 deploy-pages-side artifacts × 3 steps (generate + append + upload) = **39 content steps** + ~6 infrastructure steps (download, checkout, setup, upload-to-pages, deploy, download-release-summary) = **~45 total**

**Target:** 5 kept deploy-pages-side artifacts × 3 steps = **15 content steps** + ~6 infrastructure steps = **~21 total**

```
download pages-ready artifact → setup → upload to Pages → deploy

download artifacts for post-deploy surfaces
checkout (sparse)

public_index (3 steps: generate + append + upload)
dashboard_payload (3 steps)

delivery_request (3 steps — absorbs webhook_export)
delivery_result (3 steps)

publish_result (3 steps — absorbs snippet, prefill, draft_payload, publish_payload, dry_run)
```

**Content step reduction:** 39 → 15 (24 steps removed = 8 merged/removed artifacts × 3)
**Total step reduction:** ~45 → ~21

---

## 6. Migration Order

| Phase | Action | Risk |
|---|---|---|
| 1 | Remove signature_policy, retry_plan, dry_run (zero consumers) | Low — no downstream breakage |
| 2 | Merge webhook_export into delivery_request (delivery_request reads dashboard_payload directly) | Low — only delivery_request.py changes input |
| 3 | Merge snippet + prefill + draft_payload + publish_payload into publish_result (publish_result reads dashboard_payload directly) | Medium — publish_result.js needs to derive draft_title/body internally |
| 4 | Consolidate deploy-pages job summary and upload steps | Low — workflow YAML cleanup only |
| 5 | Verify end-to-end: all kept artifacts still materialize, all tests pass | Required gate |

---

## 7. Risk / Rollback Notes

- **Phase 1 (remove):** Safe rollback — re-add the removed scripts and workflow steps. No downstream consumer exists.
- **Phase 2 (merge webhook):** Rollback by reverting delivery_request.py to read webhook_export again. webhook_export.py can be restored.
- **Phase 3 (merge release):** Higher risk — publish_result.js changes are more substantial. Recommend keeping old scripts in repo (unused) for 1 release cycle before deleting.
- **Phase 4 (consolidate):** Pure workflow YAML — easy to revert.
- **General:** Each phase should be a separate batch with its own validation, not a single big-bang.

---

## Statement

本批无代码改动。以上为目标架构和迁移计划，待人工验收后可作为后续 rationalization batch 的执行依据。
