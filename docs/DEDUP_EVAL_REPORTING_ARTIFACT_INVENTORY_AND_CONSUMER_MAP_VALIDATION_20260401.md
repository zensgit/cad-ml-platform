# Eval Reporting Artifact Inventory and Consumer Map — Validation

日期：2026-04-01

## Validation Checklist

| # | Criterion | Status |
|---|---|---|
| 1 | Inventory covers all eval reporting artifacts in main workflow | PASS (21 artifacts) |
| 2 | Every artifact has explicit consumer or explicit "no independent consumer" | PASS |
| 3 | Classification uses controlled vocabulary (owner/public_surface/action_result/delivery_surface/thin_pass_through) | PASS |
| 4 | recommended_action uses controlled vocabulary (keep/merge/remove/defer_decision) | PASS |
| 5 | High-duplication chains explicitly identified | PASS (release chain: 4 intermediate layers; webhook chain: 2 intermediate layers + 3 terminal branches) |
| 6 | Evidence gathered from real code, not assumptions | PASS (rg commands on current repo) |
| 7 | No code changes made | PASS |

## Method

1. Listed all eval reporting workflow steps via `rg -n 'name: (Generate|Append|Upload|Post) eval reporting'`
2. For each artifact, traced consumer files via `rg -l <artifact_name> scripts/ tests/ .github/`
3. Distinguished test-only references from real runtime consumers
4. Classified each artifact by its role in the real dependency chain
5. Identified chain duplication by tracing input→output lineage

## Key Findings

1. **38% of artifacts (8/21) are thin pass-through** with at most 1 downstream consumer
2. **4 artifacts have zero real runtime consumers** (signature_policy, retry_plan, dry_run, publish_result) — they declare future intent only or have no downstream consumer beyond their own test
3. **The release chain has 4 intermediate layers** (snippet, prefill, draft_payload, publish_payload) between dashboard_payload and publish_result; dry_run is a zero-consumer side branch
4. **webhook_export is nearly identical to dashboard_payload** and serves only as input to delivery_request
5. **The deploy-pages job now has 39+ always-run steps** for eval reporting surfaces alone

## Evidence Coverage

All 21 artifacts were traced via `rg -l` against the current repository (`scripts/ci/`, `scripts/*.py`, `.github/workflows/evaluation-report.yml`). The inventory is accounted for from direct code-level evidence in the current repo.

## Commands Run

```bash
# Workflow step listing
rg -n 'name: (Generate|Append|Upload|Post) eval reporting' .github/workflows/evaluation-report.yml

# Consumer tracing for 15 deploy-pages-side artifacts
for artifact in eval_reporting_stack_summary eval_reporting_release_summary eval_reporting_public_index eval_reporting_dashboard_payload eval_reporting_release_note_snippet eval_reporting_release_draft_prefill eval_reporting_webhook_export eval_reporting_webhook_delivery_request eval_reporting_webhook_signature_policy eval_reporting_webhook_delivery_result eval_reporting_webhook_retry_plan eval_reporting_release_draft_payload eval_reporting_release_draft_dry_run eval_reporting_release_draft_publish_payload eval_reporting_release_draft_publish_result; do
  rg -l "$artifact" scripts/ci/ scripts/*.py tests/ .github/workflows/
done

# Consumer tracing for 6 evaluate-side artifacts (direct repo evidence)
for artifact in eval_reporting_bundle eval_reporting_bundle_health_report eval_reporting_index report_static report_interactive generate_eval_reporting_landing_page; do
  rg -l "$artifact" scripts/ci/ scripts/*.py .github/workflows/evaluation-report.yml
done
```
