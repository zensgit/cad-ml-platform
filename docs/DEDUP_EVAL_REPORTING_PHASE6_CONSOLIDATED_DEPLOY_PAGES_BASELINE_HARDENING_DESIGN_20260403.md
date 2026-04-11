# Eval Reporting Phase 6 Consolidated Deploy-Pages Baseline Hardening — Design

日期：2026-04-03

## Scope

Batch 22B: establish regression guards for the post-consolidation `deploy-pages` workflow baseline. No workflow YAML changes — tests only.

## Post-Consolidation Baseline

After Batch 22A (consolidation + ordering fix), the `deploy-pages` job step ordering is:

1. Download Pages-ready artifact
2. Setup Pages
3. Upload to Pages
4. Deploy to GitHub Pages
5. Checkout for public index generation
6. Download stack summary for public index
7. Download stack summary JSON for public index
8. Generate public discovery index
9. Download release summary for dashboard payload
10. Generate eval reporting dashboard payload
11. Generate eval reporting webhook delivery request
12. Generate eval reporting webhook delivery result
13. Generate eval reporting release draft publish result
14. Consolidated eval reporting deploy-pages summary
15. Upload public discovery index
16. Upload eval reporting dashboard payload
17. Upload eval reporting webhook delivery request
18. Upload eval reporting webhook delivery result
19. Upload eval reporting release draft publish result

## Hardening Guards Added

### 1. Negative guard

Old 5 per-surface summary append steps must not reappear:

- `Append public URLs to job summary`
- `Append eval reporting dashboard payload to job summary`
- `Append eval reporting webhook delivery request to job summary`
- `Append eval reporting webhook delivery result to job summary`
- `Append eval reporting release draft publish result to job summary`

### 2. Fixed-order guards

- Generate block order: public_index → dashboard_payload → delivery_request → delivery_result → publish_result
- Upload block order: public_index → dashboard_payload → delivery_request → delivery_result → publish_result

### 3. Pre-existing guards (from Batch 22A fix)

- Consolidated summary after last generate
- Consolidated summary before first upload
- Upload block contiguous

## What Was NOT Changed

- No workflow YAML changes
- No artifact contract changes
- No helper / consumer logic changes
- No generate / upload step content changes
