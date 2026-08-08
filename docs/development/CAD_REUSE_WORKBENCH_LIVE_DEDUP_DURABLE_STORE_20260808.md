# Live dedup wiring + durable store (post-MVP)

**Date**: 2026-08-08  
**Branch**: `eng/workbench-live-dedup-durable-store-20260808`  
**Boundaries**: R2 HOLD; decision default-off; no eval_integrity / cost_cap / training JSONL

## What shipped

### Live dedup (default-off)

| Env | Default | Behavior |
|---|---|---|
| `REVIEW_REUSE_LIVE_DEDUP` | off | offline `insufficient_evidence` / `tool_unavailable` |
| `REVIEW_REUSE_LIVE_DEDUP=true` | — | call private `DedupCadVisionClient.search_2d` (or injected hook) |

- Mapper: vision `duplicates`/`similar` → `CandidateDecision` (`dedup_live.vision_response_to_hits`)
- Failure → `external_service_unavailable` or empty → `tool_unavailable`
- Tests inject hooks; no hosted LLM

### Durable store

| Env | Default | Behavior |
|---|---|---|
| `REVIEW_REUSE_STORE` | `memory` | process-local |
| `REVIEW_REUSE_STORE=filesystem` | — | JSON under `REVIEW_REUSE_STORE_DIR` (default `data/review_reuse_tasks`) |

Restart-safe single-node pilot; tenant isolation via path segment + task payload.

## Not in this PR

- Redis multi-node store
- JWT reviewer identity
- Decision default-on
- PLM write-back
