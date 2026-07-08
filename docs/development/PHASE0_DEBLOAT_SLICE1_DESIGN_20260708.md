# Phase 0 de-bloat — Slice 1 design (evidence-gated deletion + prune-safety gate)

- **Status**: FOR-REVIEW. Authorized by the ratified positioning/roadmap design (`docs/PLATFORM_POSITIONING_AND_ROADMAP_DESIGN_20260706.md`, merged in #499), §5: *"Phase 0 删除在 owner ratify 本文之后,作为独立 for-review PR"*.
- **Grounded on** `origin/main @ 8337ea6e` (every path below re-verified at this SHA, not carried from the design doc).
- **Not merged by this PR.** Nothing lands without owner "合".

---

## 1. Why this is Slice 1, not "all of Phase 0"

The design doc's Phase 0 lists three deletion groups. Re-verifying each against code at `8337ea6e` **contradicted the doc on two of them**, so this slice ships only what the evidence actually supports:

| Group | Doc's premise | Verified reality | Action |
|---|---|---|---|
| 13 dead `src/core/<dir>` scaffolds | zero-reference | **confirmed**: 0 external importers, 0 dynamic/lazy refs, 0 tests reference them | **delete (this PR)** |
| `src/api/v2`, `src/api/grpc` | "unmounted" | `v2`: 0 real importers (the 1 hit is a *comment* at `src/core/vision/api_versioning.py:330`). `grpc`: 24 importers — **all** from its own `tests/unit/test_grpc_server.py`. Neither mounted in `src/api/__init__.py` / `src/main.py` | **delete (this PR)**, incl. the grpc-only test |
| `vision/` decorator zoo (~90%) | delete/downgrade | **82,249 LOC / 112 files / 106 `*VisionProvider` classes**; `experimental/` alone is 10,234 LOC; and `vision/circuit_breaker.py` is **live-imported** by the sole real consumer `src/core/dedupcad_vision.py:18` (+ tests) | **DEFERRED → Slice 2** |

Two deferrals, both evidence-driven:

- **vision zoo → Slice 2.** A ~75K-line deletion cannot be per-path evidence-verified in one pass, nor reviewed. Its mandated precondition (§60 special case) is to first **extract `vision/circuit_breaker` to a neutral module and repoint `dedupcad_vision.py`** — otherwise the deletion breaks the only real integration. Slice 2 = decouple, then prune.
- **`AuditMiddleware` → separate decision.** It has **two definitions** (`src/core/audit/logger.py:574`, `src/api/middleware/audit.py:20`) and 0 mount sites. That is a *duplication / design* question about the audit domain, not dead scaffolding — deleting it silently would be a judgement call disguised as hygiene.

## 2. What this PR deletes (8,973 lines, 20 files)

13 dead top-level scaffolds — **full paths, never bare names**:
`src/core/{circuit_breaker, dead_letter_queue, outbox, message_bus, idempotency, api_versioning, rate_limiter, webhook, caching, batch_processing, event_sourcing, health_check, notifications}` (7,642 LOC)

Unmounted surfaces: `src/api/v2` (243) · `src/api/grpc` (297) + `tests/unit/test_grpc_server.py` (its sole importer) · `src/api/v1/batch.py` (281) · `src/api/v1/websocket.py` (259) — none registered in `src/api/__init__.py`.

### Explicitly NOT deleted — the same-named LIVE twins
The bare-name trap: several pruned dirs share a name with a live module. These stay, and the new gate asserts they stay:
`src/utils/{circuit_breaker,idempotency,rate_limiter}.py` · `src/core/assistant/caching.py` · `src/core/{resilience,resilience_enhanced,gateway}/circuit_breaker.py` · `src/core/{resilience,gateway}/rate_limiter.py` · `src/core/vision/circuit_breaker.py`

## 3. The linchpin: a prune-safety gate that can actually fail

The design doc's central insight is that this bloat was **created by the agent fleet**, and the repo's own dead-code/duplicate-code checks did not stop it because every `code-quality.yml` step ends in `|| true`. Deleting without a gate is Sisyphean.

So this PR adds `scripts/ci/prune_safety_check.py` + `.github/workflows/prune-safety.yml`, enforcing two hard invariants (**no `|| true`**):

1. **No resurrection** — none of the 17 pruned module paths may be imported again.
2. **No mis-delete** — all 10 live twins must still exist.

Design choices:
- **Pure-static** (no dependency install) so the gate can never false-red on an environment problem. The runtime import smoke is already covered — `tests/test_routes_smoke.py` imports the app, so a broken import reds CI on its own.
- **Not yet a required check.** Making it *required* is a branch-protection change and belongs to the owner, per the design lock. Until then it runs and reports; flipping it to required is a one-line owner action.

## 4. Risk & rollback
- **Behavior change: none.** Every deleted path has 0 production importers and 0 dynamic refs; nothing was mounted.
- **Blast radius on mis-delete**: caught three ways — the new gate (invariant 2), the existing suite's app import, and this PR's per-path evidence (verification MD).
- **Rollback**: single `git revert` of one squash commit; no schema, no flags, no migration, no protection change.

## 5. Follow-ups
- **Slice 2** — decouple `vision/circuit_breaker` → neutral module + repoint `dedupcad_vision.py`, then prune the 106 decorators + `experimental/`.
- **Owner action** — make `Prune Safety` a required check, and strip `|| true` from `code-quality.yml`'s dead-code (`vulture`, :165) and duplicate-code (:123) steps so the anti-bloat gates bite. Add CODEOWNERS (design doc §3 轨A).
- **`AuditMiddleware`** — decide: unify the two definitions, or delete.
