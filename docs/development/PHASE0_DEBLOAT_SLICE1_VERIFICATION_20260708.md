# Phase 0 de-bloat — Slice 1 verification (per-path evidence + observed-RED)

Companion to `PHASE0_DEBLOAT_SLICE1_DESIGN_20260708.md`. Grounded on `origin/main @ 8337ea6e`.
The design doc (§60/§61) requires this PR to carry **per-path "zero external import" evidence** and to pass a **blocking prune-safety job, not a manual grep**. Both are below.

---

## 0. Evidence-integrity note (why the first pass was thrown away)

The first evidence run used an **unquoted** `--include=*.py`, which zsh glob-expanded before `grep` ever saw it. The greps silently never ran, and the run reported `importers=0` for **every live twin** — including `src/utils/idempotency.py`, which is demonstrably imported by `src/api/v1/ocr.py` and `drawing.py`.

Deleting on that output would have destroyed live code while displaying a clean bill of health. **All evidence below was re-gathered with quoted globs and a positive control.**

### Positive control (proves the grep actually fires)
Live twins must show importers **> 0**. They do:

| live module | importers |
|---|---|
| `src/utils/idempotency.py` | 24 |
| `src/utils/circuit_breaker.py` | 28 |
| `src/utils/rate_limiter.py` | 20 |
| `src/core/assistant/caching.py` | 4 |
| `src/core/resilience/circuit_breaker.py` | 45 |
| `src/core/resilience_enhanced/circuit_breaker.py` | 1 |
| `src/core/gateway/circuit_breaker.py` | 4 |
| `src/core/resilience/rate_limiter.py` | 76 |
| `src/core/gateway/rate_limiter.py` | 2 |

A zero here would mean the probe is broken. None is zero.

## 1. Per-path deletion evidence

Against that *same* working probe, every deleted path is unreferenced:

| deleted path | LOC | external importers | dynamic/lazy refs |
|---|---:|---:|---:|
| `src/core/circuit_breaker` | 501 | 0 | 0 |
| `src/core/dead_letter_queue` | 644 | 0 | 0 |
| `src/core/outbox` | 555 | 0 | 0 |
| `src/core/message_bus` | 521 | 0 | 0 |
| `src/core/idempotency` | 557 | 0 | 0 |
| `src/core/api_versioning` | 626 | 0 | 0 |
| `src/core/rate_limiter` | 431 | 0 | 0 |
| `src/core/webhook` | 554 | 0 | 0 |
| `src/core/caching` | 606 | 0 | 0 |
| `src/core/batch_processing` | 591 | 0 | 0 |
| `src/core/event_sourcing` | 716 | 0 | 0 |
| `src/core/health_check` | 628 | 0 | 0 |
| `src/core/notifications` | 712 | 0 | 0 |
| `src/api/v2` | 243 | 0 (the 1 hit is a **comment**, `src/core/vision/api_versioning.py:330`) | 0 |
| `src/api/grpc` | 297 | 24 — **all** in `tests/unit/test_grpc_server.py` (deleted with it); 0 production | 0 |
| `src/api/v1/batch.py` | 281 | 0 (unregistered in `src/api/__init__.py`) | 0 |
| `src/api/v1/websocket.py` | 259 | 0 (unregistered) | 0 |

- **Dynamic-import sweep**: `importlib` / `__import__` / string-literal module refs for all 13 names → **0 hits**.
- **Orphan tests**: no test file references any of the 13 dirs → nothing left dangling.
- **Mount check**: `AuditMiddleware` → 0 `add_middleware` sites; `grpc`/`v2` absent from `src/api/__init__.py` and `src/main.py`.

## 2. Post-delete verification (run after `git rm`)

- **Zero lingering references** to any of the 17 pruned module paths across `src/ tests/ scripts/`. ✅
- **All 10 live twins present.** ✅
- `src/**/*.py` file count: **818 → 799**. Diff: **20 files, 8,973 deletions, 0 insertions** to product code.

## 3. Observed-RED — the gate genuinely fails

A gate that only ever passes proves nothing. Both invariants were driven red on purpose, then reverted:

**RED 1 — resurrection** (simulating the fleet re-inflating scaffolding): added `from src.core.outbox import Outbox`.
```
::error::pruned module 'core.outbox' is imported again:
    src/_prune_probe.py:1: from src.core.outbox import Outbox
prune-safety: FAIL      exit=1
```

**RED 2 — mis-delete of a live twin** (the exact bare-name trap the design doc's P1 was about): removed `src/utils/idempotency.py`.
```
::error::live module 'src/utils/idempotency.py' is missing — it shares a name with a
pruned scaffold but is NOT dead code. This is a mis-delete.
prune-safety: FAIL      exit=1
```

**GREEN after reverting both probes:**
```
prune-safety: OK (17 pruned modules unreferenced, 10 live twins intact, 2040 files scanned)
exit=0
```

Both probes were reverted; `git status` shows no stray files.

## 4. What CI verifies that local cannot
Local Python is 3.9; the project targets 3.11, so the app import smoke is **not** run locally — it is covered in CI by the existing suite (`tests/test_routes_smoke.py` imports the app). If any deletion broke a real import, CI reds. This is stated rather than glossed: the local run proves the *static* invariants only.

## 5. Residual risk
- The prune-safety job is **not yet a required check** (making it required is a branch-protection change, owner-only). Until then it reports but cannot block a merge.
- `code-quality.yml`'s dead-code/duplicate-code steps still end in `|| true`, so they remain observe-only. Stripping those is the owner's linchpin action (design doc §3 轨A).
- Slice 2 (vision zoo, 82K LOC) is untouched and still gated on decoupling `vision/circuit_breaker` from `src/core/dedupcad_vision.py:18`.
