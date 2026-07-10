# Phase 0 · A2a verification — decouple `vision/circuit_breaker`

Companion to `PHASE0_A2A_DECOUPLE_VISION_CIRCUIT_BREAKER_DESIGN_20260708.md`. Grounded on `origin/main @ 8337ea6e`.

---

## 1. Acceptance criterion — met

> `src/core/dedupcad_vision.py` must import **nothing** from `src/core/vision/`.

```
$ grep -nE '^\s*(from|import)\s+.*core\.vision' src/core/dedupcad_vision.py
  (no output)
✅ zero core.vision imports — vision/ is now prunable
```

Before this change the same probe returned `src/core/dedupcad_vision.py:18: from src.core.vision.circuit_breaker import (`.

## 2. Neutral module is genuinely vision-free

```
$ grep -nE '^\s*(from|import)\s+.*(vision|\.base)' src/core/resilience/advanced_circuit_breaker.py
  (no output)   ✅
```
(The word "vision" appears only in the module docstring, explaining the extraction.)

## 3. Functional smoke — the extracted core behaves identically

Run against the **real** API (`can_execute` / `record_failure` / `record_success` / `reset`; there is no `call()`), stdlib-only so it executes on local Python 3.9:

```
✅ all 5 symbols dedupcad_vision.py imports are present
✅ opens at threshold=2 → state=open, can_execute=False
✅ reset() → state=closed, can_execute=True
✅ get_circuit_breaker registry works and is identity-stable: True

FUNCTIONAL SMOKE: PASS (extracted core behaves identically, stdlib-only)
```

## 4. Two bugs the smoke caught (and why compilation was not enough)

**Bug 1 — lossy extraction (real).** The first split collected only `class`/`def` spans, silently dropping module-level state:

```
NameError: name '_circuit_breakers' is not defined
  File "src/core/resilience/advanced_circuit_breaker.py", line 443, in get_circuit_breaker
```

`_circuit_breakers: Dict[str, CircuitBreaker] = {}` is an **annotated** assignment, so it was invisible both to the span collector and to a naive `ast.Assign` scan. The file **compiled cleanly** with this bug present — only executing it exposed it. Fixed by re-doing the split as *remove-only* (delete the docstring, the `.base` import, and the two vision-symbol spans; keep everything else verbatim). Now preserved at line 432.

**Bug 2 — a bad probe, not a bad code path.** The first smoke reported "breaker did not open after 3 failures". Cause: it called a non-existent `cb.call(...)`, and its `except Exception: pass` **swallowed the resulting `AttributeError`**, so no failure was ever recorded. The production code was never wrong. Re-probed against the real API → opens exactly at `failure_threshold`.

Both are recorded because they are the reason this PR is trustworthy: a compile-only check would have shipped Bug 1.

## 4b. CI caught a third bug my local checks missed (RED → fix → GREEN)

The first push of this PR went **RED on 4 checks** (`core-fast-gate`, `test-new-modules`, `Platform Self-Check`, `training-governance`). Root cause, single and mine:

```
src/core/vision/circuit_breaker.py:53: NameError: name 'Optional' is not defined
WARNING src.api:__init__.py:168 router_import_failed
```

When I rewrote `vision/circuit_breaker.py` as a shim I kept the two decorator bodies but **dropped the original stdlib import header** they depend on (`time`, `typing.{Callable,Dict,Optional}`). The other three failures were cascades of the same import error.

`py_compile` passed, and the AST symbol check passed, because a `NameError` in a class-body annotation is a **runtime** failure. This is the second time in this slice that compilation was insufficient — and this time I had already written that lesson into §4 and still shipped it.

**Fix**: restore the precise imports the two decorators need (computed by free-variable analysis, not blanket-copied):
```python
import time
from typing import Callable, Dict, Optional
```

**New guard added to this slice's verification** — a *runtime import smoke* of the shim, executed locally with a stubbed `.base` so no heavy deps are needed. It executes the module body (evaluating class bodies), which is exactly what surfaces this bug class:

```
✅ shim imports cleanly (class bodies evaluated)
✅ all decorators + re-exported generics present on the shim
```

Free-variable analysis after the fix reports only `e` (an `except … as e` handler variable — a known false positive of the crude scope model).

## 5. Back-compat — no incidental churn

The shim must keep every existing importer green. Verified by AST, not by eye:

| importer | symbols needed | result |
|---|---|---|
| `src/core/vision/__init__.py` | `CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitBreakerStats, CircuitBreakerVisionProvider, CircuitState, FailureType, RecoveryStrategy, SlidingWindow, create_circuit_breaker_provider, get_circuit_breaker` (11) | ✅ shim provides all |
| `tests/unit/test_vision_phase7.py` | `CallRecord, CircuitBreaker, CircuitBreakerConfig, CircuitBreakerStats, CircuitBreakerVisionProvider, CircuitState, FailureType, RecoveryStrategy, SlidingWindow` | ✅ all resolvable |

Neither file is modified by this PR.

## 6. Compilation
```
python3 -m py_compile \
  src/core/resilience/advanced_circuit_breaker.py \
  src/core/vision/circuit_breaker.py \
  src/core/vision/__init__.py \
  src/core/dedupcad_vision.py \
  tests/unit/test_dedupcad_vision_client.py
✅ all compile
```

## 7. What CI verifies that local cannot
Local Python is 3.9 and the project targets 3.11, so the full suite (and the app import smoke via `tests/test_routes_smoke.py`) runs only in CI. The functional smoke above is stdlib-only and therefore *does* run locally — it exercises the actual extracted behavior, not just its syntax. Stated plainly rather than glossed.

## 8. Residual risk
- `tests/unit/test_vision_phase7.py` still imports through the shim. That is intentional (zero churn), but it means Slice 2's deletion of the shim must update that test.
- **Cross-slice invariant**: after Slice 2 deletes the shim, #500's `prune_safety_check.py` must move `src/core/vision/circuit_breaker.py` from `LIVE_TWINS` → `PRUNED_MODULES`, **in the same PR that deletes it**. Too early disarms the guard; too late fails the gate.
