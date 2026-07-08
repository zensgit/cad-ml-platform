# Phase 0 · A2a — decouple `vision/circuit_breaker` from the live consumer

- **Status**: FOR-REVIEW. Not merged. Zero behavior change.
- **Grounded on** `origin/main @ 8337ea6e`.
- **Why now**: this is the **mandated precondition** (positioning/roadmap design, merged in #499, §60 special case) for Phase 0 **Slice 2**, which prunes the ~106-decorator vision zoo. Without it, pruning `src/core/vision/` breaks the platform's **only live integration**.

---

## 1. The coupling, precisely

`src/core/dedupcad_vision.py` — the HTTP client of the `dedupcad-vision` service, i.e. the single real consumer of this platform — imports its circuit breaker from **inside** `src/core/vision/`:

```python
# src/core/dedupcad_vision.py:18
from src.core.vision.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, FailureType, get_circuit_breaker,
)
```

A naive "delete/rename `vision/`" therefore breaks production. The design doc flagged this; the fix is to extract the breaker to a neutral module first.

## 2. What the file actually is (why a `git mv` would have been wrong)

`src/core/vision/circuit_breaker.py` (546 LOC) is **two things fused together**:

| symbols | nature | depends on `vision/.base`? |
|---|---|---|
| `CircuitState`, `FailureType`, `RecoveryStrategy`, `CircuitBreakerConfig`, `CircuitBreakerStats`, `FailureRecord`, `CallRecord`, `SlidingWindow`, `CircuitBreaker`, `CircuitBreakerError`, `get_circuit_breaker` (11) | **generic** breaker machinery | no |
| `CircuitBreakerVisionProvider`, `create_circuit_breaker_provider` (2) | **vision decorators** — they wrap a `VisionProvider` | **yes** (`from .base import VisionDescription, VisionProvider`) |

The live consumer imports **only generic symbols**. So the correct move is a **split**, not a file move:

- moving the whole file out would drag `vision/.base` into a neutral package (a new cross-dependency), and would relocate two members of the very decorator zoo Slice 2 exists to delete.

## 3. Design

1. **New neutral module** `src/core/resilience/advanced_circuit_breaker.py` — the 11 generic symbols, **verbatim**, with zero vision imports. (`resilience/` is the established home for breakers; the name matches the existing `AdvancedCircuitBreakerConfig` / `AdvancedCircuitState` aliases in `vision/__init__.py`.)
2. **`src/core/vision/circuit_breaker.py` becomes a thin shim**: re-exports the 11 generic names for backward compatibility, and keeps the 2 vision decorators (which still need `.base`). It is now purely zoo material.
3. **Repoint the live consumer** `src/core/dedupcad_vision.py` → the neutral module. Same for `tests/unit/test_dedupcad_vision_client.py`.
4. `vision/__init__.py` and `tests/unit/test_vision_phase7.py` are **untouched** — the back-compat re-export keeps them green, so this PR carries no incidental churn.

### Acceptance criterion (single, checkable)
> `src/core/dedupcad_vision.py` imports **nothing** from `src/core/vision/`.

Verified. See the verification MD.

## 4. Extraction method — and the bug it caught

The first attempt collected only `class`/`def` spans into the new module. That is **lossy**: it silently dropped module-level state living between definitions, notably the registry

```python
_circuit_breakers: Dict[str, CircuitBreaker] = {}
```

(an *annotated* assignment, so it was also invisible to a naive `ast.Assign` scan). The functional smoke caught it as `NameError: name '_circuit_breakers' is not defined` in `get_circuit_breaker`.

The extraction was redone as **remove-only**: take the original file verbatim and delete exactly three things — the module docstring, the `from .base import …` line, and the two vision-symbol spans (decorators included, via `end_lineno`). Everything else, including module-level state, is preserved byte-for-byte. This is why the neutral module is 450 lines rather than a hand-curated subset.

## 5. Risk & rollback
- **Behavior change: none.** Generic code is moved verbatim; all prior import paths still resolve via the shim.
- **Blast radius**: the live consumer's breaker. Covered by a stdlib-only functional smoke (open/reset/registry) plus the existing test suite in CI.
- **Rollback**: single `git revert`. No flags, no schema, no CI/protection change.

## 6. Follow-ups (cross-slice — important)
- **Slice 2** may now prune `src/core/vision/` freely. After it deletes the shim, **#500's `prune_safety_check.py` must move `src/core/vision/circuit_breaker.py` from `LIVE_TWINS` → `PRUNED_MODULES`** — post-A2a that path holds only zoo decorators, so its "live twin" status expires with Slice 2. Leaving it in `LIVE_TWINS` would make Slice 2 fail the gate; removing it prematurely would disarm the guard. It must flip **in the same PR that deletes it**.
- Consider, later, whether `resilience/advanced_circuit_breaker.py` should absorb or be unified with the other breaker implementations (`utils/`, `resilience/`, `gateway/`) — that is a duplication question, out of scope here.
