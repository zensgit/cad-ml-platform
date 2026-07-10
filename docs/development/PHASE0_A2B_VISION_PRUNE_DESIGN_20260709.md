# Phase 0 · A2b — Vision package prune (DESIGN)

**Date:** 2026-07-09 · **Slice:** Phase 0 / A2b · **Status:** for-review
**Depends on:** #501 (A2a — `vision/circuit_breaker` decoupled from the live consumer; generic
core now lives at `src/core/resilience/advanced_circuit_breaker.py`). Merged to main.
**Supersedes premise of:** `PHASE0_A2B_VISION_PRUNE_VERIFICATION_20260708.md §8` (execution finding).

## 1. Problem

`src/core/vision/` shipped as a 112-file, ~100k-line package: a small live core wrapped in a
large "observability/enterprise zoo" (metrics dashboards, alert managers, APM, SLA monitors,
chaos engineering, multi-region, an `experimental/` AutoML/feature-store subtree, …). The zoo is
**scaffolding**: it is imported only by per-phase unit tests (`test_vision_phase3..23`), never by a
live route or the provider factory. It is dead weight on every reader, grep, and CI run.

But `vision/` is **NOT dead code** and must not be bulk-deleted — the earlier roadmap premise
("vision is dead") was FALSE (corrected 2026-07-08). It is a **live, API-registered package**:

- `src/api/v1/vision.py` is a registered router (`src/api/__init__.py:245`) and imports
  `ResilientVisionProvider, VisionAnalyzeRequest, VisionAnalyzeResponse, VisionInputError,
  VisionManager, VisionProviderError, create_vision_provider, get_available_providers` from
  `src.core.vision`.
- `src/core/providers/vision.py` (production adapter) imports `base`, `factory`,
  `providers.deepseek_stub`.
- `factory.py` imports the **8 concrete providers** via `from .providers import (...)`.

## 2. Why the previous delete-set derivation was unsafe (corrected)

The 2026-07-08 attempt derived the delete-set by **decorator-reachability** (which `*VisionProvider`
subclasses look reachable). Executing it against a runtime import-smoke exposed two defects:

1. **Module cross-references** — `analytics.py` (would-keep) imports `ResultPersistence` from
   `persistence.py` (would-delete) → `ModuleNotFoundError`. Class-level reachability ≠ module-level
   safety.
2. **Package imports** — a naive closure dropped the 7 non-stub concrete providers because they load
   via `from .providers import (...)`, i.e. through the `providers` **package** `__init__`, which
   eagerly imports all 8 submodules.

**Corrected method: package-aware transitive-import closure** from the live entry points, verified by
actually running the import — not by compiling or class-graph reasoning.

## 3. Method (reproducible)

Keep-set = transitive closure of intra-package imports, seeded from the production entry points and
resolved through re-exports:

- Seed names imported via `from src.core.vision import …` (resolved through `__init__`'s export map):
  `ResilientVisionProvider→resilience`, the `Vision*` request/response/error/manager types→`base`/
  `manager`, `create_vision_provider`/`get_available_providers`→`factory`, plus `VisionDescription`,
  `VisionProvider`.
- Seed direct submodule imports: `base`, `factory`, `manager`, the `providers` **package** (which
  pulls all 8 provider submodules through its `__init__`), `providers.deepseek_stub`.
- BFS the import graph. **Package targets are expanded, not short-circuited** (the bug that dropped
  the 7 providers): a module is marked kept only when popped and its own imports are expanded.

**KEEP (14 files):** `__init__`, `base`, `factory`, `manager`, `resilience`,
`providers/{__init__, anthropic, deepseek, deepseek_stub, doubao, glm4v, openai, qwen_vl, vllm_vision}`.

**DELETE (98 files):** every other module + the `experimental/` subtree. Full manifest in the
verification MD.

The closure is **import-closed**: no kept module imports a deleted module (verified empirically), and
there are **zero dynamic/lazy imports** (`importlib`/`__import__`/`import_module`) in the kept set, so
the static closure is sound.

## 4. `__init__.py` surgery

`__init__.py` re-exported the whole zoo (465 relative `from .X import …` statements, 3705 lines). AST
surgery (using `end_lineno` line-ranges) removes every `ImportFrom` whose target resolves to a deleted
module (460 statements, 1756 names) and prunes the matching `__all__` entries → **117 lines, 28
exports**, all backed by kept modules.

## 5. `circuit_breaker` moves LIVE_TWINS → PRUNED_MODULES (same PR, mandated)

`src/core/vision/circuit_breaker.py` was a #501 shim consumed only by `__init__` (and a now-deleted
phase test). It is in the delete-set. `scripts/ci/prune_safety_check.py`:
- **remove** `src/core/vision/circuit_breaker.py` from `LIVE_TWINS` (else the no-mis-delete invariant
  reds on its deletion);
- **add** `core.vision.circuit_breaker` (and the other 97 pruned modules + `core.vision.experimental`)
  to `PRUNED_MODULES` — the gate's stated purpose is "stop the agent fleet from re-inflating the
  scaffolding it deleted", and an autonomous cadence loop now runs against this repo, so the guard is
  load-bearing, not cosmetic. Its live generic twin remains `resilience/advanced_circuit_breaker.py`.

## 6. Orphaned phase tests

21 `tests/unit/test_vision_phase*.py` / `test_vision_{advanced,extended,persistence}.py` import only
delete-set modules → deleted with the scaffolding. 13 also touch `base` incidentally; `base`'s real
contract coverage is retained by `tests/test_contract_schema.py` + `tests/test_metrics_consistency.py`
(both kept, both still collect).

## 7. Non-goals

No behavior change to the live vision route, factory, or any provider. No new abstraction. This is a
delete-only slice defended by a runtime import-smoke + the prune-safety gate.
