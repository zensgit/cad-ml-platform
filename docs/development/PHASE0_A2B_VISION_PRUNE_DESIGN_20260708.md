# Phase 0 · A2b — vision decorator-zoo prune (DESIGN, evidence-gated)

- **Status**: DESIGN / for-review groundwork. **No code deleted here.** Not a PR yet (review-backlog cap: 4 already open). Becomes the A2b PR once **#501 merges** and the backlog clears.
- **Grounded on** `origin/main @ 8337ea6e`, verified by running the code + a 9-agent adversarial triage.
- **Authorized by** the ratified roadmap (#499), Phase 0. **But it overturns that roadmap's premise** — see §1.

---

## 1. The roadmap premise was wrong: `vision/` is NOT dead scaffolding

The #499 roadmap called `vision/` an "82K decorator zoo, ~90% deletable." Running the code shows the *package* is **live and API-registered**, so a bulk delete would break a shipped route:

- `src/api/v1/vision.py` is a **registered router** (`src/api/__init__.py:245`) — a live HTTP surface. It imports `VisionManager`, `ResilientVisionProvider`, `create_vision_provider`, `get_available_providers` from `src.core.vision`.
- `src/core/providers/vision.py` imports `base` / `factory` / `deepseek_stub` in production.

**Reachability model (verified):** a vision class is live **iff** it is in `factory.PROVIDER_REGISTRY` (`:34`) / `FACTORY_REGISTRY` (`:65`), or imported by the live route / production wrapper, or transitively wrapped by something that is. `get_available_providers` (`factory.py:252`) is a **hardcoded dict of ~7 concretes** with no subclass scan — so an unregistered `*VisionProvider` subclass is reachable from nothing.

So A2b is **surgical triage**, not a bulk delete.

## 2. What stays (the live core, ~15.5K LOC)

`base`, `factory` + the ~7 concrete providers (`DeepSeek/OpenAI/Anthropic/Qwen/GLM4V/Doubao/stub`), `ResilientVisionProvider`, `VisionManager`, `create_vision_provider`, `get_available_providers`, `providers/`, and the API glue. Plus `circuit_breaker.py` **until #501** (see §5). After the prune, `vision/` goes from **82,249 → ~15,563 LOC** — which is exactly the roadmap's "core → ~15–20 real modules" milestone, reached honestly.

## 3. What gets deleted (87 files, ~66,686 LOC)

87 `*VisionProvider` scaffolding modules + `experimental/`. Every one: defined in its own feature-named module (`ab_testing`, `apm_integration`, `chaos_engineering`, `alert_manager`, `anomaly_detection`, `access_control`, …), carries a self-module `create_X_provider` helper never called externally, is **re-exported in `__init__.py` but consumed by nobody**, is in neither factory registry, and is reached by no dynamic/string dispatch. The full manifest (86 module files + `experimental/` + the 87 verdicts with per-class file:line evidence) is in the verification MD and `/private/tmp/cadml-a2b-delete-manifest.json`.

## 4. The make-or-break implementation detail: `__init__.py` surgery

`src/core/vision/__init__.py` has **465 `from .` import lines** and **imports all 87 deleted modules** (`ab_testing:26`, `apm_integration:176`, `chaos_engineering:263`, …). Because `import src.core.vision` — which the live route does — **executes every re-exported module**, deleting the 87 files without editing `__init__.py` **breaks the live package import**.

So the A2b PR must, in lockstep:
1. `git rm` the 87 module files + `experimental/`;
2. **strip the 397 `__init__.py` import lines** that reference a deleted module, **and** every `__all__` entry naming a symbol from a deleted module (each import pulls a *set* — e.g. `apm_integration` imports `APMConfig, APMManager, APMProvider, APMVisionProvider`, all of which go);
3. delete the corresponding `test_vision_phaseN.py` tests that import only deleted symbols (the ~18 phase-test files are the only other referrers).

`__init__.py`'s import count drops from 465 → ~68.

## 5. Cross-slice: `circuit_breaker` + the prune-safety gate

`CircuitBreakerVisionProvider` (`circuit_breaker.py`) is in the delete-set, but that file also holds the generic breaker that `dedupcad_vision.py` imports — which **#501** extracts to `resilience/advanced_circuit_breaker.py`. So:
- **A2b is BLOCKED until #501 merges.** After #501, `vision/circuit_breaker.py` is a pure zoo shim and joins the delete-set.
- The A2b PR **must move `src/core/vision/circuit_breaker.py` from `LIVE_TWINS` → `PRUNED_MODULES`** in `scripts/ci/prune_safety_check.py` (from #500) **in the same PR that deletes it** — too early disarms the guard, too late fails the gate. (Recorded identically in #501's MDs.)

## 6. Verification plan (what the A2b PR must prove — observed-RED)

The design-time triage says "safe"; the PR's **enforcement** is a runtime import smoke, because static reachability can be fooled and this deletes from a live package:
1. **Import smoke (the gate):** after delete + `__init__` surgery, `import src.core.vision` must succeed, `src/api/v1/vision.py`'s named imports must all resolve, and `create_vision_provider("stub"|"openai"|…)` must construct the 7 concretes.
2. **Observed-RED for it:** leave *one* deleted module's `__init__.py` import line in place → `import src.core.vision` raises `ImportError` (proves the surgery is load-bearing and an incomplete deletion is caught, not silently green).
3. **prune-safety extension:** the deleted module paths join `PRUNED_MODULES`; observed-RED that re-importing any of them reds the gate.
4. Positive control on every "0 references" grep (an unquoted `--include=*.py` once reported `importers=0` for live modules earlier in this program).

## 7. Blocked-on / sequencing
- **#501 merge** (removes the `circuit_breaker` dependency) — owner-gated merge window.
- **Backlog clearance** to under the 4-PR cap.
- Model: this design used the **9-agent adversarial triage on Opus 4.8** (Fable 5 was at its daily cap; per the tiering, Fable-cap → Opus 4.8, not Sonnet). The eventual mechanical deletion + `__init__` surgery is Sonnet-5-class spec-complete work.
