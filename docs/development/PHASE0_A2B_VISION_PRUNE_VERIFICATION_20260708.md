# Phase 0 · A2b verification — vision prune triage (design-time)

Companion to `PHASE0_A2B_VISION_PRUNE_DESIGN_20260708.md`. Grounded on `origin/main @ 8337ea6e`.
This documents how the delete/keep partition was derived and adversarially checked. The **runtime enforcement** (import smoke + observed-RED) is executed in the A2b PR itself — see design MD §6; it can't run until #501 is on main.

---

## 1. Reachability model + positive controls

A vision class is live iff it is in `factory.PROVIDER_REGISTRY` (`:34`) / `FACTORY_REGISTRY` (`:65`), imported by the live route (`api/v1/vision.py`) / production wrapper (`core/providers/vision.py`), or transitively reached from those. Every "0 references" probe below is paired with a control that must fire:

```
POSITIVE CONTROL: VisionManager referenced in 3 files  → OK (probe fires)
```

`import src.core.vision` succeeds today and executes all re-exported modules (confirmed by running it) — this is the invariant the prune must preserve.

## 2. Deterministic partition

93 `*VisionProvider` classes total. Against the reachability roots:
- **6 reachable (keep):** `AnthropicVisionProvider`, `DeepSeekVisionProvider`, `DoubaoVisionProvider`, `OpenAIVisionProvider` (in `PROVIDER_REGISTRY`), `VisionProvider` (base), `ResilientVisionProvider` (live import). *(Qwen/GLM concretes are also kept — reached via the factory registries; their class names differ slightly from the `*VisionProvider` pattern.)*
- **87 delete-candidates:** in neither registry, zero non-test references outside `src/core/vision/`, all `intra-vision-refs = 1` (self-module + a dead `__init__` re-export).

## 3. Adversarial triage — 9 agents, Opus 4.8

Each candidate was handed to an agent instructed to **prove reachability** (fail-safe toward KEEP) via the paths a name-grep misses: `__init__` re-export a consumer actually imports, transitive wrapping by `ResilientVisionProvider`/`VisionManager`/the factory chain, dynamic/string dispatch, non-test production instantiation.

**Result: 87 delete / 0 keep / 0 uncertain.** The agents did real work, not rubber-stamping — evidenced by nuances they caught:
- `CircuitBreakerVisionProvider`: flagged that `resilience.py`'s `CircuitState`/`CircuitBreakerConfig` hits are a **distinct in-file impl**, not this class.
- `ComplianceVisionProvider` vs `PrivacyComplianceVisionProvider`: distinguished two similarly-named classes and their separate helpers.
- `ConfigurableVisionProvider`: noted `"Config"` substring hits are `ProviderConfig`/`RetryConfig`, not this class.

The uniform verdict is `class def → self-module helper (never called externally) → dead __init__ re-export → absent from registries/live-route/dynamic-dispatch`.

## 4. Skeptical cross-check (87/87 with 0 uncertain is suspicious — so I verified the linchpins myself)

**Linchpin 1 — are the `__init__` re-exports actually dead?** Only if nothing does `import *` or dynamic attribute access on the package:
```
from src.core.vision import *  (outside vision/)  → 0 matches
getattr(...vision) / importlib(...vision) (outside vision/) → 0 matches
```
✅ The re-exports carry no consumer. Live code names only the ~8 live symbols. Verdict holds.

**Linchpin 2 — does `__init__.py` import the 87 (making them execute on package load)?**
```
__init__.py total `from .` import lines: 465
  ab_testing        → line 26   (imported)
  apm_integration   → line 176  (imported: APMConfig, APMManager, APMProvider, APMVisionProvider)
  chaos_engineering → line 263  (imported: ChaosExperiment, ChaosManager, ChaosVisionProvider, ErrorInjector)
```
✅ Confirmed, and this is the **critical constraint**: deleting a module without stripping its `__init__` line breaks `import src.core.vision`. **397** import lines reference deleted modules and must be removed in the same PR (design MD §4).

So the 87/87 is well-founded — it's uniformly agent-fleet-generated scaffolding with a consistent dead-re-export shape — **not** an over-eager verdict. But it is *design-time*; §5 is the runtime enforcement.

## 5. Sizing (measured)
```
delete: 87 files, ~66,686 LOC
vision/ today: 82,249 LOC  →  after prune: ~15,563 LOC (the "~15-20 modules" milestone)
__init__.py import lines to strip: 397  (465 → ~68)
```
Manifest with the full file list + `__init__` line numbers: `/private/tmp/cadml-a2b-delete-manifest.json`.

## 6. What the A2b PR must still do (not done here)
- Execute the delete + the 397-line `__init__` surgery + phase-test cleanup.
- Run the **import smoke** and bank its **observed-RED** (leave one `__init__` line → `ImportError`).
- Move `vision/circuit_breaker.py` `LIVE_TWINS → PRUNED_MODULES` in `prune_safety_check.py`, in the same PR, after #501.
- Positive control on every re-run grep.

## 7. Honesty note
This is **design groundwork**, not a completed slice. No code was deleted; no PR opened (backlog cap). The triage is thorough and cross-checked, but the load-bearing safety net is the runtime import smoke in the PR — a static reachability audit of a *live* package is necessary, not sufficient. Stated plainly rather than presented as "done."
