# Phase 0 · B1 verification — delete the orphan `FeedbackLearningPipeline`

Companion to `PHASE0_B1_DROP_ORPHAN_FLYWHEEL_DESIGN_20260708.md`. Grounded on `origin/main @ 8337ea6e`.

---

## 1. Positive control (the probe must be able to fire)

Every "zero references" claim below is worthless unless the probe actually runs. Control:

```
ml.classifier importers = 59      ✅ probe fires
```

(This session already produced one near-miss where an unquoted `--include=*.py` was glob-eaten by zsh and reported `importers=0` for demonstrably live modules. Every grep here uses quoted globs and is paired with this control.)

## 2. Orphan evidence

```
class FeedbackLearningPipeline   → src/ml/learning/feedback_loop.py:47

importers of ml.learning.feedback_loop:
  src/ml/learning/__init__.py:9        (its own package re-export)
  tests/unit/test_feedback_loop.py:13  (its only test)
  — nothing else in src/, tests/, scripts/
```

Top-level symbols in `feedback_loop.py`: **exactly one** (`FeedbackLearningPipeline`). No other symbol from that module is referenced anywhere outside it.

Nobody imports the `ml.learning` package as a whole (`from src.ml.learning import …` → **0 hits**), so mutating `__all__` cannot break a caller.

## 3. The load-bearing claim — the EMA weights are write-only

The design doc asserts the pipeline's output is never read back. Verified:

```
weight_history            → 0 readers outside feedback_loop.py
_current_weights          → only src/core/api_gateway/loadbalancer.py:130,138,139,143
                            (an unrelated load-balancer variable, not classifier weights)
```

So the pipeline persisted EMA fusion-branch weights that **no inference path ever loaded**. This is what makes it "delete", not "wire": reconnecting it would require inventing an inference-side reader and a reset trigger.

## 4. What must survive — `SmartSampler`

`feedback_loop.py:24` imported `SmartSampler`, so the dependency direction matters. `SmartSampler` lives in a **separate module** (`src/ml/learning/smart_sampler.py`) and is live:

| consumer | |
|---|---|
| `scripts/run_performance_baseline.py` | live script |
| `tests/performance/test_benchmark_new_modules.py` | perf suite |
| `src/ml/learning/__init__.py` | remaining sole export |

Files modified in those consumers by this PR: **0**.

## 5. Post-delete verification

```
'FeedbackLearningPipeline'      → 1 reference   (the __init__ docstring explaining the removal)
'ml.learning.feedback_loop'     → 0 references
smart_sampler.py                → present ✅
consumers modified              → 0 ✅
```

AST check of the rewritten package `__init__`:
```
imports:          ['src.ml.learning.smart_sampler']
__all__:          ['SmartSampler']
executable lines: ['from src.ml.learning.smart_sampler import SmartSampler']
✅ no dangling import; __all__ correct; only the docstring mentions the removed class
```

## 6. Runtime import smoke — real execution, not compilation

This slice applies the lesson A2a paid for: `py_compile` and AST checks **cannot** catch a `NameError` or a dangling import that only fires at module execution. So the package is actually imported:

```
✅ import src.ml.learning OK; __all__ = ['SmartSampler']
✅ SmartSampler resolvable: True
✅ FeedbackLearningPipeline no longer exported
```

Plus `py_compile` on `__init__.py` and `smart_sampler.py`: clean.

## 7. Orphaned data artifacts (documented, not deleted)
`weight_history.jsonl` and `data/feedback/corrections.jsonl` are no longer written. Nothing ever read them, so no consumer breaks. Any files already on disk are left untouched — they are data, and deleting user/runtime data is not in scope for a code-hygiene slice.

## 8. Deliberately out of scope
`feedback_log.jsonl` (`src/api/v1/feedback.py:229`) is **not** touched. Removing that write changes what the live `POST /feedback` endpoint durably promises — a product decision, not dead-code hygiene, and not something to do blind in an unattended pass. Split into its own slice.

## 9. What CI verifies that local cannot
Local Python is 3.9; the project targets 3.11. The package import above succeeds locally because `smart_sampler` is importable, but the full suite (and the app-level import smoke in `tests/test_routes_smoke.py`) runs only in CI. Stated rather than glossed.

## 10. Residual risk
- **Cross-slice**: once #500 lands, `ml.learning.feedback_loop` must be added to `PRUNED_MODULES` in `scripts/ci/prune_safety_check.py`, or nothing stops the fleet from re-creating it.
