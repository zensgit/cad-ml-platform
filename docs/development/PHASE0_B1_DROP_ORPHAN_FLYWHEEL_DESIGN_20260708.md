# Phase 0 · B1 — delete the orphan `FeedbackLearningPipeline` (delete, don't wire)

- **Status**: FOR-REVIEW. Not merged. Zero behavior change.
- **Grounded on** `origin/main @ 8337ea6e`.
- **Authorized by** the ratified positioning/roadmap design (merged in #499), track B: *"删重复死环: feedback_log.jsonl(占位) + 孤儿 FeedbackLearningPipeline(EMA 权重 classifier 不读回)= 删,不是接"*.

---

## 1. Why "delete", not "wire"

The design doc's most consequential correction was that the flywheel's missing piece is a **feedback source**, not a reconnection. `FeedbackLearningPipeline` looks like the missing link — its package docstring literally claims the module *"closes the feedback loop between user corrections and model improvement"*. It does not.

Verified at `8337ea6e`:

| claim | evidence |
|---|---|
| It has no production consumer | only importers are its own package `__init__.py:9` and `tests/unit/test_feedback_loop.py`. Zero elsewhere. |
| Its output is never read back | `weight_history` has **0** readers outside the module. The only `_current_weights` hits in the tree belong to an unrelated variable in `src/core/api_gateway/loadbalancer.py`. |
| The live classifier ignores it | `HybridClassifier` resolves its fusion-branch weights from env/config at construction, not from `weight_history.jsonl`. |

So it computed EMA fusion weights and persisted them to a file **nothing ever loaded**. Wiring it would mean *inventing* an inference-side reader plus a reset trigger — a build, not a reconnection. The real spine already works (classifier → `low_conf.csv` → `auto_retrain.sh`, behind a hard governance gate); what's missing is a feedback **source** and a human-review action. Keeping dead plumbing that advertises a working feedback loop is precisely the "code claims a capability it lacks" dishonesty the design doc's §2.4 iron law forbids.

## 2. Change (740 LOC removed)

- **delete** `src/ml/learning/feedback_loop.py` (418 LOC) — defines exactly one symbol, `FeedbackLearningPipeline`.
- **delete** `tests/unit/test_feedback_loop.py` (322 LOC) — its only consumer.
- **rewrite** `src/ml/learning/__init__.py`: drop the import and `__all__` entry; replace the false "closes the feedback loop" docstring with an honest note explaining *why* the class was removed and where the real flywheel work lives.

### What must survive
`SmartSampler` lives in the **separate** module `src/ml/learning/smart_sampler.py` and is genuinely live (`scripts/run_performance_baseline.py`, `tests/performance/test_benchmark_new_modules.py`). `feedback_loop.py` imported it, not the reverse, so deletion is safe. It remains the package's sole export. Neither consumer file is touched by this PR.

Nothing imports the `ml.learning` **package** as a whole (`from src.ml.learning import …` → 0 hits), so changing `__all__` cannot break a caller.

## 3. Explicitly NOT in scope

The design doc pairs this deletion with removing the `feedback_log.jsonl` placeholder (`src/api/v1/feedback.py:229`). **That is deliberately excluded.** Deleting the JSONL write changes the behavior of a **live HTTP endpoint** (`POST /feedback` currently persists there). That is a product decision about what the endpoint promises, not dead-code hygiene, and it should not be done blind in an unattended pass. It is split into its own slice.

## 4. Risk & rollback
- **Behavior change: none.** No production path imported the deleted class; the package's remaining export is unchanged.
- **Orphaned artifacts**: `weight_history.jsonl` and `data/feedback/corrections.jsonl` are no longer written. Nothing read them, so no consumer breaks. Existing files on disk are left alone (data, not code).
- **Rollback**: one `git revert`. No flags, no schema, no CI/branch-protection change.

## 5. Follow-ups (cross-slice)
- **When #500 lands**, add `ml.learning.feedback_loop` to `PRUNED_MODULES` in `scripts/ci/prune_safety_check.py`, so the gate stops the fleet from resurrecting it. (Same pattern as A2a's note about `vision/circuit_breaker`.)
- **Feedback-source slice** (track B, design-lock first): a correction channel + a human-review action that fills `reviewed_label` / `human_verified` in `low_conf.csv`. `auto_retrain.sh` gates on `MIN_REVIEWED=200`, so the pipe stays dry until this exists.
- **`feedback_log.jsonl` slice**: decide what `POST /feedback` should durably promise, then implement.
