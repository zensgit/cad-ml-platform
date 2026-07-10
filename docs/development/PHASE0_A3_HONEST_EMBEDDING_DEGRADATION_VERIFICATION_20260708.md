# Phase 0 · A3 verification — stop laundering a disclosed fallback

Companion to `PHASE0_A3_HONEST_EMBEDDING_DEGRADATION_DESIGN_20260708.md`. Grounded on `origin/main @ 8337ea6e`.

---

## 1. The premise was corrected by *running* the code (twice)

**First attempt — wrong.** Reading `domain_embedding_provider.py` showed a `try/except` whose `except` branch returns zero vectors, so I wrote a docstring asserting *"this provider is `available == False` in every current checkout"* and *"returns zero vectors."*

Executing it refuted that in one line:

```
available=True  dim=512
embed_text all-zeros=False
AssertionError: expected today's reality: unavailable -> zeros
```

The `except` branch never fires. **I had shipped a false docstring into the working tree and the run caught it.** This is the fourth time in this Phase-0 program that reading was insufficient and executing was decisive.

**Ground truth**, from `DomainEmbeddingModel` directly:

```
get_model_info() = {'name': 'tfidf-fallback', 'dimension': 512,
                    'fine_tuned': False, 'fallback': True, 'corpus_size': 0}
encode shape=(2, 512)  distinct=True  norm0=1.0000
```

`manufacturing_v2/` weight-file count: **0** (`*.bin`, `*.safetensors`, `*.pth`, `*.h5`).

So the model lands on its "always available" TF-IDF char-ngram tier, **discloses it**, and the *provider* was the layer throwing that disclosure away.

## 2. Post-change runtime state

```
available=True  is_fallback=True  dim=512
model_info={'name': 'tfidf-fallback', 'dimension': 512, 'fine_tuned': False,
            'fallback': True, 'corpus_size': 0}
nonzero=True  norm=1.0000  distinct=True
✅ reality matches the new tests' expectations
```

The degradation is now visible through the public API (`is_fallback`, `model_info`) and is logged as a **warning** at construction and again when `create_semantic_retriever` adopts the provider.

## 3. Observed-RED — the anti-laundering guard genuinely fails

A guard that only ever passes proves nothing. Reconstructed the **pre-fix behavior** (reads `model_info`, discards `fallback`) and ran the new assertion against it:

```
=== OBSERVED-RED: does the anti-laundering assertion catch the OLD behavior? ===
  ✅ observed-RED: is_fallback must mirror model_info['fallback'] --
     provider is discarding it again
     (old.is_fallback=False vs model_info['fallback']=True)
```

And on the fixed provider the same assertion passes. So `test_is_fallback_mirrors_model_self_report` is a real regression guard, not decoration.

## 4. False-green proof

The original suite's assertions, applied verbatim to a vector of zeros:

```
isinstance(zeros, list)                    -> True
len(zeros) == provider.dimension           -> True
all(isinstance(v, float) for v in zeros)   -> True
```

All pass. The old tests therefore could not distinguish a working encoder from a fully degraded one — and their own docstring said they were *"written to pass regardless."* `TestFalseGreenRegression::test_shape_only_assertions_cannot_detect_zero_vectors` now encodes that fact so the pattern isn't reintroduced.

## 5. What the new tests actually pin

| test | fails when |
|---|---|
| `test_is_fallback_mirrors_model_self_report` | the provider discards `model_info['fallback']` again (**observed-RED above**) |
| `test_fallback_and_fine_tuned_are_consistent` | something reports `fallback=True, fine_tuned=True` |
| `test_embed_text_is_nonzero_and_unit_norm_when_available` | an "available" provider emits zeros, or drops `normalize=True` |
| `test_distinct_texts_give_distinct_vectors` | vectors collapse (the zero-vector failure mode) |
| `test_available_is_public_and_boolean` | the public accessor drifts from `_available` |
| `test_shape_only_assertions_cannot_detect_zero_vectors` | — (documents the defect) |

All branch on `available` / `is_fallback`, so they stay correct if someone later ships real encoder weights.

## 6. Local pytest is blocked — environmental, proven with two controls

`python3 -m pytest tests/unit/test_domain_embedding_provider.py` reports 9 **errors** (not failures). Root cause is **not** this change:

```
tests/conftest.py:214: in migration_history_isolation   (autouse fixture)
    import src.api.v1.vectors as vectors_mod
E   TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

`src/api/v1/vectors` uses PEP-604 `X | None`, which is a `TypeError` on the local Python 3.9; the project targets 3.11.

Two independent controls confirm it is environmental:

| control | result |
|---|---|
| `tests/unit/test_feedback_loop.py` (untouched file) | 15 errors, same conftest `TypeError` |
| the **original** `test_domain_embedding_provider.py` (git-stashed my version) | 4 errors, same conftest `TypeError` |

So the entire `tests/` tree is unrunnable locally, before and after this change. **CI (3.11) runs the suite.** The runtime evidence in §2/§3 was gathered by importing the modules directly, bypassing `conftest`, so the logic *was* actually executed — not merely compiled. Stated plainly rather than glossed.

## 7. Behavior change: none
- Vectors returned: unchanged (same model, same `normalize=True`).
- Provider selection: unchanged — `create_semantic_retriever` still adopts the provider when `available`; only its **log level and message** change.
- No flags, no schema, no CI, no branch-protection change. `available` deliberately still `True` on fallback (see design MD §4 — flipping it is an owner decision).
- `compile` clean on all three touched files.

## 8. Residual risk
- The provider is still *named* `DomainEmbeddingProvider` and still selected while serving TF-IDF. This slice makes that **loud**; it does not make it stop. Ship weights, or rename, or flip `available` — all owner calls.
- `corpus_size: 0` means the TF-IDF fallback isn't fitted on the shipped `training_corpus.jsonl`. Not investigated here; worth its own look.
