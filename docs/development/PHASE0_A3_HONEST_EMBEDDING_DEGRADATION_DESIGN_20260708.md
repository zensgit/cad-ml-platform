# Phase 0 · A3 — stop laundering a disclosed fallback (honesty sweep, slice 1)

- **Status**: FOR-REVIEW. Not merged. **Zero behavior change** — no vector, no provider selection, no flag changes.
- **Grounded on** `origin/main @ 8337ea6e`, verified by *running* the code.
- **Authorized by** the ratified positioning/roadmap design (merged in #499), §2.4 iron law: *"别再让目录名/README 声称代码没有的能力"* — don't let names/docs claim capabilities the code lacks.

---

## 1. The design doc's premise was wrong. Here is what actually happens.

The doc says `manufacturing_v2` *"silently degrades to TF-IDF"*. Running the code shows something more precise, and the real bug is one layer higher than the doc places it.

**Layer 1 — `models/embeddings/manufacturing_v2/`**: ships `config.json`, `tokenizer.json`, `training_corpus.jsonl`, `1_Pooling/` … and **zero encoder weights** (no `*.bin`, `*.safetensors`, `*.pth`).

**Layer 2 — `DomainEmbeddingModel`** has a documented three-tier load: fine-tuned sentence-transformer → base sentence-transformer → **TF-IDF character-ngram fallback that is "always available"**. With no weights it lands on tier 3. It does **not** raise, and it is **not silent** — it logs a warning and self-reports honestly:

```python
get_model_info() == {"name": "tfidf-fallback", "dimension": 512,
                     "fine_tuned": False, "fallback": True, "corpus_size": 0}
```

**Layer 3 — `DomainEmbeddingProvider` (the actual bug).** It read that dict, took only `dimension` and `name`, **discarded `fallback`**, set `available = True`, and logged:

```
INFO  DomainEmbeddingProvider loaded (dim=512, model=tfidf-fallback)
```

`create_semantic_retriever` then checked `candidate._available`, adopted it, and logged `Using DomainEmbeddingProvider (dim=512)`.

So: **the model layer disclosed the degradation; the provider layer laundered it away.** Downstream callers believed they were getting fine-tuned manufacturing-domain embeddings and were getting unfitted (`corpus_size: 0`) TF-IDF char-ngrams.

Two secondary corrections to the doc:
- The vectors are **not zeros**. They're real, unit-norm, and distinct per input — just far weaker than advertised. (The provider *does* have a zero-vector branch, but it only fires if `DomainEmbeddingModel` raises, which the always-available fallback prevents.)
- Zero vectors would be a *wrong-answer* mode (constant pairwise similarity), not "graceful degradation" as the old test docstring claimed.

## 2. The other defect: the tests were false-green

`tests/unit/test_domain_embedding_provider.py` asserted only shape:

```python
assert isinstance(result, list)
assert len(result) == provider.dimension
assert all(isinstance(v, float) for v in result)
```

**A vector of 384 zeros satisfies every one of those.** The suite passed identically whether the encoder worked or the provider had degraded — and its docstring explicitly said the tests were *"written to pass regardless"*. This is precisely the "假绿测试" the design doc names.

## 3. Change

- **`domain_embedding_provider.py`** — stop discarding the disclosure:
  - new public `available`, **`is_fallback`**, **`model_info`** properties (callers previously had to read the private `_available`);
  - construction logs a **warning** naming the fallback, `fine_tuned=False`, and `corpus_size` when the fine-tuned encoder is absent; the cheerful `INFO … loaded` is now reserved for a genuinely fine-tuned model;
  - module/class docstrings rewritten to state the verified truth (they previously advertised a *"fine-tuned"* checkpoint *"shipped with the repository"*).
- **`semantic_retrieval.py`** — `create_semantic_retriever` no longer logs a bare `Using DomainEmbeddingProvider`; when `is_fallback` it warns that the provider is **not** the fine-tuned encoder. Also replaces `candidate._available  # noqa: SLF001` with the public `candidate.available`.
- **`tests/unit/test_domain_embedding_provider.py`** — rewritten to discriminate. Centerpiece is `test_is_fallback_mirrors_model_self_report`: **re-discarding `fallback` makes it fail.** Also pins non-zero / unit-norm / distinct vectors, and encodes the false-green defect itself so nobody reintroduces a shape-only check believing it proves something.

## 4. Deliberately NOT changed (owner decisions)

- **`available` stays `True` on fallback.** Flipping it to `False` would make `create_semantic_retriever` skip this provider entirely and select a different one — a real behavior change to a live route. It may well be the right call; it is the owner's, not an unattended pass's.
- **The zero-vector branch stays.** It is a documented contract, and callers construct speculatively relying on no-raise.
- **Retired Claude model IDs are untouched.** The repo has `claude-3-opus-20240229` (retired 2026-01-05), `claude-3-sonnet-20240229` (retired 2025-07-21), and `claude-sonnet-4-20250514` (deprecated) across `vision_analyzer.py`, `vision/factory.py`, `vision/providers/anthropic.py`, `assistant/llm_providers.py`, `assistant/assistant.py`, `assistant/function_calling.py`. Not fixed here because: (a) `src/core/assistant/` is explicitly **multi-provider** (Claude, GPT, Qwen, Ollama), (b) the assistant is a **live registered route** (`api/__init__.py:266,494`), so changing default model IDs is a behavior *and cost* change, and (c) which replacement model to standardize on is an owner call. **Proposed as its own slice** — see §5.

## 5. Follow-ups
- **A3b (propose, not build)**: sweep the retired/deprecated Claude model IDs. Correct replacement per the current model catalog is `claude-sonnet-5` for the retired Sonnet IDs; `claude-opus-4-8` for `claude-3-opus-20240229`. Needs an owner decision on target model + cost, and touches a live route.
- **Owner decision**: should `available` be `False` when `is_fallback`? That would stop the fallback from being adopted at all.
- **Ship or delete the weights**: either publish `manufacturing_v2` encoder weights, or rename the directory/provider so the name stops claiming a fine-tuned model.
- **When #500 lands**, nothing to add to `PRUNED_MODULES` — this slice deletes no modules.
