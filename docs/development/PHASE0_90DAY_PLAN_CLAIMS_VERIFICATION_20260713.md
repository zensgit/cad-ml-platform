# Phase-0 90-Day Plan — Independent Claims Verification

**Date:** 2026-07-13
**Verified against:** `origin/main` @ `e2facd99` (pinned; the local checkout was 5 commits behind and was **not** used)
**Method:** model-tiered adversarial workflow, **default-refute** (a claim is REFUTED unless reproduced with concrete `file:line` evidence). 5 independent verifiers — Fable 5 ×3, Opus 4.8 ×1, Sonnet 5 ×1 — each reading authoritatively via `git show origin/main:<path>` / `git grep … origin/main`.
**Why this document exists:** the owner's 2026-07-13 conclusion pivots the entire next phase onto a small set of factual claims. Before spending 30–40 dev-days on that ordering, each load-bearing claim was independently verified. This is the §4.7 independent-critic mechanism standing in for a human reviewer while the repo is single-maintainer.

---

## TL;DR

The owner's **top-level instinct is validated**: the production security boundary — specifically **default-open auth** — is the real next P0, not model accuracy. But the adversarial pass **downgraded two specific sub-claims** (and corrected two of the verifier-author's own earlier inline calls):

- `/model/reload` is **P2, not P0** — it is gated by a second admin-token dependency, not `api_key` alone.
- `x-user-id` override is **latent, not active** — real in code, but its only reader is dead, uncalled code.

Both of those, plus the P0, collapse behind **one fix**: force non-default credentials and non-disabled auth in production.

The structural/quantitative claims (511 `src/core` files, ~115.8k deleted lines, Track E ~15%) **all reproduce**.

---

## Verdict table

| # | Claim (owner) | Verdict | Severity | Model |
|---|---|---|---|---|
| 1 | Default-open auth: `X-API-Key`/`ADMIN_TOKEN` default `test`, integration auth default `disabled`, and **no production guard** rejects them | **CONFIRMED** | **P0** | Sonnet 5 |
| 2 | `/model/reload` accepts caller path + `force`, reachable with default `api_key=test` | **PARTIAL** | P2 | Fable 5 |
| 3 | After JWT auth, `x-user-id` overrides token subject and downstream trusts it (real impersonation) | **REFUTED** | info (latent) | Fable 5 |
| 4 | Track E (eval-integrity-v2) ≈ 15% done; 6 core capabilities missing | **CONFIRMED** | info | Fable 5 |
| 5 | Exhaustive production activation/deserialization surface | **CONFIRMED** (list below) | info | Opus 4.8 |
| — | `src/core` = 511 `.py` / ~63 top dirs; ~113k deleted lines | **CONFIRMED** (511 exact; 115.8k over ~80 commits) | info | (inline) |

---

## 1. Default-open auth — CONFIRMED · **P0** (the real one)

**Claim holds unconditionally.** There is no environment-gated fail-closed.

- `src/api/dependencies.py:8` — `get_api_key(x_api_key = Header(default="test"))`; rejects only empty, never checks for the literal default.
- `src/api/dependencies.py:38` — `expected_token = os.getenv("ADMIN_TOKEN", "test")`; admin token defaults to `test` when unset.
- `src/core/config.py:27` & `src/core/config/__init__.py:50` — `INTEGRATION_AUTH_MODE = "disabled"` (two duplicate Settings classes, both default disabled).
- `src/api/middleware/integration_auth.py:46-49` — falls back to `disabled` on unset/invalid, and `if self.mode == "disabled": return await call_next(request)` skips all auth.
- `src/main.py:61-141` — the only startup validator (`_validate_optional_feature_flags`) checks GRAPH2D/FUSION flags only; **zero** references to `ADMIN_TOKEN` / `X-API-Key` / `INTEGRATION_AUTH_MODE`.
- `config/feature_flags.py:34,79-81` — the nearest thing to a guard only `warnings.warn(...)` (non-fatal), checks emptiness not the `test` literal, is independent of the real check at `dependencies.py:38`, and has **no** `ENVIRONMENT`/production condition.
- `ENVIRONMENT` is read only for OTel resource tagging (`src/core/observability/*`), never branched on for auth.

**Conclusion:** a deployment left at defaults has effectively no authentication, in every environment. This is the load-bearing security finding and the anchor for **Track I (production identity fail-closed)**.

## 2. `/model/reload` path + force — PARTIAL · **P2**

Two independent sub-claims resolve differently:

- **Caller path + force: CONFIRMED.** `src/api/v1/model.py:46` `POST /api/v1/model/reload` → `reload_model(payload.path, expected_version=…, force=payload.force)` (`:71`). `ModelReloadRequest` carries `path: Optional[str]`, `force: bool=False` (`:19-24`). The path is deserialized via pickle at `src/ml/classifier.py:535` — **behind** magic-number / hash-whitelist / size / opcode-scan guards, not a raw load.
- **"reachable with only `api_key=test`": REFUTED.** The endpoint *also* requires `Depends(get_admin_token)`; a request with only `X-API-Key` and no `X-Admin-Token` gets **401** (`dependencies.py:14-31`).

**Residual risk = weak default, not open door:** `ADMIN_TOKEN` also defaults to `test`, so a misconfigured deployment (ADMIN_TOKEN unset) is reachable by an attacker who sends `X-Admin-Token: test` and a server-resident file that passes the pickle guards. **This is subsumed by fixing #1** — once production refuses default creds, this endpoint sits behind a real admin token. The remaining hardening (disable dynamic reload in prod entirely) is **Track F (activation freeze / #513 Phase A)**.

Other reload endpoints are not path loaders: `/knowledge/reload` (`maintenance.py:92`) and `/vectors/backend/reload` (`maintenance.py:614`) are cached-instance reloads with no path; the deprecated `/analyze/model/reload` only raises a redirect.

## 3. `x-user-id` override — REFUTED · latent (info)

The override is **real in code** but currently **inert**:

- `integration_auth.py:104` — `user_id = request.headers.get("x-user-id") or str(subject)` (spoofable); `:108` stores it to `request.state.user_id`; `:109` stores the authentic `subject` to `request.state.auth_subject`. Tenant *is* validated (`:95` → 401 on mismatch); user is **not**.
- **But** the *only* reader of `request.state.user_id` is `create_api_actor_from_request` (`src/core/audit/service.py:531`), and that function has **zero callers** on `origin/main` (dead code). `request.state.auth_subject` has **zero** consumers. So no live authorization or attribution decision reads the spoofable value.
- Separately, several modules read the **raw** `x-user-id` header directly (`audit/logger.py:614`, `feature_flags/decorators.py:136`, `request_context/__init__.py:213`) — a pre-existing surface independent of the JWT branch, behaving identically with or without the middleware.

**Conclusion:** not an active impersonation exploit; it is a latent trap (if a route were wired to `create_api_actor_from_request`, audit attribution would become spoofable within a tenant). Cheap correct fix belongs in **Track I**: bind identity to `auth_subject`, drop the header override, and delete the dead sink or wire it correctly.

## 4. Track E ≈ 15% — CONFIRMED (bound 5–15%)

**0 of the 7 `PRODUCT_STRATEGY.md §8.1` deliverables are implemented on `origin/main`.** What exists is scaffolding, not features:

- The merged surface is only the fail-closed seam: `scripts/eval_integrity_gate.py:63-73` (`check()` = single unconditional `raise GateBlocked`; `main()` ignores argv), `scripts/auto_retrain.sh:54-60` (Step 0 runs it, exits 1), and 2 test files asserting the bypass is *absent* (`tests/unit/test_eval_integrity_gate.py`).
- **Missing:** content-hash leakage detection (governance gate `check_training_data_governance.py:146-160` is **path-only**), family/label split (`golden_*_set.csv` headers are `file_path,cache_path,taxonomy_v2_class` only — no hash/family/split/source/license columns), conflict quarantine, portable/versioned manifest, real/synth/augmented separated metrics, two-phase candidate-model binding.
- Track E implementation exists **only on the now-closed branches** `origin/claude/track-e-*-20260712` (PRs #510/#511), deliberately excluded per the L3 ratified order.

**Two corrections for downstream docs:**
- The gate lives at `scripts/eval_integrity_gate.py`, **not** `scripts/ci/eval_integrity_gate.py`.
- `scripts/validate_brep_golden_manifest.py` (schema `brep_golden_manifest.v1`, with license/provenance fields) **is** a versioned provenance manifest, but it governs the **B-Rep (STEP/IGES) benchmark**, not the 2D train/val split — do not cite it as Track E item 6.

## 5. Activation surface — CONFIRMED (authoritative enumeration)

The design has miscounted this ≥3×. This is the exhaustive list at `e2facd99`, to be enforced by the **Track F CI activation-surface enumerator** (not a hand-count):

**Production-reachable (mounted via `src/main.py` routers/startup; nearly all lazy on first use, several env-flag-gated):**

| # | Location | Loads | Deserializes arbitrary objects? | Reach |
|---|---|---|---|---|
| 1 | `src/ml/classifier.py:85` | sklearn `.pkl` (`CLASSIFICATION_MODEL_PATH`) | **pickle — yes** | lazy via `predict()`; model/shadow/health |
| 2 | `src/ml/classifier.py:535` | reload path | **pickle — yes** | `/api/v1/model/reload` + auto_remediation rollback |
| 3 | `src/core/vectors/stores/faiss_store.py:96` | faiss `.meta` sidecar | **pickle — yes** | startup when `VECTOR_STORE_BACKEND=faiss` |
| 4 | `src/ml/part_classifier.py:655,695` | PartClassifier **V16 ensemble** | **torch `weights_only=False` — yes** | **primary hot path** via `analyze` router |
| 5 | `src/ml/part_classifier.py:62` | base PartClassifier | **torch `weights_only=False` — yes** | analyzer/classify/batch/decision |
| 6 | `src/ml/vision_2d.py:136` | Graph2DClassifier | torch `weights_only=False` | providers/classifier + health; `GRAPH2D_ENABLED` |
| 7 | `src/ml/vision_3d.py:196` | UVNet encoder | torch (default varies) | feature_pipeline; `UVNET_ENABLED` |
| 8 | `src/ml/pointnet/inference.py:108` | PointNet | torch (default varies) | pointcloud router; `POINTNET_ENABLED` |
| 9 | `src/ml/hybrid_classifier.py:448` | StatMLP | **torch `weights_only=False` — yes** | providers/assistant; `stat_mlp_enabled` |
| 10 | `src/ml/hybrid_classifier.py:476` | TF-IDF TextMLP | **torch `weights_only=False` — yes** | providers/assistant; `tfidf_text_enabled` |
| 11 | `src/ml/history_sequence_classifier.py:162` | history-sequence | torch | hybrid `@property` |
| 12 | `src/core/assistant/semantic_retrieval.py:156` | SentenceTransformer **by name** (HF cache) | no (not a file pickle) | assistant router |
| 13 | `src/core/assistant/embedding_retriever.py:259` | `np.load` **`allow_pickle=False`** | no (numeric `.npy`) | assistant knowledge_retriever |

**Arbitrary-object deserializers in prod = items 1, 2, 3 (pickle) + 4, 5, 9, 10, 11 (torch `weights_only=False`)** — the 8 points a `weights_only`/allow-list freeze must cover; 6/7/8 depend on the torch default at their call sites.

**Not prod-reachable (must be excluded from the freeze, documented so the enumerator's count is defensible):**
- **Separate standalone app:** `src/inference/classifier_api.py:613,627` (its own FastAPI + lifespan; reachable only if deployed as its own service — a deployment question source can't resolve).
- **Dead (method never called):** `confidence_calibrator.py:411` (`load_calibrator` — 0 callers), `anomaly_detector.py:337,342` (`load_models` — 0 callers).
- **Unmounted:** `src/ml/serving/*`, `src/core/model/hot_reload.py:166`, `src/ml/pipeline/stages.py:252`, `src/ml/compression/quantization.py:332`.
- **Training/eval only:** `src/ml/train/*`, all `scripts/*.py` CLI entrypoints.

---

## Corrected P0 ordering (consequence of this verification)

1. **Track I — production identity fail-closed (the P0).** In production, refuse default `test` creds and `INTEGRATION_AUTH_MODE=disabled` (boot-refuse or reject). This single fix also puts `/model/reload` (#2) behind a real admin token.
2. **Track I (cheap add) — bind identity to `auth_subject`**, drop the `x-user-id` override, resolve the dead sink (#3).
3. **Track F — activation freeze (#513 Phase A):** disable dynamic `/model/reload` in prod; fail-closed on unproven startup loads; a CI enumerator that asserts the 8-point arbitrary-deserializer surface above is fully covered (not hand-counted).
4. **Track E — eval-integrity-v2:** the 7 §8.1 items, rebuilt fresh from `origin/main` after governance gates (#512 + design-lock ratification).

## Method notes

- **Fixed HEAD, default-refute, positive controls.** Verifiers were instructed to REFUTE unless reproduced; the Track E verifier ran a positive-control grep (same pattern set found 2 `def` in the gate) to prove a null result wasn't a silent grep failure.
- **Scope boundary:** verification is against `origin/main` only. The unmerged `track-e-*` branches were deliberately excluded (they may contain partial implementations) per the L3 ratified execution order.
- **Author self-correction:** the inline pre-workflow assessment overstated #2 (P0→P2) and #3 (active→latent). Recorded here rather than silently amended, per evidence-integrity discipline.

*Companion document:* `PHASE0_90DAY_EXECUTION_PLAN_DESIGN_20260713.md` (development order, model tiering, buildable-vs-owner-gated matrix).
