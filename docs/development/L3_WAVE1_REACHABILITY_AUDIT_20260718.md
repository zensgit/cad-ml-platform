# L3 Wave-1 Reachability Audit — the 38 conservatively-gated activation sites

**Date:** 2026-07-18 · **Grounding:** re-verified against `main@7160694d` (post-#526) · **Read-only:** no loader, manifest, runtime, or CI change — this document and its JSON evidence companion are the only artifacts.

This is the Wave-1 deliverable of the ratified Phase-A plan: a per-site **logical-reachability audit** of the 38 sites classified `gated` by the activation-surface enumerator (`scripts/ci/activation_surface.json`, currently 129 sites / 38 gated / 11 families). The post-audit total-site increase is one `offline` UVNet inspector site; the 38-site gated set and this audit's denominator are unchanged. It answers, for each site: does a real caller chain from the production service reach the **load call itself**, and under which conditions?

## 1. Method

- **Production anchor (verified first, then every chain is walked from it):** the container runs `python -m uvicorn src.main:app` (Dockerfile CMD, 3 occurrences); `src/main.py:410` `app.include_router(api_router, prefix="/api")`; `:414` mounts `/metrics`. "Production-reachable" = reachable from `src.main:app`'s mounted routes, startup/lifespan hooks, or background tasks it schedules.
- **Labels:** **LIVE** — a proven hop-by-hop caller chain from the anchor to the load call (trigger recorded: startup / lazy-first-request / per-request / env-conditioned). **LATENT** — importable from live modules but no caller reaches the load call (gate-before-wired). **UNMOUNTED** — reachable only inside an app object/entrypoint the production command does not serve. **OFFLINE** — reachable only from scripts (training/eval/CLI).
- **Discipline:** import ≠ load; a lazy double-checked first-load triggered by a request is LIVE lazy-first-request, not startup; docstrings saying "initial load" are not evidence; the manifest `reason` strings were treated as seed hypotheses and re-proven or corrected, never copied.
- **Model routing (per the ratified plan):** every family was audited by one agent and then **adversarially re-verified by an independent opus agent** (LIVE chains re-walked hop-by-hop in source; non-LIVE labels attacked by hunting for missed callers, including dynamic dispatch, provider/DI registries, lifespan hooks, and background tasks). Audits: opus for pickle-classifier / anomaly-monitor / part-v16, sonnet for the other eight.
- **Denominator rule (ratified):** `in_phase_a_denominator == (label == LIVE)`. Latent/unmounted/offline sites are recorded gate-before-wired and do **not** pad the Phase-A completion denominator.

## 2. Result summary

| Family | Sites | LIVE | LATENT | UNMOUNTED | OFFLINE | Routing (audit + verify) |
|---|---|---|---|---|---|---|
| pickle-classifier | 3 | 1 | 2 |  |  | opus (audit) + opus (verify) |
| anomaly-monitor | 2 |  | 2 |  |  | opus + opus |
| part-v16 | 4 |  |  | 4 |  | opus + opus |
| graph2d | 2 | 2 |  |  |  | sonnet + opus |
| history | 2 | 2 |  |  |  | sonnet + opus |
| hybrid | 4 |  | 4 |  |  | sonnet + opus |
| ocr | 5 | 4 | 1 |  |  | sonnet + opus |
| part | 6 | 6 |  |  |  | sonnet + opus |
| pointnet | 5 | 5 |  |  |  | sonnet + opus |
| vision3d-uvnet | 2 | 2 |  |  |  | sonnet + opus |
| embedding | 3 | 1 | 1 |  | 1 | sonnet + opus |
| **Total** | **38** | **23** | **10** | **4** | **1** | |

**Phase-A completion denominator: 23 LIVE sites, grouping into 11 proposed logical activations:**

- `embedding/assistant-retriever-minilm-l12` — 1 site(s)
- `graph2d/main` — 2 site(s)
- `history/sequence` — 2 site(s)
- `ocr/deepseek-hf-mini` — 2 site(s)
- `ocr/deepseek-hf-paddle-align` — 1 site(s)
- `ocr/paddle-primary` — 1 site(s)
- `part/v16-v6pt` — 4 site(s)
- `part/v6` — 2 site(s)
- `pickle-classifier/main` — 1 site(s)
- `pointnet/main` — 5 site(s)
- `vision3d-uvnet/main` — 2 site(s)

The 15 non-LIVE sites (10 LATENT + 4 UNMOUNTED + 1 OFFLINE) are **gated-before-wired**: Phase-A records them and the enumerator keeps them `gated`, but they are NOT wired in W3 and NOT counted toward completion.

## 3. Seed-map corrections (what this audit changed vs. the design-lock seed map)

1. **hybrid — all 4 sites LATENT under the repository-shipped production-container anchor; the seed "reached via classify tools" is wrong twice over.** (a) `ClassifyTool.execute` calls `classify(file_id)` positionally, so `file_bytes=None` and the stat-MLP/TF-IDF gates can never fire via that tool; the tool's own drivers (`FunctionCallingEngine`, `AnalysisReportGenerator`) are never instantiated in `src/` anyway. (b) The REAL near-miss caller is the live analyze shadow pipeline — `HybridClassifier.classify()` itself IS live (the graph2d chain runs through it) and the property getters ARE reached — but the load lines inside `stat_mlp` (:448/:453) and `tfidf_text_classifier` (:476/:481) sit behind a `scripts.*` import (:445/:473). Every repository-shipped production container excludes `scripts/`, so those imports fail before the load calls. Direct source-tree serving can import `scripts/`; it is outside this production-container denominator and would require re-auditing these four sites if adopted as a production deployment. Lesson: audit the **load line**, not the enclosing function.
2. **part-v16 (`src/inference/classifier_api.py::V16Classifier.load`) — UNMOUNTED ×4 confirmed.** The `classifier_api` FastAPI app is not served by the production command; `health.py` imports only `result_cache`. (Consistent with the pre-ratification correction in #513.)
3. **embedding — the seed conflated two same-named class hierarchies.** `embedding_retriever.py` (wired into the assistant) and `semantic_retrieval.py` (parallel, only reachable via the orphaned `SimilarityTool`) both define `EmbeddingProvider`/`SemanticRetriever`. Result: assistant retriever **LIVE** (lazy-first-request via `POST /api/v1/assistant/query`, default HYBRID mode), `SentenceTransformerProvider` **LATENT** (sole ctor site requires `use_transformers=True`, which no caller passes), `DomainEmbeddingModel` **OFFLINE** (only `scripts/train_domain_embeddings.py --demo` reaches it — the seed's "production embedding path" over-claims).
4. **ocr — the warmup PaddleOCR ctor is LATENT.** `DeepSeekHfProvider.warmup()` has zero production callers (no lifespan/startup warmup wiring, no admin route; readiness only introspects module state). The other 4 ocr sites are LIVE. The manifest reason "MOUNTED /ocr" conflated the file's mount status with this method's reachability.
5. **pickle-classifier reload — a second gate-before-wire upstream found.** Beyond the mounted-but-sealed-403 `/api/v1/model/reload` route, `auto_remediation._action_rollback_model` → `reload_model` → `_reload_model_impl` → `pickle.loads(:535)` is a second would-be path; both are unwired/sealed, so the site stays LATENT. (LATENT, not UNMOUNTED: the `/model` router IS served — it is unreachable because of the runtime 403 seal.)

## 4. Operational notes surfaced by the audit (no action taken — Wave-1 is read-only)

- **pickle-classifier/main is LIVE but dormant in this checkout:** `models/classifier_v1.pkl` is absent, so `load_model` short-circuits at the `:79` exists-guard until an operator supplies the artifact (env condition `CLASSIFICATION_MODEL_PATH`).
- **embedding/assistant-retriever is LIVE but currently always falls to its TF-IDF fallback:** `sentence-transformers` is NOT installed in the shipped image (absent from `requirements.txt`; commented out even in `requirements-assistant.txt`, which the Dockerfile never installs) — the ctor sits behind `try/except ImportError`. The call graph is live; the optional dependency gates the actual load.
- **The whole `part` family is hard-gated by `PART_CLASSIFIER_PROVIDER_ENABLED` (default `false`).** With it true, the default `PART_CLASSIFIER_PROVIDER_NAME=v16` selects `part/v16-v6pt`; the two `part/v6` sites additionally require `PART_CLASSIFIER_PROVIDER_NAME=v6`. Both classes are genuinely selectable (`.env.example` documents `v16|v6`).
- **graph2d loads via a module-level import side-effect singleton** (`vision_2d.py:266`), fired lazily by whichever live path first imports the module (hybrid fallback path is live under shipped defaults); its `load_state_dict#0` branch only executes for legacy checkpoint shapes (`arch != GraphEncoderV2`) — a data-shape condition of the same load.
- **DeepSeek OCR loads are pin-gated:** `from_pretrained` requires `DEEPSEEK_HF_REVISION` set to a commit hash, or `DEEPSEEK_HF_ALLOW_UNPINNED=1`; otherwise the provider stubs. Paddle is the default provider for `strategy='auto'`.

## 5. Open questions for the owner (block W2/W3 modeling, not W2 build start)

1. **Two-distinct-files, one logical activation (KIND decision needed before W3 wiring of `part/v16-v6pt`):** `PartClassifierV16._load_models` loads `cad_classifier_v6.pt` (:655/:683) AND `cad_classifier_v14_ensemble.pt` (:695/:698) in one activation — options: two tuples/one-id, per-file pins, or bundle KIND. The same shape exists in the UNMOUNTED `classifier_api.py::V16Classifier.load` (out of Phase-A scope, still needs the decision recorded).
2. **Reload pathway id-modeling:** `pickle-classifier` sites 2+3 (`_reload_model_impl::pickle.loads` and `auto_remediation::reload_model()`) are two call points of ONE logical reload/rollback activation — one id (`pickle-classifier/reload`) vs two pins.
3. **OCR alignment instance:** `DeepSeekHfProvider`'s inline `self._paddle` aligner is a separate PaddleOCR instance from `paddle.py`'s primary provider — kept as its own activation id (`ocr/deepseek-hf-paddle-align`).

## 6. Per-site findings (verified labels, triggers, and caller-chain evidence)

### pickle-classifier — LIVE×1 / LATENT×2

| Site | Label | Trigger | kind | activation id | conf. |
|---|---|---|---|---|---|
| `load_model::pickle.load#0` | **LIVE** | lazy-first-request (lazy-first-predict) | single-file | `pickle-classifier/main` | high |
| `_reload_model_impl::pickle.loads#0` | **LATENT** | none live | single-file | `pickle-classifier/reload` | high |
| `AutoRemediation._action_rollback_model::reload_model()#0` | **LATENT** | none live | single-file | `pickle-classifier/reload` | high |

<details><summary><code>src/ml/classifier.py::load_model::pickle.load#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request (lazy-first-predict) — load_model (classifier.py:47) uses double-checked locking; the one-time pickle.load (classifier.py:85) fires on the first analyze request that reaches classification, then _MODEL is cached in the module global. NOT app-startup: the lifespan hook's await load_models() (main.py:183) only refreshes a read-only ModelReadinessSnapshot (readiness_registry.build_model_readiness_snapshot inspects sys.modules getattr + on-disk path existence; it never imports classifier.predict/load_model).

**Env conditions:** Two runtime conditions on the terminal load line, neither a default-off feature flag: (1) analysis_options.classify_parts must be True — the analyze route's options Form defaults to '{"extract_features": true, "classify_parts": true}' (analyze.py:91), so True on a bare request; the classify branch is gated at analysis_parallel_pipeline.py:50. (2) pickle.load (classifier.py:85) is only reached when the model file exists — load_model returns at :81 (records classification_model_load_total status=absent) if `not _MODEL_PATH.exists()` at :79. _MODEL_PATH = Path(CLASSIFICATION_MODEL_PATH, default models/classifier_v1.pkl). NOTE: models/classifier_v1.pkl is ABSENT in this checkout (only cad_classifier_v16_config.json present), so in this tree the load short-circuits at :81 — this is the standard model-path-present env condition, which keeps the site LIVE-but-env-conditioned (the caller chain to the load call exists in code), not LATENT. CLASSIFICATION_MODEL_VERSION only affects metric labels / settings-refresh, not reachability.

**Caller chain (top-down):**

1. Dockerfile CMD `python -m uvicorn src.main:app` (anchor, 3 occurrences) -> src/main.py:410 app.include_router(api_router, prefix="/api")
1. src/api/__init__.py:532 api_router.include_router(v1_router); analyze router imported :243 and mounted :288-296 at prefix /analyze -> live path /api/v1/analyze/
1. src/api/v1/analyze.py:88 @router.post("/") async def analyze_cad_file
1. src/api/v1/analyze.py:110 return await run_analysis_live_pipeline(..., classification_pipeline_fn=run_classification_pipeline @:138, run_parallel_pipeline_fn=run_analysis_parallel_pipeline @:124)
1. src/core/analysis_live_pipeline.py:123 await run_parallel_pipeline_fn(..., classify_pipeline=classification_pipeline_fn @:134)
1. src/core/analysis_parallel_pipeline.py:50 if analysis_options.classify_parts -> :52 async def _run_classify -> :54 cls_payload = await classify_pipeline(...); appended to parallel_tasks @:71 and executed via `stage_results = await asyncio.gather(*parallel_tasks)` @:121
1. src/core/classification/classification_pipeline.py:33 run_classification_pipeline -> :63 await build_shadow_classification_context(cls_payload, features=features, ...)
1. src/core/classification/shadow_pipeline.py:613 build_shadow_classification_context -> :622 cls_payload, ml_result = _apply_ml_overlay(payload, features=features)  [unconditional first line]
1. src/core/classification/shadow_pipeline.py:341 _apply_ml_overlay -> :349 ml_result = _predict_ml_from_features(features)  [inside try, unconditional]
1. src/core/classification/shadow_pipeline.py:320 _predict_ml_from_features -> :322 from src.ml.classifier import predict -> :325 return predict(vec_for_model)
1. src/ml/classifier.py:123 def predict -> :124 load_model()
1. src/ml/classifier.py:47 def load_model -> :85 _MODEL = pickle.load(f)  # nosec B301

**Notes:** Adversarial verify CONFIRMS the first-pass LIVE verdict; no relabel. Every hop read in source. Independently established that the module-level src.ml.classifier.predict has EXACTLY one importer/caller (shadow_pipeline.py:322) via unfiltered grep — the many `classifier.predict(...)` hits elsewhere (analyzer.py:144/189, classify_tool.py:56, batch_classify_pipeline.py:205, knowledge_distillation, hybrid_classifier.py:744/815, classifier_api.py) are instance methods on OTHER classifiers (V16/part/filename/hybrid — they take file paths/ids/entities, not a feature vector), not this module function. load_model's only caller is predict (classifier.py:124). Startup cannot reach the load: readiness snapshot is read-only. Trigger is lazy-first-request despite the 'initial load' docstring at classifier.py:48/102. Batch analyze (build_batch_router @analyze.py:155, analyze_file_fn=analyze_cad_file @:156) reuses the same handler and shares this chain. Distinct module/artifact from the PartClassifier ('part' family). Single configured .pkl (classifier_v1.pkl) -> kind single-file. Advisor's flagged inferred hop (parallel_tasks await) now read at :121.

</details>

<details><summary><code>src/ml/classifier.py::_reload_model_impl::pickle.loads#0</code> — LATENT (evidence)</summary>

**Trigger:** none live. Would-be modes: (a) per-request via the /api/v1/model/reload route — MOUNTED in src.main:app but fail-closed at an unconditional 403; (b) in-process anomaly-rollback via auto_remediation — never wired. Both are gate-before-wire, so no live caller reaches the load.

**Env conditions:** No env var can enable this from the production app in this tree (cd1b737d). The API path is fail-closed by an unconditional 403 seal (design-lock §3.2 / #509 membrane default) independent of api_key/admin_token. The auto-remediation path has no scheduler/wiring at all. model.py:86-90 notes the route would re-open only behind verify_and_load with a signed proof — not present.

**Caller chain (top-down):**

1. LOAD SITE: src/ml/classifier.py:535 obj = pickle.loads(data)  # nosec B301, inside _reload_model_impl (:259)
1. Only caller of _reload_model_impl: src/ml/classifier.py:254 within reload_model (:227) (acquires _MODEL_LOCK then delegates)
1. reload_model caller (a) — MOUNTED BUT SEALED: model router included at src/api/__init__.py:349-357 (prefix /model -> route /api/v1/model/reload); handler src/api/v1/model.py:51 model_reload is decorated status_code=403 (:48) and raises HTTPException(403) unconditionally at :78. The pre-seal body that called reload_model was removed; only a docstring reference remains at model.py:59. Route NEVER reaches the load.
1. reload_model caller (b) — LATENT (unwired): src/ml/monitoring/auto_remediation.py:301 result = reload_model(path=prev_path) inside _action_rollback_model (:294); AutoRemediation is never instantiated/invoked by any production caller (see site 3).

**Notes:** Adversarial verify CONFIRMS LATENT; no relabel. Independently grepped reload_model repo-wide: exactly two references beyond classifier.py itself — auto_remediation.py:297/301 (unwired) and model.py:59 (docstring only; the live route body raises 403 at :78). No getattr/importlib dynamic dispatch targets reload/remediation (grep empty). Label is LATENT not UNMOUNTED because the /reload route IS mounted inside the served src.main:app (api/__init__.py:349) — it is unreachable due to a runtime 403 seal, not because it lives in an unserved app. Shares its logical activation with site 3: sites 2 and 3 are two call sites of ONE reload/rollback activation (auto_remediation.reload_model -> _reload_model_impl -> pickle.loads @535); owner may prefer one id (pickle-classifier/reload) over two pins — recorded, not resolved. These are NOT the two-file open-question cases, so kind=single-file is correct.

</details>

<details><summary><code>src/ml/monitoring/auto_remediation.py::AutoRemediation._action_rollback_model::reload_model()#0</code> — LATENT (evidence)</summary>

**Trigger:** none live. Would-be mode: anomaly-driven via evaluate_and_act -> _execute_action -> _action_rollback_model, but nothing in production constructs AutoRemediation or calls evaluate_and_act.

**Env conditions:** None — there is no caller at all, so no env var, feature flag, or provider-selection var enables it. The monitoring package IS live-touched (src/ml/hybrid_classifier.py:284 imports src.ml.monitoring.prediction_monitor -> :288 PredictionMonitor(...) via the live hybrid-shadow provider; that submodule import runs monitoring/__init__.py which merely IMPORTS AutoRemediation, never instantiates it) — which is why auto_remediation is exported from a live module tree, but auto_remediation.py has no invoker.

**Caller chain (top-down):**

1. CALL SITE: src/ml/monitoring/auto_remediation.py:301 result = reload_model(path=prev_path) inside _action_rollback_model (:294) [reload_model imported from src.ml.classifier @:297]
1. _action_rollback_model registered in dispatch table self._action_handlers['rollback_model'] @:134, invoked via handler(anomaly) @:267 inside _execute_action (:252)
1. _execute_action called from evaluate_and_act @:194 (matched to REMEDIATION_RULES @:61)
1. evaluate_and_act / AutoRemediation(): NO production caller. AutoRemediation is only re-exported at src/ml/monitoring/__init__.py:51-54,97 and appears in docstrings (auto_remediation.py:116 usage example; model.py:70/83 aspirational 'emergency rollback runs in-process via auto-remediation'). No route, no lifespan/startup hook (main.py lifespan spawns only prune/faiss/faiss-age/faiss-recovery/orphan/analysis-cleanup tasks), no scheduler, no getattr/importlib dispatch constructs it.

**Notes:** Adversarial verify CONFIRMS LATENT, gate-before-wired; no relabel. Exhaustive grep across src/ for AutoRemediation / evaluate_and_act / auto_remediation / remediat: only the module itself, the __init__ re-export, and docstrings — zero production instantiation/invocation. The only `AutoRemediation()` occurrence is a docstring usage example (auto_remediation.py:116); the sole live monitoring instantiation is PredictionMonitor(...) at hybrid_classifier.py:288 (a DIFFERENT engine). model.py:70/83 'emergency rollback runs in-process via auto-remediation' is aspirational — that wiring does not exist at cd1b737d. This call site is the entry to the SAME reload/rollback logical activation as site 2 (reload_model -> _reload_model_impl -> pickle.loads @classifier.py:535). Rollback loads _MODEL_PREV_PATH (server-tracked prior path), a single .pkl -> kind single-file. Shared id pickle-classifier/reload with site 2 is an owner decision (two-sites/one-id), recorded not resolved.

</details>

### anomaly-monitor — LATENT×2

| Site | Label | Trigger | kind | activation id | conf. |
|---|---|---|---|---|---|
| `MetricsAnomalyDetector.load_models::joblib.load#0` | **LATENT** | none (unwired) | single-file | `anomaly-monitor/metrics-detector` | high |
| `MetricsAnomalyDetector.load_models::pickle.load#0` | **LATENT** | none (unwired) | single-file | `anomaly-monitor/metrics-detector` | high |

<details><summary><code>src/ml/monitoring/anomaly_detector.py::MetricsAnomalyDetector.load_models::joblib.load#0</code> — LATENT (evidence)</summary>

**Trigger:** none (unwired) — no production caller reaches load_models under any env; fires only on an explicit MetricsAnomalyDetector().load_models(path) call, of which there are ZERO in src/

**Env conditions:** Executes only in the joblib-present branch (_JOBLIB_AVAILABLE == True, anomaly_detector.py:50-55) — the normal prod branch. Moot in practice: no caller reaches load_models under any env, so the joblib branch never runs in the service.

**Caller chain (top-down):**

1. src/ml/monitoring/anomaly_detector.py:337 payload = joblib.load(src) — inside MetricsAnomalyDetector.load_models(path) (def at :329), in the `if _JOBLIB_AVAILABLE` branch (:335-337)
1. MetricsAnomalyDetector.load_models has ZERO callers in src/ (verified: grep '\.load_models(' src/ = 0 hits; the other load_models tokens are unrelated — src/models/loader.py::load_models, core/graphql/dataloader.py::batch_load_models, ml/vision_2d.py::_load_models, ml/part_classifier.py::_load_models)
1. MetricsAnomalyDetector has NO real instantiation in src/ (verified: 'MetricsAnomalyDetector(' in src/ = only anomaly_detector.py:132, which is inside the class docstring `Usage::` example; real instantiations only in tests/ and scripts/run_performance_baseline.py:168). No getattr/importlib/__import__ in src/ml/monitoring/, and no string-registry references 'MetricsAnomalyDetector' repo-wide
1. Class IS present in the live import graph: HybridClassifier.__init__ (src/ml/hybrid_classifier.py:284, lazy import at instantiation) does `from src.ml.monitoring.prediction_monitor import PredictionMonitor`, which executes src/ml/monitoring/__init__.py:46-49 and loads anomaly_detector.py. HybridClassifier is live via src/core/assistant/tools/classify_tool.py:43 and src/core/providers/classifier.py:92 (get_hybrid_classifier). BUT this is import-only: prediction_monitor.py has ZERO references to the detector/load_models, and auto_remediation.py imports only the AnomalyResult dataclass — no load call is reached
1. No src.main:app startup/lifespan/background/route reaches it: src/main.py lifespan (:145+) schedules load_models()@:183 (= src.models.loader, imported :40), background_prune_task, and faiss/vector loops — none touch anomaly; grep 'monitor|anomaly|remediat' src/main.py = only line 421 (a /metrics doc comment). No anomaly/monitoring API route exists

**Notes:** ADVERSARIAL VERIFY CONFIRMS auditor's LATENT. Gate-before-wired: the class is exported from a live package (src/ml/monitoring/__init__.py) and its module executes in the live process via the HybridClassifier->prediction_monitor import chain, but load_models() is wired to NO route, startup/lifespan hook, background scheduler, DI/factory registry, or getattr/importlib dispatch — every negative was read directly in source. Not LIVE (no proven caller to the load). Not OFFLINE (module lives in service source src/ml/monitoring/ and is in the live import graph, not a research/eval/training script). kind=single-file: load_models(path) loads ONE dump file; joblib-preferred with a pickle fallback branch — two load calls / one logical file share this proposed_logical_activation_id with the pickle.load#0 sibling (two tuples, one id; not a conflict). Correctly NOT 'open-question' (reserved for the V16Classifier and PartClassifierV16 two-distinct-file cases). Only real invocation is tests/unit/test_anomaly_detector.py. Seed 'production monitoring' hypothesis overstated. in_phase_a_denominator=false since not LIVE.

</details>

<details><summary><code>src/ml/monitoring/anomaly_detector.py::MetricsAnomalyDetector.load_models::pickle.load#0</code> — LATENT (evidence)</summary>

**Trigger:** none (unwired) — no production caller; additionally sits in the joblib-absent `else` branch a normal prod image (joblib installed) would never take even if a caller existed

**Env conditions:** Executes only in the joblib-UNAVAILABLE fallback branch (_JOBLIB_AVAILABLE == False, i.e. `import joblib` failed at anomaly_detector.py:50-55). In a standard prod image joblib is installed, so this branch is doubly unreachable: no caller AND, even with one, the else branch would not be taken. Distinct from joblib.load#0, which occupies the joblib-present branch.

**Caller chain (top-down):**

1. src/ml/monitoring/anomaly_detector.py:342 payload = pickle.load(fh) — inside MetricsAnomalyDetector.load_models(path) (def at :329), in the `else` (joblib-unavailable) branch (:338-342)
1. MetricsAnomalyDetector.load_models has ZERO callers in src/ (same verified finding as the joblib sibling; grep '\.load_models(' src/ = 0 hits)
1. MetricsAnomalyDetector has NO real instantiation in src/ (line 132 is a docstring Usage example; real instantiations only in tests/ and scripts/run_performance_baseline.py:168); no getattr/importlib/config/string-registry dispatch reaches it
1. Class is in the live import graph via HybridClassifier.__init__:284 -> src.ml.monitoring.prediction_monitor (running the monitoring package __init__ which imports anomaly_detector), but import-only — prediction_monitor never touches the detector and auto_remediation imports only AnomalyResult; no load call is reached
1. No src.main:app startup/lifespan/background/route path reaches it (same negative trace as the joblib sibling; main.py:183 load_models() is the unrelated src.models.loader)

**Notes:** ADVERSARIAL VERIFY CONFIRMS auditor's LATENT. Same enclosing method and same absent-caller analysis as joblib.load#0, independently re-read in source. LATENT (exported from a live module, no caller reaches the load), not OFFLINE (service source in the live import graph), not LIVE. Shares proposed_logical_activation_id with the joblib sibling: both are branches of one single-file load (load_models(path) loads one dump file; joblib preferred, pickle fallback) — two load calls / one logical file -> two tuples, one id (not a conflict). NOT 'open-question' KIND (reserved for V16Classifier and PartClassifierV16). Additional gate vs the sibling: also requires joblib absent (_JOBLIB_AVAILABLE == False), which prod normally is not. in_phase_a_denominator=false since not LIVE.

</details>

### part-v16 — UNMOUNTED×4

| Site | Label | Trigger | kind | activation id | conf. |
|---|---|---|---|---|---|
| `V16Classifier.load::torch.load#0` | **UNMOUNTED** | Not served by production | open-question | `part-v16/classifier-api-v6pt` | high |
| `V16Classifier.load::load_state_dict#0` | **UNMOUNTED** | Not served by production | open-question | `part-v16/classifier-api-v6pt` | high |
| `V16Classifier.load::torch.load#1` | **UNMOUNTED** | Not served by production | open-question | `part-v16/classifier-api-v14ens` | high |
| `V16Classifier.load::load_state_dict#1` | **UNMOUNTED** | Not served by production | open-question | `part-v16/classifier-api-v14ens` | high |

<details><summary><code>src/inference/classifier_api.py::V16Classifier.load::torch.load#0</code> — UNMOUNTED (evidence)</summary>

**Trigger:** Not served by production. In its own (non-production) entrypoints the load fires as: standalone-app lifespan startup (eager, classifier_api.py:729) AND lazy-first-request via POST /classify on the standalone app (:852 predict -> :653 self.load) AND CLI __main__ (:1054 classify_cli -> :1038). None of these entrypoints is served by `python -m uvicorn src.main:app`. VERIFIED: importing classifier_api via the only prod importer (health.py:484 `from src.inference.classifier_api import result_cache`) runs module-level code including :758 `classifier = V16Classifier()`, whose __init__ (:567-577) sets attrs to None with NO load; a whole-module grep confirms there is NO column-0/top-level classifier.load()/predict()/_warmup_model() statement, so the lazy import cannot fire the load.

**Env conditions:** V16Classifier.load requires TORCH_AVAILABLE (torch import guard; TORCH_AVAILABLE=torch is not None at :65; else ModuleNotFoundError :586-590) and models/cad_classifier_v6.pt present under MODEL_DIR (=<repo>/models, :64; :610-612 else FileNotFoundError). Standalone-app lifespan additionally gates on TORCH_AVAILABLE (:728). No env var selects this module/app into src.main:app; conditions are moot for production reachability because no production entrypoint serves this app/CLI. (DISABLE_V16_CLASSIFIER gates the analyzer/part_classifier path, NOT this classifier_api class.)

**Caller chain (top-down):**

1. PRODUCTION NON-REACHABLE. Verified prod entry: Dockerfile:90/109/152 CMD `python -m uvicorn src.main:app` -> src/main.py:410 app.include_router(api_router, prefix='/api'). src/api/__init__.py:271 api_router = APIRouter(); :280-530 includes exactly 27 v1 routers (drift, analyze, compare, vectors, vectors_stats, process, features, model, maintenance, health, capabilities, vision, ocr, drawing, dedup, feedback, render, active_learning, twin, materials, tolerance, standards, design_standards, assistant, cost, diff, pointcloud) + :532 api_router.include_router(v1_router) + :534 compare alias -- NONE is classifier_api. Repo-wide grep: the only src importer of classifier_api is src/api/v1/health.py:484 (lazy import of result_cache inside GET /classifier/cache) = module import -> V16Classifier.__init__ (:567, attrs=None, NO load). Startup path main.py:183 await load_models() -> src/models/readiness_registry.build_model_readiness_snapshot -> _v16_item (:229) only reads sys.modules['src.core.analyzer']._v16_classifier (part_classifier family), never imports/loads classifier_api. Provider registry (main.py:152 bootstrap_core_provider_registry) V16 adapter (src/core/providers/classifier.py:232) explicitly targets src/ml/part_classifier.py and does not load. health GET /health/v16-classifier (:865/:872/:881) uses src.core.analyzer._get_v16_classifier -> src.ml.part_classifier.PartClassifierV16 (DIFFERENT class/family).
1. NON-PROD chain A (standalone app, eager startup): src/inference/classifier_api.py:1049 __main__ else-branch -> :1058 uvicorn.run(app) [OR `python src/inference/classifier_api.py`] -> :736/:754 app=FastAPI(lifespan=lifespan) -> :726 lifespan -> :728 if TORCH_AVAILABLE -> :729 classifier.load() -> :579 V16Classifier.load -> :613 torch.load(v6_path)
1. NON-PROD chain B (standalone app, lazy-first-request): :819 @app.post('/classify') classify_file -> :852 classifier.predict(tmp_path) -> :645 predict -> :652-653 if not self.loaded: self.load() -> :579 load -> :613 torch.load(v6_path)  [also /classify/batch :890 -> :882 _predict_single -> classifier.predict]
1. NON-PROD chain C (CLI): :1049 __main__ if len(sys.argv)>1 -> :1054 classify_cli(sys.argv[1:]) -> :1036 classify_cli -> :1038 classifier.load() -> :579 load -> :613 torch.load(v6_path)

**Notes:** Loads models/cad_classifier_v6.pt (the V6 component of the V16 ensemble). OPEN-QUESTION per design-lock: V16Classifier.load is ONE logical activation that loads TWO distinct files (v6.pt via torch.load#0/load_state_dict#0 here, v14_ensemble.pt via torch.load#1/load_state_dict#1) -- owner decision pending: two-tuples/one-id vs per-file pins vs bundle KIND (id split into ...-v6pt/...-v14ens is a per-file-pins PROPOSAL, not a conclusion). ADVERSARIAL VERIFY CONFIRMED (not refuted): every load()/predict()/_warmup_model() call site is inside a def (:653/:729/:730/:852/:882/:1038/:1045); no top-level invocation exists, so health.py:484's lazy module import cannot fire the load -> UNMOUNTED, not LIVE-lazy. Distinct class from src.ml.part_classifier.PartClassifierV16 (that one IS reached by CADAnalyzer/_get_v16_classifier/provider-registry -- separate family).

</details>

<details><summary><code>src/inference/classifier_api.py::V16Classifier.load::load_state_dict#0</code> — UNMOUNTED (evidence)</summary>

**Trigger:** Not served by production. Same enclosing method as torch.load#0: fires immediately after :613 torch.load(v6_path) at :615 within V16Classifier.load, reached only via standalone-app lifespan (:729, eager), lazy-first-request /classify on the standalone app (:852 predict -> :653 self.load), or CLI (:1038). None served by `python -m uvicorn src.main:app`. VERIFIED no module-level load call fires this on import.

**Env conditions:** Same as torch.load#0: requires TORCH_AVAILABLE (:65; ModuleNotFoundError :586-590) and models/cad_classifier_v6.pt (torch.load at :613 must succeed first, else FileNotFoundError :611-612). Standalone lifespan gates on TORCH_AVAILABLE (:728). No env var wires this into src.main:app; moot for production reachability.

**Caller chain (top-down):**

1. PRODUCTION NON-REACHABLE (same as torch.load#0): Dockerfile CMD `uvicorn src.main:app` -> src/api/__init__.py api_router (27 v1 routers, none classifier_api); only src importer is health.py:484 lazy `from src.inference.classifier_api import result_cache` (module import -> __init__ :567, NO load); startup load_models -> readiness_registry._v16_item reads analyzer._v16_classifier (part_classifier family) not classifier_api; provider registry V16 adapter -> src/ml/part_classifier.py; health /v16-classifier uses part_classifier (different family). Whole-module grep: no column-0 classifier.load()/predict() statement.
1. NON-PROD chain A (standalone app, eager): :1058 uvicorn.run(app) -> :726 lifespan -> :729 classifier.load() -> :579 V16Classifier.load -> :614 self.v6_model=ImprovedClassifierV6(...) -> :615 self.v6_model.load_state_dict(v6_ckpt['model_state_dict'])
1. NON-PROD chain B (standalone app, lazy): :819 classify_file -> :852 classifier.predict -> :653 self.load() -> :579 load -> :615 v6_model.load_state_dict(...)
1. NON-PROD chain C (CLI): :1054 classify_cli -> :1038 classifier.load() -> :579 load -> :615 v6_model.load_state_dict(...)

**Notes:** Applies the v6.pt state_dict onto ImprovedClassifierV6 (:614-615); part of the same v6.pt logical file-load as torch.load#0, co-fires within V16Classifier.load. OPEN-QUESTION: one V16Classifier.load logical activation spans v6.pt + v14_ensemble.pt (owner: two-tuples/one-id vs per-file pins vs bundle KIND). ADVERSARIAL VERIFY CONFIRMED not-live: __init__ is load-free, no top-level load call, health.py imports only result_cache; the standalone FastAPI app/CLI that reach this are not served by the production `uvicorn src.main:app` command.

</details>

<details><summary><code>src/inference/classifier_api.py::V16Classifier.load::torch.load#1</code> — UNMOUNTED (evidence)</summary>

**Trigger:** Not served by production. Fires within V16Classifier.load at :627 right after the V6 block, reached only via standalone-app lifespan (:729, eager), lazy-first-request /classify on the standalone app (:852 predict -> :653 self.load), or CLI (:1038). None served by `python -m uvicorn src.main:app`. VERIFIED no module-level load call fires this on import.

**Env conditions:** Requires TORCH_AVAILABLE (:65) and models/cad_classifier_v14_ensemble.pt present under MODEL_DIR (:624-626, else FileNotFoundError). Reached only after the V6 block (:610-621) succeeds. Standalone lifespan gates on TORCH_AVAILABLE (:728). No env var wires this into src.main:app; moot for production reachability.

**Caller chain (top-down):**

1. PRODUCTION NON-REACHABLE (same as torch.load#0): Dockerfile CMD `uvicorn src.main:app` -> src/api/__init__.py api_router (27 v1 routers, none classifier_api); only src importer health.py:484 lazy-imports result_cache (module import -> __init__, no load); startup load_models -> readiness_registry reads part_classifier-family attr, not classifier_api; provider registry V16 adapter -> src/ml/part_classifier.py; health /v16-classifier uses src.ml.part_classifier.PartClassifierV16 (different family). No column-0 load call in module.
1. NON-PROD chain A (standalone app, eager): :1058 uvicorn.run(app) -> :726 lifespan -> :729 classifier.load() -> :579 V16Classifier.load -> :624-625 v14_path exists check -> :627 torch.load(v14_path)
1. NON-PROD chain B (standalone app, lazy): :819 classify_file -> :852 classifier.predict -> :653 self.load() -> :579 load -> :627 torch.load(v14_path)
1. NON-PROD chain C (CLI): :1054 classify_cli -> :1038 classifier.load() -> :579 load -> :627 torch.load(v14_path)

**Notes:** Loads models/cad_classifier_v14_ensemble.pt (the V14 ensemble component). OPEN-QUESTION: this is the SECOND file of the single V16Classifier.load logical activation (torch.load#0/load_state_dict#0 = v6.pt; torch.load#1/load_state_dict#1 = v14_ensemble.pt) -- owner decision pending: two-tuples/one-id vs per-file pins vs bundle KIND. v14_ensemble.pt itself contains multiple fold_states (:628 loop). ADVERSARIAL VERIFY CONFIRMED not-live via mounted app: no module-level load call, health.py imports only result_cache; reachable only from the standalone app/CLI production does not serve.

</details>

<details><summary><code>src/inference/classifier_api.py::V16Classifier.load::load_state_dict#1</code> — UNMOUNTED (evidence)</summary>

**Trigger:** Not served by production. Runs inside the per-fold loop of V16Classifier.load (:628-635, load_state_dict at :630), reached only via standalone-app lifespan (:729, eager), lazy-first-request /classify on the standalone app (:852 predict -> :653 self.load), or CLI (:1038). None served by `python -m uvicorn src.main:app`. VERIFIED no module-level load call fires this on import.

**Env conditions:** Requires TORCH_AVAILABLE (:65) and models/cad_classifier_v14_ensemble.pt (torch.load at :627 must succeed and yield 'fold_states'). Standalone lifespan gates on TORCH_AVAILABLE (:728). No env var wires this into src.main:app; moot for production reachability.

**Caller chain (top-down):**

1. PRODUCTION NON-REACHABLE (same as torch.load#0): Dockerfile CMD `uvicorn src.main:app` -> src/api/__init__.py api_router (27 v1 routers, none classifier_api); only src importer health.py:484 lazy-imports result_cache (module import -> __init__, no load); startup readiness/provider/analyzer paths all target part_classifier family, not classifier_api. No column-0 load call in module.
1. NON-PROD chain A (standalone app, eager): :1058 uvicorn.run(app) -> :726 lifespan -> :729 classifier.load() -> :579 V16Classifier.load -> :627 torch.load(v14_path) -> :628 for fold_state in v14_ckpt['fold_states'] -> :629 model=FusionModelV14(48,5) -> :630 model.load_state_dict(fold_state)
1. NON-PROD chain B (standalone app, lazy): :819 classify_file -> :852 classifier.predict -> :653 self.load() -> :579 load -> :628-630 per-fold model.load_state_dict(fold_state)
1. NON-PROD chain C (CLI): :1054 classify_cli -> :1038 classifier.load() -> :579 load -> :628-630 per-fold model.load_state_dict(fold_state)

**Notes:** Applies each fold state_dict from v14_ensemble.pt onto FusionModelV14 (loop over fold_states, :628-635). Part of the same v14_ensemble.pt logical file-load as torch.load#1. OPEN-QUESTION: single V16Classifier.load logical activation spans v6.pt + v14_ensemble.pt (owner: two-tuples/one-id vs per-file pins vs bundle KIND). ADVERSARIAL VERIFY CONFIRMED not-live via mounted app: no top-level load call in module, health.py imports only result_cache; the standalone app/CLI reaching this are not served by production `uvicorn src.main:app`.

</details>

### graph2d — LIVE×2

| Site | Label | Trigger | kind | activation id | conf. |
|---|---|---|---|---|---|
| `Graph2DClassifier._load_model::torch.load#0` | **LIVE** | lazy-first-request (import side-effect of a module-level singleton, not a double-checked c | single-file | `graph2d/main` | high |
| `Graph2DClassifier._load_model::load_state_dict#0` | **LIVE** | lazy-first-request (same module-level-singleton import side-effect as torch | single-file | `graph2d/main` | high |

<details><summary><code>src/ml/vision_2d.py::Graph2DClassifier._load_model::torch.load#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request (import side-effect of a module-level singleton, not a double-checked cache inside predict()); env-conditioned but fires under the SHIPPED DEFAULT config with zero env-var overrides

**Env conditions:** Fires under the SHIPPED DEFAULT config (config/hybrid_classifier.yaml:14 sets graph2d.enabled:true; HybridClassifierConfig.graph2d dataclass default is also True at src/ml/hybrid_config.py:109) with NO env vars set, provided: (1) torch is importable (HAS_TORCH=True), (2) a checkpoint file exists at GRAPH2D_MODEL_PATH (default 'models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth'), (3) the uploaded file's file_format=='dxf', (4) classify_parts option is true (default). HYBRID_CLASSIFIER_ENABLED must not be explicitly set false (default true). Note this is a WEAKER gate than the naive expectation of GRAPH2D_ENABLED=true: that var only gates the sibling direct-graph2d shadow path (shadow_pipeline.py:371, default false); it is NOT required for this hybrid fallback path, and explicitly setting GRAPH2D_ENABLED=false would also disable HybridClassifier's own _is_graph2d_enabled() check via the same env var, since _resolve_bool reads it for both. So the true minimal precondition to disable this load site is either uninstalling torch, deleting/never placing the checkpoint file, or setting GRAPH2D_ENABLED=false / HYBRID_CLASSIFIER_ENABLED=false explicitly.

**Caller chain (top-down):**

1. src/api/v1/analyze.py:87 @router.post("/") def analyze_cad_file (mounted as POST /api/v1/analyze/ — v1_router prefix /v1 + analyze router prefix /analyze, v1_router included into api_router at src/api/__init__.py:532, api_router mounted at prefix /api in src/main.py:410)
1. src/api/v1/analyze.py:110 -> return await run_analysis_live_pipeline(..., classification_pipeline_fn=run_classification_pipeline, ...)
1. src/core/analysis_live_pipeline.py:12 run_analysis_live_pipeline() -> :123-134 await run_parallel_pipeline_fn(..., classify_pipeline=classification_pipeline_fn, ...)
1. src/core/analysis_parallel_pipeline.py:20 run_analysis_parallel_pipeline() -> :54 cls_payload = await classify_pipeline(...)  [gated: analysis_options.classify_parts, default True per Form default '{"extract_features": true, "classify_parts": true}']
1. src/core/classification/classification_pipeline.py:33 run_classification_pipeline() -> :63 shadow_context = await build_shadow_classification_context(...)
1. src/core/classification/shadow_pipeline.py:613 build_shadow_classification_context() -> :629 cls_payload, hybrid_result = await _run_hybrid_shadow(..., graph2d_result=graph2d_result)  [graph2d_result is None here unless the sibling GRAPH2D_ENABLED=true path at :623 already ran]
1. src/core/classification/shadow_pipeline.py:399 _run_hybrid_shadow() gated: hybrid_enabled = os.getenv("HYBRID_CLASSIFIER_ENABLED","true")=="true" (default TRUE) and file_format=="dxf" -> :426 provider = _get_classifier_provider("hybrid") -> :427 hybrid_result = await provider.process(request, graph2d_result=graph2d_result)
1. src/core/classification/shadow_pipeline.py:328-332 _get_classifier_provider() -> bootstrap_core_provider_registry(); return ProviderRegistry.get("classifier","hybrid")
1. src/core/providers/registry.py:98-116 ProviderRegistry.get() -> provider_cls() i.e. HybridCoreProvider() (registered at src/core/providers/classifier.py:371-378, extends HybridClassifierProviderAdapter)
1. src/core/providers/classifier.py:76-85 HybridClassifierProviderAdapter.__init__ -> self._build_default_classifier() -> :88 from src.ml.hybrid_classifier import get_hybrid_classifier -> :90 get_hybrid_classifier() (src/ml/hybrid_classifier.py:1609, lazy module singleton _HYBRID_CLASSIFIER = HybridClassifier())
1. src/core/providers/base.py:91-92 BaseProvider.process() -> await self._process_impl(request, **kwargs) [trivial pass-through, no extra gating]
1. src/core/providers/classifier.py:94-101 HybridClassifierProviderAdapter._process_impl() -> result = self._wrapped_classifier.classify(filename=..., file_bytes=..., graph2d_result=kwargs.get("graph2d_result"), ...)
1. src/ml/hybrid_classifier.py:709 HybridClassifier.classify() -> :755 if graph2d_pred is None and self._is_graph2d_enabled() and file_bytes: -> :757 classifier = self.graph2d_classifier  [self._is_graph2d_enabled() at :534-536 = _resolve_bool("GRAPH2D_ENABLED", self._config.graph2d.enabled); config default AND shipped config/hybrid_classifier.yaml:14 both set graph2d.enabled: true, so this is TRUE with zero env vars set]
1. src/ml/hybrid_classifier.py:341-358 HybridClassifier.graph2d_classifier property ("懒加载 Graph2DClassifier") -> :352/356 from src.ml.vision_2d import get_ensemble_2d_classifier / get_2d_classifier -> calls it
1. This import statement is the FIRST import of src.ml.vision_2d in the process (unless already imported via the sibling GRAPH2D_ENABLED=true direct path at shadow_pipeline.py:380/classifier.py:174, which reaches the identical module-level line) -> module body executes -> src/ml/vision_2d.py:266 _graph2d = Graph2DClassifier()
1. src/ml/vision_2d.py:38-64 Graph2DClassifier.__init__ -> :63 if HAS_TORCH and os.path.exists(self.model_path): -> :64 self._load_model()
1. src/ml/vision_2d.py:133-136 Graph2DClassifier._load_model() -> :136 checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)  <-- TARGET SITE

**Notes:** Load mechanism is a MODULE-LEVEL SINGLETON (src/ml/vision_2d.py:266 `_graph2d = Graph2DClassifier()`), not a double-checked cache inside predict(). It fires once, at the first import of src.ml.vision_2d in the process, from whichever caller imports it first at runtime (two live callers exist: the direct graph2d shadow path at shadow_pipeline.py:380/classifier.py:174, gated by GRAPH2D_ENABLED=true; and the HybridClassifier fallback path traced above, which is live under shipped defaults). Both callers are lazy-first-request (no app-startup or lifespan trigger reaches this import — verified: src/main.py lifespan calls load_models() -> src/models/readiness_registry.py's _graph2d_item() which uses _module_object() = sys.modules.get() only, i.e. a passive read of already-imported state, not an import; it never imports src.ml.vision_2d itself, so it does not trigger the load). Once the singleton is constructed (successfully or not), _load_model() and thus both target lines run at most once per process for the primary classifier instance; EnsembleGraph2DClassifier (src/ml/vision_2d.py:273-324, reached via get_ensemble_2d_classifier when GRAPH2D_ENSEMBLE_ENABLED=true) constructs additional Graph2DClassifier instances per configured checkpoint path and reaches the same two lines per instance. Not an 'open-question' two-file case per the seed taxonomy (that's V16Classifier/PartClassifierV16 only) — Graph2DClassifier loads a single checkpoint file, so kind=single-file.

</details>

<details><summary><code>src/ml/vision_2d.py::Graph2DClassifier._load_model::load_state_dict#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request (same module-level-singleton import side-effect as torch.load#0, same enclosing _load_model call)

**Env conditions:** Same base preconditions as torch.load#0 (torch importable, checkpoint file present at GRAPH2D_MODEL_PATH, dxf file_format, classify_parts default-true, HYBRID_CLASSIFIER_ENABLED not explicitly false, HybridClassifier's graph2d.enabled resolves true which it does by default). ADDITIONALLY gated on checkpoint content: this line is reached only when the loaded .pth file's dict does NOT have checkpoint.get("arch")=="GraphEncoderV2" (i.e. a legacy-format checkpoint, model_type in {'edge_sage','gcn'} etc.) — a data-shape condition on the checkpoint file itself, not an env var. Whether the actual default checkpoint on disk exercises this line vs. the GraphEncoderV2WithHead.from_checkpoint() branch (line 143) depends on that file's contents, which was not inspected (out of scope: this is a source-code reachability audit, not a runtime data probe) — the code path unconditionally exists and is reachable for any legacy-format checkpoint placed at that path.

**Caller chain (top-down):**

1. Identical chain to torch.load#0 above, through src/ml/vision_2d.py:133-163 Graph2DClassifier._load_model()
1. src/ml/vision_2d.py:136 checkpoint = torch.load(...) executes first and returns the checkpoint dict
1. src/ml/vision_2d.py:142-151 if checkpoint.get("arch")=="GraphEncoderV2": takes the GraphEncoderV2WithHead.from_checkpoint() branch and returns early (load_state_dict NOT reached on this branch)
1. src/ml/vision_2d.py:153-163 else (legacy checkpoint, arch key absent/different): builds EdgeGraphSageClassifier or SimpleGraphClassifier per checkpoint['model_type'], then :163 self.model.load_state_dict(checkpoint["model_state_dict"])  <-- TARGET SITE

**Notes:** Shares proposed_logical_activation_id with torch.load#0 since both are inside the same _load_model() call for the same logical model family (Graph2D single-checkpoint classifier) — this is a mutually-exclusive branch of the same load call (checkpoint['arch']=='GraphEncoderV2' goes through GraphEncoderV2WithHead.from_checkpoint() at line 143 instead), not a second distinct artifact file, so this does NOT qualify for the two-distinct-files-one-logical-activation 'open-question' treatment (that's reserved for V16Classifier v6.pt+v14.pt and PartClassifierV16's two-file _load_models). Confidence is 'high' on reachability of the source line itself; the only inference-flavored element is which checkpoint format is actually deployed at the default path, which is immaterial to the LIVE/reachability determination since the code branch is unconditionally live for that checkpoint shape.

</details>

### history — LIVE×2

| Site | Label | Trigger | kind | activation id | conf. |
|---|---|---|---|---|---|
| `HistorySequenceClassifier._load_model::load_state_dict#0` | **LIVE** | lazy-first-request (construction side-effect on first qualifying DXF POST /api/v1/analyze/ | single-file | `history/sequence` | high |
| `HistorySequenceClassifier._load_model::torch.load#0` | **LIVE** | lazy-first-request (construction side-effect on first qualifying DXF POST /api/v1/analyze/ | single-file | `history/sequence` | high |

<details><summary><code>src/ml/history_sequence_classifier.py::HistorySequenceClassifier._load_model::load_state_dict#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request (construction side-effect on first qualifying DXF POST /api/v1/analyze/ request; the HybridClassifier is cached in the module-level get_hybrid_classifier() singleton and the provider in the ProviderRegistry singleton thereafter, so _load_model runs once on first construction)

**Env conditions:** Full gate stack, all independently required: (1) HYBRID_CLASSIFIER_ENABLED != "false" (default "true") AND request file_format=="dxf" [shadow_pipeline.py:410-411]; (2) history_file_path must resolve non-empty via _resolve_history_sequence_file_path [shadow_pipeline.py:236-317] from one of: analysis_options.history_file_path (:266-271), env HISTORY_SEQUENCE_FILE_PATH -> existing .h5 (:273-278), or env HISTORY_SEQUENCE_SIDECAR_DIR + matching sidecar .h5 (:283-316) -- and if env HISTORY_SEQUENCE_ALLOWED_ROOT is set, ALL three sources are path-containment-checked against it inside _resolve_existing_h5 (:259-263), not just the options branch; (3) _should_attempt_history [:581-586] needs the resolved path AND (HISTORY_SEQUENCE_ENABLED=true [hybrid_config history_sequence.enabled default False] OR auto_enable_history [env HISTORY_SEQUENCE_AUTO_ENABLE / config auto_enable.history_on_path, default True]) -- net under defaults: a resolved history_file_path suffices; (4) inside _load_model: HAS_TORCH true (torch + src.ml.train.sequence_encoder.SequenceCommandClassifier importable, real prod dep) AND self.model_path non-empty (constructor default ""; requires env HISTORY_SEQUENCE_MODEL_PATH or hybrid_config history_sequence.model_path, config default "") AND that path exists on disk. All failures inside _load_model are swallowed (try/except -> warning, self.model=None) -- silent no-op, not a crash.

**Caller chain (top-down):**

1. src/main.py:410 app.include_router(api_router, prefix="/api") [prod CMD: python -m uvicorn src.main:app, Dockerfile:90/109/152]
1. src/api/__init__.py:532 api_router.include_router(v1_router) [v1_router = APIRouter(prefix="/v1") at :274]
1. src/api/__init__.py:288-296 _include_router(v1_router, module=analyze, prefix="/analyze") -> :202-206 v1_router.include_router(analyze.router, prefix="/analyze"); analyze = _import_router("analyze","src.api.v1.analyze") at :243 -> effective route POST /api/v1/analyze/
1. src/api/v1/analyze.py:88 @router.post("/", response_model=AnalysisResult) async def analyze_cad_file(...)
1. src/api/v1/analyze.py:110-138 -> run_analysis_live_pipeline(..., classification_pipeline_fn=run_classification_pipeline) [run_classification_pipeline imported from src.core.classification at :57-60]
1. src/core/analysis_live_pipeline.py:122-134 run_parallel_pipeline_fn(..., classify_pipeline=classification_pipeline_fn)
1. src/core/analysis_parallel_pipeline.py:50 if analysis_options.classify_parts (options default {"classify_parts": true}): :54 cls_payload = await classify_pipeline(...) [== run_classification_pipeline]
1. src/core/classification/classification_pipeline.py:63 shadow_context = await build_shadow_classification_context(...) [imported from shadow_pipeline at :25]
1. src/core/classification/shadow_pipeline.py:629 build_shadow_classification_context calls await _run_hybrid_shadow(...) unconditionally
1. src/core/classification/shadow_pipeline.py:410-412 _run_hybrid_shadow gate: HYBRID_CLASSIFIER_ENABLED (default "true") AND file_format=="dxf"; :415 _resolve_history_sequence_file_path(...); :426 provider = _get_classifier_provider("hybrid"); :427-434 hybrid_result = await provider.process(ClassifierRequest(..., history_file_path=history_file_path), graph2d_result=...)
1. src/core/classification/shadow_pipeline.py:328-332 _get_classifier_provider -> bootstrap_core_provider_registry() [src/core/providers/bootstrap.py:322 -> bootstrap_core_classifier_providers()] which registers classifier/hybrid = HybridCoreProvider(HybridClassifierProviderAdapter) at src/core/providers/classifier.py:368-378; then ProviderRegistry.get("classifier","hybrid") -> src/core/providers/registry.py:104-112 instantiates provider_cls() (cached singleton)
1. src/core/providers/base.py:91-92 BaseProvider.process -> unconditionally: return await self._process_impl(request, **kwargs)
1. src/core/providers/classifier.py:96-110 HybridClassifierProviderAdapter._process_impl -> self._wrapped_classifier.classify(..., history_file_path=request.history_file_path); _wrapped_classifier = get_hybrid_classifier() singleton (classifier.py:86-94)
1. src/ml/hybrid_classifier.py:710 HybridClassifier.classify(...) -> :982 if self._should_attempt_history(history_file_path) [:581-586: needs truthy path AND (HISTORY_SEQUENCE_ENABLED or auto_enable_history)]
1. src/ml/hybrid_classifier.py:986 classifier = self.history_sequence_classifier (lazy @property :390-424; first access constructs HistorySequenceClassifier(prototypes_path=..., model_path=...) at :415)
1. src/ml/history_sequence_classifier.py:82 HistorySequenceClassifier.__init__ -> self._load_model()
1. src/ml/history_sequence_classifier.py:155-215 _load_model(): guard :156 (if not HAS_TORCH or not self.model_path: return) and :159 (if not model_path.exists(): return); :162 checkpoint = torch.load(str(model_path), map_location="cpu"); :193-202 self.model = SequenceCommandClassifier(...); :203 self.model.load_state_dict(checkpoint["model_state_dict"])

**Notes:** Independently re-walked hop-by-hop at cd1b737d; label CONFIRMED LIVE. The two hops a prior self-check left inferred are now read directly: ProviderRegistry.get instantiates the registered provider_cls (registry.py:104-112, cached singleton) and BaseProvider.process dispatches unconditionally to _process_impl (base.py:91-92) -- so confidence=high is honest. torch.load#0 (line 162) and load_state_dict#0 (line 203) are two sequential statements in the SAME _load_model try-block loading the SAME self.model_path checkpoint under the identical gate stack and caller chain; load_state_dict is strictly dominated by torch.load succeeding. Single logical activation 'history/sequence' (kind=single-file, one checkpoint artifact) -- NOT one of the task's named open-question two-distinct-files cases (V16Classifier v6.pt+v14.pt; PartClassifierV16._load_models two files); history_sequence_classifier is not in that list. Corrected auditor caller_chain errors: torch.load is :162 (auditor cited ~172); __init__->_load_model is :82 (auditor cited :171); the _load_model guard is two early-returns at :156 and :159 (auditor cited one combined 'if HAS_TORCH and self.model_path and model_path.exists()'); property span :390-424 (auditor :391). LIVE-but-env-conditioned per the ratified rule (chain reaches the load call; execution of torch.load/load_state_dict additionally requires HISTORY_SEQUENCE_MODEL_PATH set to an existing checkpoint).

</details>

<details><summary><code>src/ml/history_sequence_classifier.py::HistorySequenceClassifier._load_model::torch.load#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request (construction side-effect on first qualifying DXF POST /api/v1/analyze/ request; the HybridClassifier is cached in the module-level get_hybrid_classifier() singleton and the provider in the ProviderRegistry singleton thereafter, so _load_model runs once on first construction)

**Env conditions:** Identical gate stack to load_state_dict#0 (same _load_model try-block, same caller chain): (1) HYBRID_CLASSIFIER_ENABLED != "false" (default "true") AND file_format=="dxf" [shadow_pipeline.py:410-411]; (2) history_file_path resolves via _resolve_history_sequence_file_path [236-317]: analysis_options.history_file_path (:266-271), env HISTORY_SEQUENCE_FILE_PATH (:273-278), or HISTORY_SEQUENCE_SIDECAR_DIR sidecar match (:283-316) -- all three containment-checked against HISTORY_SEQUENCE_ALLOWED_ROOT via _resolve_existing_h5 (:259-263) when that env var is set, not just the options branch; (3) _should_attempt_history [:581-586]: resolved path AND (HISTORY_SEQUENCE_ENABLED [config default False] OR auto_enable_history [HISTORY_SEQUENCE_AUTO_ENABLE / auto_enable.history_on_path default True]); (4) HAS_TORCH true AND non-empty self.model_path (default "", requires env HISTORY_SEQUENCE_MODEL_PATH or hybrid_config history_sequence.model_path) AND file exists on disk. torch.load exceptions are caught and logged as a warning (self.model=None), not propagated.

**Caller chain (top-down):**

1. src/main.py:410 app.include_router(api_router, prefix="/api") [prod CMD: python -m uvicorn src.main:app, Dockerfile:90/109/152]
1. src/api/__init__.py:532 api_router.include_router(v1_router) [v1_router = APIRouter(prefix="/v1") at :274]
1. src/api/__init__.py:288-296 _include_router(v1_router, module=analyze, prefix="/analyze") -> :202-206 v1_router.include_router(analyze.router, prefix="/analyze"); analyze = _import_router("analyze","src.api.v1.analyze") at :243 -> effective route POST /api/v1/analyze/
1. src/api/v1/analyze.py:88 @router.post("/", response_model=AnalysisResult) async def analyze_cad_file(...)
1. src/api/v1/analyze.py:110-138 -> run_analysis_live_pipeline(..., classification_pipeline_fn=run_classification_pipeline) [run_classification_pipeline imported from src.core.classification at :57-60]
1. src/core/analysis_live_pipeline.py:122-134 run_parallel_pipeline_fn(..., classify_pipeline=classification_pipeline_fn)
1. src/core/analysis_parallel_pipeline.py:50 if analysis_options.classify_parts (options default {"classify_parts": true}): :54 cls_payload = await classify_pipeline(...) [== run_classification_pipeline]
1. src/core/classification/classification_pipeline.py:63 shadow_context = await build_shadow_classification_context(...) [imported from shadow_pipeline at :25]
1. src/core/classification/shadow_pipeline.py:629 build_shadow_classification_context calls await _run_hybrid_shadow(...) unconditionally
1. src/core/classification/shadow_pipeline.py:410-412 _run_hybrid_shadow gate: HYBRID_CLASSIFIER_ENABLED (default "true") AND file_format=="dxf"; :415 _resolve_history_sequence_file_path(...); :426 provider = _get_classifier_provider("hybrid"); :427-434 hybrid_result = await provider.process(ClassifierRequest(..., history_file_path=history_file_path), graph2d_result=...)
1. src/core/classification/shadow_pipeline.py:328-332 _get_classifier_provider -> bootstrap_core_provider_registry() [src/core/providers/bootstrap.py:322 -> bootstrap_core_classifier_providers()] which registers classifier/hybrid = HybridCoreProvider(HybridClassifierProviderAdapter) at src/core/providers/classifier.py:368-378; then ProviderRegistry.get("classifier","hybrid") -> src/core/providers/registry.py:104-112 instantiates provider_cls() (cached singleton)
1. src/core/providers/base.py:91-92 BaseProvider.process -> unconditionally: return await self._process_impl(request, **kwargs)
1. src/core/providers/classifier.py:96-110 HybridClassifierProviderAdapter._process_impl -> self._wrapped_classifier.classify(..., history_file_path=request.history_file_path); _wrapped_classifier = get_hybrid_classifier() singleton (classifier.py:86-94)
1. src/ml/hybrid_classifier.py:710 HybridClassifier.classify(...) -> :982 if self._should_attempt_history(history_file_path) [:581-586: needs truthy path AND (HISTORY_SEQUENCE_ENABLED or auto_enable_history)]
1. src/ml/hybrid_classifier.py:986 classifier = self.history_sequence_classifier (lazy @property :390-424; first access constructs HistorySequenceClassifier(prototypes_path=..., model_path=...) at :415)
1. src/ml/history_sequence_classifier.py:82 HistorySequenceClassifier.__init__ -> self._load_model()
1. src/ml/history_sequence_classifier.py:155-162 _load_model(): guard :156 (if not HAS_TORCH or not self.model_path: return) and :159 (if not model_path.exists(): return); :162 checkpoint = torch.load(str(model_path), map_location="cpu")

**Notes:** Independently re-walked hop-by-hop at cd1b737d; label CONFIRMED LIVE. Same single logical activation as load_state_dict#0 -- torch.load (line 162) is a strict precondition for load_state_dict (line 203) in the same _load_model try-block, loading the same self.model_path checkpoint under the identical gate stack. kind=single-file (one checkpoint artifact); NOT an open-question two-distinct-files case (task names only V16Classifier and PartClassifierV16 for that). Registry/dispatch hops now read directly (registry.py:104-112 instantiates; base.py:91-92 dispatches to _process_impl), supporting confidence=high. Corrected auditor line number: torch.load is :162 (auditor cited ~172); __init__->_load_model is :82 (auditor cited :171); guard is two early-returns at :156/:159 (not one combined if). LIVE-but-env-conditioned: the chain reaches this load call; whether torch.load actually executes further requires HISTORY_SEQUENCE_MODEL_PATH pointing at an existing checkpoint.

</details>

**Verifier corrections (5):**
- `src/ml/history_sequence_classifier.py::HistorySequenceClassifier._load_model::torch.load#0` — Corrected the terminal load line: torch.load is at :162 (auditor wrote 'line ~172'). Line 172 is actually a `raise ValueError` inside the try-block, not the load call.
- `both sites (load_state_dict#0 and torch.load#0)` — Corrected the __init__ -> _load_model hop: the call is at :82 (auditor wrote ':171'). Line 171 is inside _load_model's label_map loop. __init__ invokes self._load_model() at :82.
- `both sites (load_state_dict#0 and torch.load#0)` — Corrected the _load_model guard structure: it is TWO separate early-returns -- :156 `if not HAS_TORCH or not self.model_path: return` and :159 `if not model_path.exists(): return` -- not a single combined `if HAS_TORCH and self.model_path and model_path.exists():`.
- `both sites (load_state_dict#0 and torch.load#0)` — Broadened env_conditions: HISTORY_SEQUENCE_ALLOWED_ROOT path-containment is enforced by _resolve_existing_h5 (shadow_pipeline.py:259-263) for ALL THREE history-path sources (options, HISTORY_SEQUENCE_FILE_PATH env, sidecar), not only the options branch as the auditor scoped it.
- `both sites (load_state_dict#0 and torch.load#0)` — Added the two hops the auditor's self-check left inferred, now read in source: ProviderRegistry.get instantiates the registered provider_cls (registry.py:104-112, cached singleton) and BaseProvider.process dispatches unconditionally to _process_impl (base.py:91-92). No label change.

### hybrid — LATENT×4

| Site | Label | Trigger | kind | activation id | conf. |
|---|---|---|---|---|---|
| `HybridClassifier.stat_mlp::load_state_dict#0` | **LATENT** | Would-be per-request lazy-first-predict: property `stat_mlp` is accessed inside `classify( | single-file | `hybrid/stat_mlp` | high |
| `HybridClassifier.stat_mlp::torch.load#0` | **LATENT** | Same property, same blocked path as HybridClassifier | single-file | `hybrid/stat_mlp` | high |
| `HybridClassifier.tfidf_text_classifier::load_state_dict#0` | **LATENT** | Would-be per-request lazy-first-predict: property `tfidf_text_classifier` is accessed insi | single-file | `hybrid/tfidf_text` | high |
| `HybridClassifier.tfidf_text_classifier::torch.load#0` | **LATENT** | Same property, same blocked path as HybridClassifier | single-file | `hybrid/tfidf_text` | high |

<details><summary><code>src/ml/hybrid_classifier.py::HybridClassifier.stat_mlp::load_state_dict#0</code> — LATENT (evidence)</summary>

**Trigger:** Would-be per-request lazy-first-predict: property `stat_mlp` is accessed inside `classify()` at hybrid_classifier.py:946, gated by `self.stat_mlp_enabled and graph2d_pred and graph2d_pred.get('status')=='ok'` (:944), on the first analyze request whose Graph2D prediction returns status=='ok'. NOT app-startup (main.py lifespan at :146/:391 never instantiates HybridClassifier nor calls classify()). The load is never reached in production: the property's first try-block import `from scripts.train_stat_mlp import ...` (:445) raises ModuleNotFoundError because `scripts/` is not shipped in any serving image; the bare `except Exception` at :462 swallows it before `torch.load` (:448)/`load_state_dict` (:453) execute.

**Env conditions:** Gate 1 (blocking, deterministic, NOT runtime-flippable): `from scripts.train_stat_mlp import StatMLP, extract_stat_features, STAT_FEAT_DIM` (:445) raises ModuleNotFoundError because `scripts/` is never copied into ANY serving image and is not importable at runtime. Verified airtight across all four Dockerfiles: root Dockerfile production stage COPYs only src/,config/,data/ (:71-73), development `FROM production` (:95) adds only tests/, gpu-base COPYs only src/,config/,data/ (:140-142) — all three serve `src.main:app`; docker/assistant/Dockerfile COPYs only src/,configs/ and serves a different app (src.core.assistant.server:app); deployments/docker/Dockerfile COPYs only src/,config/,models/,knowledge_base/ and runs `python -m src.main`; Dockerfile.nginx is a proxy. Repo-wide grep for `COPY *scripts` / `COPY . .` / `ADD .`: zero hits. PYTHONPATH=/app (Dockerfile:47/144), no setup.py/pyproject.toml/setup.cfg (no editable install), no site-package named scripts, and no docker-compose (only ./data,./models) or k8s (only config,models) volume mounts scripts/. Gate 2 (secondary, would also block independently): __init__ self.stat_mlp_enabled (:174-177) defaults to os.path.exists(os.getenv("STAT_MLP_MODEL_PATH","models/stat_mlp_24class.pth")); this checkout's models/ has no stat_mlp_24class.pth (verified absent) and no default env sets STAT_MLP_ENABLED, so the flag defaults False and the property getter body is never entered. Gate 3 (cross-family precondition, moot given Gate 1): requires a prior Graph2D prediction with status=='ok'.

**Caller chain (top-down):**

1. src/main.py:410 app.include_router(api_router, prefix="/api") [ASGI mount, per verified anchor]
1. src/api/__init__.py analyze = _import_router("analyze", "src.api.v1.analyze") (:243); v1_router.include_router of analyze module under prefix "/analyze" (:288-294); v1_router (prefix "/v1") folded into api_router
1. src/api/v1/analyze.py @router.post("/") async def analyze_cad_file(...) -> run_analysis_live_pipeline(..., classification_pipeline_fn=run_classification_pipeline)
1. src/core/analysis_live_pipeline.py -> run_parallel_pipeline_fn(..., classify_pipeline=classification_pipeline_fn)
1. src/core/analysis_parallel_pipeline.py:54 cls_payload = await classify_pipeline(..., content=content)  # content = real uploaded file bytes
1. src/core/classification/classification_pipeline.py:63 build_shadow_classification_context(cls_payload, ..., content=content)
1. src/core/classification/shadow_pipeline.py:629 _run_hybrid_shadow(...) -> :426-434 provider = _get_classifier_provider("hybrid"); await provider.process(ClassifierRequest(file_bytes=content, ...), graph2d_result=graph2d_result)
1. src/core/providers/base.py process -> _process_impl
1. src/core/providers/classifier.py:104 HybridClassifierProviderAdapter._process_impl: self._wrapped_classifier.classify(filename=..., file_bytes=request.file_bytes, graph2d_result=kwargs.get("graph2d_result"), ...)  # real file_bytes flow
1. src/ml/hybrid_classifier.py:944-946 classify(): if self.stat_mlp_enabled and graph2d_pred and graph2d_pred.get("status")=="ok": stat_clf = self.stat_mlp  [property getter reached]
1. src/ml/hybrid_classifier.py:440-465 @property stat_mlp: :445 from scripts.train_stat_mlp import ... -> ModuleNotFoundError in every serving image (scripts/ absent) -> caught by :462 except Exception -> self._stat_mlp=None; :448 torch.load and :453 load_state_dict NEVER EXECUTE

**Notes:** Independently verified; auditor's LATENT stands. Seed hypothesis ("reached via classify tools") is wrong as literally stated: ClassifyTool.execute (src/core/assistant/tools/classify_tool.py:44) calls classifier.classify(file_id) positionally -> file_id binds to `filename`, file_bytes=None -> graph2d_pred stays None -> the :944 stat_mlp gate can never fire via that tool. The real near-miss caller is the HTTP analyze shadow-classification pipeline (above), which carries real file_bytes end-to-end on every DXF analyze request (default classify_parts=true, HYBRID_CLASSIFIER_ENABLED default true). LATENT (not OFFLINE): a live route genuinely reaches the property getter; only the load LINE inside is blocked by the failing import. Not LIVE-env-conditioned: scripts/ absence is an image-packaging fact, not a runtime knob — flipping STAT_MLP_ENABLED + mounting a checkpoint would NOT make the load fire without an image rebuild that adds scripts/. If a future build adds COPY scripts/ ./scripts/ (or the module is vendored into src/) AND a checkpoint ships to models/ AND STAT_MLP_ENABLED is set, re-audit — it would plausibly flip to LIVE env-conditioned. kind=single-file (loads one artifact models/stat_mlp_24class.pth); NOT one of the two-file open-question cases (those are V16Classifier / PartClassifierV16, a different family).

</details>

<details><summary><code>src/ml/hybrid_classifier.py::HybridClassifier.stat_mlp::torch.load#0</code> — LATENT (evidence)</summary>

**Trigger:** Same property, same blocked path as HybridClassifier.stat_mlp::load_state_dict#0. Would-be per-request lazy-first-predict via classify():946; torch.load at :448 sits one statement above load_state_dict at :453 in the same try block, both gated by the :445 scripts import that raises in every serving image.

**Env conditions:** Identical to HybridClassifier.stat_mlp::load_state_dict#0 (same try block, same blocking import at :445, same secondary model-file-absence gate at :174-177, same cross-family Graph2D status=='ok' precondition).

**Caller chain (top-down):**

1. Identical chain to HybridClassifier.stat_mlp::load_state_dict#0 (same property getter, hybrid_classifier.py:440-465). torch.load at :448 executes before load_state_dict at :453 within the same try block; both are preceded by the :445 `from scripts.train_stat_mlp import ...` which raises ModuleNotFoundError in production, so neither line runs.

**Notes:** Same underlying activation as load_state_dict#0, tokenized as two audited call-sites per the manifest. In production the getter reaches NEITHER torch.load (:448) nor load_state_dict (:453) — the :445 import raises first. (Correcting a minor imprecision in the first-pass phrasing: a SUCCESSFUL getter run reaches BOTH lines sequentially — torch.load then load_state_dict — not 'at most one'; the label is unaffected since production reaches neither.) See load_state_dict#0 notes for the full Gate-1/2/3 reasoning and the seed-hypothesis correction.

</details>

<details><summary><code>src/ml/hybrid_classifier.py::HybridClassifier.tfidf_text_classifier::load_state_dict#0</code> — LATENT (evidence)</summary>

**Trigger:** Would-be per-request lazy-first-predict: property `tfidf_text_classifier` is accessed inside `classify()` at hybrid_classifier.py:912 as a keyword-miss fallback, gated by `text_content_pred is None and self.tfidf_text_enabled and _dxf_text and len(_dxf_text.strip())>=4` (:905-910), on the first analyze request where the primary keyword TextContentClassifier finds no match but extracted DXF text (len>=4) is present. NOT app-startup. The load is never reached in production: the property's first try-block import `from scripts.train_text_classifier_ml import ...` (:473) raises ModuleNotFoundError because `scripts/` is not shipped in any serving image; the bare `except Exception` at :492 swallows it before `torch.load` (:476)/`load_state_dict` (:481) execute.

**Env conditions:** Gate 1 (blocking, deterministic, NOT runtime-flippable): `from scripts.train_text_classifier_ml import TextMLP, SimpleVectorizer` (:473) raises ModuleNotFoundError for the same packaging reason as the stat_mlp sites — scripts/ is absent from all four Dockerfiles (root prod/dev/gpu serving src.main:app COPY only src/config/data[/tests]; docker/assistant serves src.core.assistant.server:app with only src/,configs/; deployments/docker/Dockerfile runs python -m src.main with only src/,config/,models/,knowledge_base/), no COPY scripts / COPY . . anywhere, PYTHONPATH=/app, no editable install, no compose/k8s scripts mount. Gate 2 (secondary, would also block independently): __init__ self.tfidf_text_enabled (:178-181) defaults to os.path.exists(os.getenv("TFIDF_TEXT_MODEL_PATH","models/text_classifier_tfidf.pth")); this checkout's models/ has no text_classifier_tfidf.pth (verified absent) and no default env sets TFIDF_TEXT_ENABLED, so the flag defaults False and the getter body is never entered. Gate 3 (data-dependent, still needed even if Gate 1/2 fixed): primary TextContentClassifier keyword match must fail (text_content_pred is None) and extracted DXF text length>=4 — a normal per-document fallback condition, not a hard block.

**Caller chain (top-down):**

1. src/main.py:410 app.include_router(api_router, prefix="/api") [ASGI mount, per verified anchor]
1. src/api/__init__.py analyze router imported (:243) and included under v1_router prefix "/analyze" (:288-294); v1_router (prefix "/v1") folded into api_router
1. src/api/v1/analyze.py @router.post("/") async def analyze_cad_file(...) -> run_analysis_live_pipeline(..., classification_pipeline_fn=run_classification_pipeline)
1. src/core/analysis_live_pipeline.py -> run_parallel_pipeline_fn(..., classify_pipeline=classification_pipeline_fn)
1. src/core/analysis_parallel_pipeline.py:54 cls_payload = await classify_pipeline(..., content=content)  # content = real uploaded file bytes
1. src/core/classification/classification_pipeline.py:63 build_shadow_classification_context(cls_payload, ..., content=content)
1. src/core/classification/shadow_pipeline.py:629 _run_hybrid_shadow(...) -> :426-434 provider = _get_classifier_provider("hybrid"); await provider.process(ClassifierRequest(file_bytes=content, ...))
1. src/core/providers/base.py process -> _process_impl
1. src/core/providers/classifier.py:104 HybridClassifierProviderAdapter._process_impl: self._wrapped_classifier.classify(filename=..., file_bytes=request.file_bytes, ...)  # real file_bytes flow; TEXT_CONTENT_ENABLED default true so _dxf_text extraction runs at :870-878
1. src/ml/hybrid_classifier.py:905-912 classify(): if text_content_pred is None and self.tfidf_text_enabled and _dxf_text and len(_dxf_text.strip())>=4: tfidf_clf = self.tfidf_text_classifier  [property getter reached]
1. src/ml/hybrid_classifier.py:467-495 @property tfidf_text_classifier: :473 from scripts.train_text_classifier_ml import ... -> ModuleNotFoundError in every serving image (scripts/ absent) -> caught by :492 except Exception -> self._tfidf_text_clf=None; :476 torch.load and :481 load_state_dict NEVER EXECUTE

**Notes:** Independently verified; auditor's LATENT stands. Seed hypothesis ("reached via classify tools") is wrong as literally stated: ClassifyTool.execute (classify_tool.py:44) calls classify(file_id) positionally -> file_bytes=None -> the :870 text-extraction block (needs file_bytes) never runs -> _dxf_text stays "" -> the :905 tfidf gate can never fire via that tool. The real near-miss caller is the HTTP analyze shadow-classification pipeline (above); file_bytes flows end-to-end and TEXT_CONTENT_ENABLED defaults true so _dxf_text extraction runs. LATENT (not OFFLINE): a live route reaches the getter; only the load LINE inside is blocked by the failing import. Not LIVE-env-conditioned: scripts/ absence needs an image rebuild, not a config flag. If a future build adds COPY scripts/ ./scripts/ (or vendors the module into src/) AND a checkpoint ships to models/ AND TFIDF_TEXT_ENABLED is set, re-audit -> plausibly flips to LIVE env/data-conditioned. kind=single-file (loads one artifact models/text_classifier_tfidf.pth); NOT one of the two-file open-question cases.

</details>

<details><summary><code>src/ml/hybrid_classifier.py::HybridClassifier.tfidf_text_classifier::torch.load#0</code> — LATENT (evidence)</summary>

**Trigger:** Same property, same blocked path as HybridClassifier.tfidf_text_classifier::load_state_dict#0. Would-be per-request lazy-first-predict via classify():912; torch.load at :476 sits one statement above load_state_dict at :481 in the same try block, both gated by the :473 scripts import that raises in every serving image.

**Env conditions:** Identical to HybridClassifier.tfidf_text_classifier::load_state_dict#0 (same try block, same blocking import at :473, same secondary model-file-absence gate at :178-181, same data-dependent keyword-miss precondition).

**Caller chain (top-down):**

1. Identical chain to HybridClassifier.tfidf_text_classifier::load_state_dict#0 (same property getter, hybrid_classifier.py:467-495). torch.load at :476 executes before load_state_dict at :481 within the same try block; both are preceded by the :473 `from scripts.train_text_classifier_ml import ...` which raises ModuleNotFoundError in production, so neither line runs.

**Notes:** Same underlying activation as load_state_dict#0, tokenized as two audited call-sites per the manifest. In production the getter reaches NEITHER torch.load (:476) nor load_state_dict (:481) — the :473 import raises first. (Correcting a minor imprecision in the first-pass phrasing: a SUCCESSFUL getter run reaches BOTH lines sequentially — torch.load then load_state_dict — not 'at most one'; label unaffected since production reaches neither.) See tfidf_text_classifier::load_state_dict#0 notes for the full Gate-1/2/3 reasoning and the seed-hypothesis correction.

</details>

**Verifier corrections (2):**
- `ALL FOUR (hybrid/stat_mlp x2, hybrid/tfidf_text x2)` — No label change — all four confirmed LATENT / in_phase_a_denominator=false / confidence high. Strengthened the Gate-1 evidence beyond the first pass: independently confirmed scripts/ is excluded from ALL FOUR Dockerfiles in the repo (not just the root one the auditor cited). Root prod/dev/gpu stages
- `src/ml/hybrid_classifier.py::HybridClassifier.stat_mlp::torch.load#0 and HybridClassifier.tfidf_text_classifier::torch.load#0` — Corrected the first-pass note phrasing 'one property-getter execution reaches at most one of these lines'. A SUCCESSFUL getter execution reaches BOTH torch.load and load_state_dict sequentially (load first, then the model.load_state_dict on the loaded checkpoint); it is not 'at most one'. In product

### ocr — LIVE×4 / LATENT×1

| Site | Label | Trigger | kind | activation id | conf. |
|---|---|---|---|---|---|
| `DeepSeekHfProvider._lazy_load::from_pretrained#0` | **LIVE** | per-request lazy-first-load (double-checked: `if self | bundle | `ocr/deepseek-hf-mini` | high |
| `DeepSeekHfProvider._lazy_load::from_pretrained#1` | **LIVE** | per-request lazy-first-load (same call as #0, immediately follows in the same critical sec | bundle | `ocr/deepseek-hf-mini` | high |
| `DeepSeekHfProvider.extract::ctor:PaddleOCR#0` | **LIVE** | per-request lazy-first-load, self-triggered inline within extract() (not via a separate wa | bundle | `ocr/deepseek-hf-paddle-align` | high |
| `DeepSeekHfProvider.warmup::ctor:PaddleOCR#0` | **LATENT** | none | bundle | `ocr/deepseek-hf-paddle-align` | high |
| `PaddleOcrProvider._init_paddle::ctor:PaddleOCR#0` | **LIVE** | per-request lazy-first-load, self-triggered inline within extract() via an internal warmup | bundle | `ocr/paddle-primary` | high |

<details><summary><code>src/core/ocr/providers/deepseek_hf.py::DeepSeekHfProvider._lazy_load::from_pretrained#0</code> — LIVE (evidence)</summary>

**Trigger:** per-request lazy-first-load (double-checked: `if self._model is None` inside an asyncio.Lock, so subsequent requests reuse the already-loaded/stub model)

**Env conditions:** Requires `transformers` importable (AutoModelForCausalLM/AutoTokenizer not None at import time, line 16-19) else immediate stub at :98-100. Requires DEEPSEEK_HF_REVISION to be set to a 7+ hex-char commit-hash-looking string (:104-114, :115-123), OR DEEPSEEK_HF_ALLOW_UNPINNED=1 (env, read once at __init__ :70) to bypass the pin check (revision defaults to 'main' if unset, :114). Without one of these, the code sets self._model='stub' and returns before reaching from_pretrained. Reachable via explicit `provider=deepseek_hf` request param, or via OcrManager 'auto' strategy fallback when the primary (paddle) result is low-confidence or missing key dimension/symbol fields.

**Caller chain (top-down):**

1. src/main.py:410 app.include_router(api_router, prefix="/api")
1. src/api/__init__.py:271,274,532 api_router = APIRouter(); v1_router = APIRouter(prefix="/v1"); api_router.include_router(v1_router)
1. src/api/__init__.py:395-401 _include_router(v1_router, module=ocr, prefix="/ocr") -> mounted path /api/v1/ocr/*
1. src/api/v1/ocr.py:393 @router.post("/extract") async def ocr_extract(...)  [also :503 ocr_extract_base64]
1. src/api/v1/ocr.py:428 response = await _run_ocr_extract(image_bytes, provider, trace_id)
1. src/api/v1/ocr.py:233 manager = get_manager()  (bootstraps + registers 'paddle' and 'deepseek_hf' providers on first call)
1. src/api/v1/ocr.py:235 result = await manager.extract(image_bytes, strategy=provider, trace_id=trace_id)
1. src/core/ocr/manager.py:98 OcrManager.extract() -> :106 provider_name = self._select_provider(strategy)  [explicit provider='deepseek_hf'/'deepseek'/'deepseek-hf', OR auto-strategy fallback at :242-246 (missing key fields) / :296-304 (low confidence) which call deepseek.extract(...) directly]
1. src/core/ocr/manager.py:186 result = await provider.extract(image_bytes, trace_id=trace_id)  (provider = DeepSeekHfCoreProvider, an OcrProviderAdapter)
1. src/core/providers/ocr.py:79-83 OcrProviderAdapter.extract() -> return await self.process(image_bytes, trace_id=trace_id)
1. src/core/providers/base.py:91-92 BaseProvider.process() -> return await self._process_impl(request, **kwargs)
1. src/core/providers/ocr.py:51-58 OcrProviderAdapter._process_impl() -> return await self._wrapped_provider.extract(image_bytes=image_bytes, trace_id=trace_id)  (wrapped_provider = DeepSeekHfProvider instance)
1. src/core/ocr/providers/deepseek_hf.py:150 DeepSeekHfProvider.extract() -> :155-156 if not self._model: await self._lazy_load()
1. src/core/ocr/providers/deepseek_hf.py:90 _lazy_load() -> :128-131 self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self._revision)  [LOAD SITE #0]

**Notes:** Model name is configurable via DEEPSEEK_HF_MODEL env (default 'deepseek-ocr-mini'); an HF from_pretrained call pulls a multi-file repo, hence 'bundle'. This site (tokenizer) and from_pretrained#1 (causal LM, same file, lines 132-135) are two calls against the SAME model_name+revision inside the same _lazy_load() critical section — they represent one logical model activation, so they share this proposed_logical_activation_id. A secondary independent LIVE chain to the same OcrManager.extract() exists via src/api/v1/analyze.py:64,135 -> src/core/ocr/analysis_ocr_pipeline.py:30,53 run_analysis_ocr_pipeline(), and a third via src/api/v1/vision.py:37-94 get_vision_manager(); both corroborate but were not needed as the primary chain.

</details>

<details><summary><code>src/core/ocr/providers/deepseek_hf.py::DeepSeekHfProvider._lazy_load::from_pretrained#1</code> — LIVE (evidence)</summary>

**Trigger:** per-request lazy-first-load (same call as #0, immediately follows in the same critical section)

**Env conditions:** Identical gating to from_pretrained#0 (same guard block, lines 94-127, executes before either from_pretrained call): transformers importable, and DEEPSEEK_HF_REVISION=<commit-hash> or DEEPSEEK_HF_ALLOW_UNPINNED=1. If the tokenizer call (#0) throws, this call is skipped (caught by the outer except at :143-148, sets self._model='stub').

**Caller chain (top-down):**

1. (identical to from_pretrained#0 chain through DeepSeekHfProvider._lazy_load())
1. src/core/ocr/providers/deepseek_hf.py:90 _lazy_load() -> :132-135 self._model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self._revision)  [LOAD SITE #1]

**Notes:** Same logical activation as from_pretrained#0 (tokenizer + causal LM loaded together as one DeepSeek OCR model deployment); see notes on that entry for the shared id rationale and corroborating chains.

</details>

<details><summary><code>src/core/ocr/providers/deepseek_hf.py::DeepSeekHfProvider.extract::ctor:PaddleOCR#0</code> — LIVE (evidence)</summary>

**Trigger:** per-request lazy-first-load, self-triggered inline within extract() (not via a separate warmup call)

**Env conditions:** Reached unconditionally on every extract() call to this provider (independent of whether the from_pretrained pinning gate above passed or fell back to stub), gated only by: self._align_with_paddle defaulting True (constructor default; no OcrProviderAdapter kwargs override it in production, confirmed via bootstrap_core_ocr_providers using an empty OcrProviderConfig.provider_kwargs) AND the `paddleocr` python package being importable (PaddleOCR not None, checked at import time lines 42-45) AND self._paddle is None (first call on this provider instance). Same provider-selection conditions as from_pretrained#0/#1 (explicit provider=deepseek_hf, or manager auto-fallback).

**Caller chain (top-down):**

1. (same top-down chain as from_pretrained#0 through DeepSeekHfProvider.extract() at src/core/ocr/providers/deepseek_hf.py:150)
1. src/core/ocr/providers/deepseek_hf.py:155-156 await self._lazy_load()  (returns regardless of success/failure — sets self._model to real model or 'stub')
1. src/core/ocr/providers/deepseek_hf.py:264-270 (same extract() call, after parsing) if self._align_with_paddle and self._paddle is None and PaddleOCR: self._paddle = PaddleOCR(lang="ch", use_angle_cls=True, use_gpu=False)  [LOAD SITE at :268]

**Notes:** This is a SEPARATE PaddleOCR instance/activation from paddle.py's PaddleOcrProvider._init_paddle site below — different provider object (self._paddle on DeepSeekHfProvider, used only for bbox alignment), even though both construct PaddleOCR with similar default kwargs (lang='ch'). warmup() (lines 84-88) duplicates this same PaddleOCR construction but via a different, uncalled code path — see the warmup site entry.

</details>

<details><summary><code>src/core/ocr/providers/deepseek_hf.py::DeepSeekHfProvider.warmup::ctor:PaddleOCR#0</code> — LATENT (evidence)</summary>

**Trigger:** none — dead in production; DeepSeekHfProvider.warmup() is never invoked from any request path, startup hook, or background task

**Env conditions:** N/A — the load call is never reached regardless of env, because no caller exists in the production app graph.

**Caller chain (top-down):**

1. src/core/ocr/providers/deepseek_hf.py:81 async def warmup(self): ... :84-88 self._paddle = PaddleOCR(lang="ch", use_angle_cls=True, use_gpu=False)  -- reachable ONLY if something calls DeepSeekHfProvider.warmup() or the pass-through OcrProviderAdapter.warmup() (src/core/providers/ocr.py:71-77); grep across src/ for '.warmup(' found exactly two production call sites: src/core/ocr/providers/paddle.py:139 (self.warmup() inside PaddleOcrProvider.extract, unrelated class) and src/ml/serving/server.py:144 (unrelated ML-serving worker, not OCR). No caller of OcrProviderAdapter.warmup(), OcrManager (no warmup method exists on OcrManager at all), get_manager()/get_vision_manager(), or the FastAPI lifespan hook (src/main.py:145-353) invokes DeepSeekHfProvider.warmup() or its adapter wrapper.

**Notes:** This duplicates the PaddleOCR-construction logic already reachable via DeepSeekHfProvider.extract() (previous site), but as a standalone public warmup() method it has no production caller: it is not wired to FastAPI startup/lifespan (src/main.py:145-353 has no OCR-specific warmup call, only bootstrap_core_provider_registry() at :152 which just registers provider CLASSES, not instances), not wired to src/models/loader.py::load_models() (delegates to build_model_readiness_snapshot()->readiness_registry._ocr_item() at src/models/readiness_registry.py:328-347 which only introspects module state via _module_object/getattr on the existing src.api.v1.ocr._manager global, never constructs/warms the manager), and not exposed via any admin/warmup HTTP route (grep of src/api/ for 'warmup' returns nothing). Confirmed importable/exported (class is instantiated in prod, and this is a normal method on it) but truly gate-before-wired per the LATENT definition. Manifest 'reason' string ('MOUNTED /ocr') is misleading here since it conflates the file's overall mount status with this specific method's reachability -- do not copy that hypothesis.

</details>

<details><summary><code>src/core/ocr/providers/paddle.py::PaddleOcrProvider._init_paddle::ctor:PaddleOCR#0</code> — LIVE (evidence)</summary>

**Trigger:** per-request lazy-first-load, self-triggered inline within extract() via an internal warmup() call (paddle is the DEFAULT provider for strategy='auto')

**Env conditions:** Requires the `paddleocr` python package importable (PaddleOCR not None, checked at import time lines 18-21); if unavailable, warmup() sets self._initialized=True without constructing PaddleOCR and extract() falls to the hardcoded stub text branch (:205-212). No env var gate beyond that — this is the DEFAULT path for any /api/v1/ocr/extract or /extract-base64 call with the default strategy='auto' (or explicit provider='paddle').

**Caller chain (top-down):**

1. src/main.py:410 app.include_router(api_router, prefix="/api")
1. src/api/__init__.py:271,274,532 api_router / v1_router(prefix="/v1") wiring
1. src/api/__init__.py:395-401 _include_router(v1_router, module=ocr, prefix="/ocr") -> /api/v1/ocr/*
1. src/api/v1/ocr.py:393 @router.post("/extract") async def ocr_extract(...)  [also :503 ocr_extract_base64; default `provider` param = 'auto']
1. src/api/v1/ocr.py:428 -> :233-235 get_manager() / manager.extract(image_bytes, strategy=provider='auto', ...)
1. src/core/ocr/manager.py:98 OcrManager.extract() -> :106 _select_provider('auto') -> :413-414 'if "paddle" in self.providers: return "paddle"'  (paddle is preferred first; always registered by get_manager())
1. src/core/ocr/manager.py:186 result = await provider.extract(image_bytes, trace_id=trace_id)  (provider = PaddleCoreProvider, an OcrProviderAdapter)
1. src/core/providers/ocr.py:79-83 OcrProviderAdapter.extract() -> self.process(...) -> src/core/providers/base.py:91-92 -> src/core/providers/ocr.py:51-58 _process_impl() -> self._wrapped_provider.extract(...)  (wrapped_provider = PaddleOcrProvider)
1. src/core/ocr/providers/paddle.py:119 PaddleOcrProvider.extract() -> :138-139 if self._ocr is None and PaddleOCR and not self._initialized: await self.warmup()
1. src/core/ocr/providers/paddle.py:93 warmup() -> :99 self._ocr = self._init_paddle(kwargs)
1. src/core/ocr/providers/paddle.py:86 _init_paddle() -> :91 return PaddleOCR(**filtered)  [LOAD SITE]  (fallback re-entry at :110 on primary-init exception, same function/site)

**Notes:** This is the primary/default OCR provider construction path — the most directly and unconditionally reachable of the 5 assigned sites (no explicit strategy override needed, no revision-pinning env gate). Distinct provider instance/activation from the DeepSeekHfProvider._paddle aligner (site above) despite similar default kwargs (lang='ch'); each is a separate PaddleOCR() object with independent lifecycle. Also independently reachable via src/api/v1/analyze.py's run_analysis_ocr_pipeline() and src/api/v1/vision.py's get_vision_manager(), both of which register 'paddle' the same way. Note: OCR_PROVIDER_DEFAULT env (readiness_registry._ocr_item) is display-only metadata and does NOT feed OcrManager._select_provider, which hardcodes paddle-then-deepseek for 'auto'.

</details>

**Verifier corrections (2):**
- `src/core/ocr/providers/deepseek_hf.py::DeepSeekHfProvider._lazy_load::from_pretrained#0` — caller_chain hop 0 line ref corrected 414->410 (app.include_router); hop 1 v1_router line ref corrected 273->274.
- `src/core/ocr/providers/paddle.py::PaddleOcrProvider._init_paddle::ctor:PaddleOCR#0` — caller_chain hop 0 line ref corrected 414->410 (app.include_router); hop 1 v1_router line ref corrected 273->274.

### part — LIVE×6

| Site | Label | Trigger | kind | activation id | conf. |
|---|---|---|---|---|---|
| `PartClassifier._load_model::torch.load#0` | **LIVE** | lazy-first-request (env-conditioned): module-level singleton `_ml_classifier` in src/core/ | single-file | `part/v6` | high |
| `PartClassifier._load_model::load_state_dict#0` | **LIVE** | lazy-first-request (env-conditioned), identical trigger to the torch | single-file | `part/v6` | high |
| `PartClassifierV16._load_models::torch.load#0` | **LIVE** | lazy-first-request (env-conditioned): PartClassifierV16 | open-question | `part/v16-v6pt` | high |
| `PartClassifierV16._load_models::load_state_dict#0` | **LIVE** | lazy-first-request (env-conditioned), identical trigger to the torch | open-question | `part/v16-v6pt` | high |
| `PartClassifierV16._load_models::torch.load#1` | **LIVE** | lazy-first-request (env-conditioned), identical trigger to the other _load_models sites -- | open-question | `part/v16-v6pt` | high |
| `PartClassifierV16._load_models::load_state_dict#1` | **LIVE** | lazy-first-request (env-conditioned), identical trigger to the other _load_models sites -- | open-question | `part/v16-v6pt` | high |

<details><summary><code>src/ml/part_classifier.py::PartClassifier._load_model::torch.load#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request (env-conditioned): module-level singleton `_ml_classifier` in src/core/analyzer.py is constructed (and _load_model runs synchronously inside PartClassifier.__init__) the first time a POST /api/v1/analyze/ request reaches the shadow classifier stage with the v6 provider selected; cached process-wide thereafter.

**Env conditions:** PART_CLASSIFIER_PROVIDER_ENABLED=true (default false, hard gate) AND PART_CLASSIFIER_PROVIDER_NAME=v6 (default is "v16", so must be explicitly overridden -- otherwise the v16 provider is used instead) AND file_format in PART_CLASSIFIER_PROVIDER_SHADOW_FORMATS (default dxf,dwg) AND upload size <= PART_CLASSIFIER_PROVIDER_MAX_MB (default 10.0) AND CAD_CLASSIFIER_MODEL path (default models/cad_classifier_v6.pt) exists on disk AND analysis_options.classify_parts=true (request-level, defaults true).

**Caller chain (top-down):**

1. Dockerfile CMD: python -m uvicorn src.main:app (re-verified, 3 occurrences)
1. src/main.py:410 app.include_router(api_router, prefix="/api")
1. src/api/__init__.py:288-294 analyze router mounted into v1_router at prefix /analyze -> POST /api/v1/analyze/
1. src/api/v1/analyze.py:89 @router.post("/") async def analyze_cad_file(...)
1. src/api/v1/analyze.py:110 return await run_analysis_live_pipeline(...)
1. src/core/analysis_live_pipeline.py:12 run_analysis_live_pipeline(...) -> calls run_parallel_pipeline_fn(..., content=content, ...)
1. src/core/analysis_parallel_pipeline.py:20 run_analysis_parallel_pipeline(...); :50 `if analysis_options.classify_parts:` (default True, src/api/v1/analyze_live_models.py:13)
1. src/core/analysis_parallel_pipeline.py:54 cls_payload = await classify_pipeline(..., content=content, ...) [== run_classification_pipeline]
1. src/core/classification/classification_pipeline.py:33 run_classification_pipeline(...) -> :63 shadow_context = await build_shadow_classification_context(..., content=content, ...)
1. src/core/classification/shadow_pipeline.py:613 build_shadow_classification_context(...) -> :637 cls_payload = await _run_part_classifier_shadow(..., content=content)
1. src/core/classification/shadow_pipeline.py:481 _run_part_classifier_shadow(...); gate :489-491 env PART_CLASSIFIER_PROVIDER_ENABLED must be "true" (default "false"); gate :495-501 file_format in env PART_CLASSIFIER_PROVIDER_SHADOW_FORMATS (default "dxf,dwg"); gate :518 size_mb <= env PART_CLASSIFIER_PROVIDER_MAX_MB (default 10.0); :532 writes uploaded bytes to a real NamedTemporaryFile -> tmp_path; :507 provider_name = os.getenv("PART_CLASSIFIER_PROVIDER_NAME", "v16") -- must be explicitly set to "v6" to hit this site; :534 provider = _get_classifier_provider(provider_name); :537 part_result = await asyncio.wait_for(provider.process(_make_classifier_request(filename=file_name, file_path=tmp_path)), timeout=...)
1. src/core/classification/shadow_pipeline.py:329-332 _get_classifier_provider -> bootstrap_core_provider_registry(); ProviderRegistry.get("classifier", "v6")
1. src/core/providers/classifier.py:357-419 bootstrap_core_classifier_providers() registers classifier/v6 -> V6CoreProvider(V6PartClassifierProviderAdapter) (also called eagerly at src/main.py:152 startup)
1. src/core/providers/base.py:91-92 BaseProvider.process(request) -> self._process_impl(request)
1. src/core/providers/classifier.py:326 V6PartClassifierProviderAdapter._process_impl(request); :331 requires request.file_path truthy (satisfied); clf = _get_ml_classifier(); :339 result = clf.predict(str(request.file_path))
1. src/core/analyzer.py:90 _get_ml_classifier(); :99 gated by os.path.exists(model_path) where model_path = os.getenv("CAD_CLASSIFIER_MODEL", "models/cad_classifier_v6.pt"); :101 _ml_classifier = PartClassifier(model_path)
1. src/ml/part_classifier.py:43 PartClassifier.__init__(model_path) -> :55 self._load_model() (eager, synchronous inside constructor)
1. src/ml/part_classifier.py:57 def _load_model(self): -> :62 checkpoint = torch.load(self.model_path, map_location=self.device)  <-- LOAD SITE

**Notes:** Two OTHER direct-construction sites for PartClassifier() exist but are NOT live: (1) src/core/analyzer.py CADAnalyzer._classify_with_ml (analyzer.py:176-186) calls the SAME _get_ml_classifier() singleton, but always short-circuits at analyzer.py:180-182 before reaching it -- getattr(doc, "file_path"/"source_path") is always None because CadDocument (src/models/cad_document.py) has no such Pydantic field and no adapter in src/adapters/factory.py or src/core/document_pipeline.py ever sets one (extra kwargs are silently dropped under default Pydantic extra="ignore"). Wired but dead in the current /analyze flow. (2) src/core/assistant/tools/classify_tool.py:55 constructs PartClassifier() directly when ClassifyTool.execute() is called with use_hybrid=False, but that tool's only callers -- FunctionCallingEngine (src/core/assistant/function_calling.py) and AnalysisReportGenerator (src/core/assistant/report_generator.py) -- are never instantiated or imported anywhere else in src/ (verified via grep); the mounted src/api/v1/assistant.py router has zero imports from either module, so this whole subtree is orphaned/unmounted, not reachable from the production ASGI app. /health does not reach this load call either: _health_check_impl on V6PartClassifierProviderAdapter (classifier.py) only checks os.path.exists() (_model_present), never calls .predict().

</details>

<details><summary><code>src/ml/part_classifier.py::PartClassifier._load_model::load_state_dict#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request (env-conditioned), identical trigger to the torch.load#0 site above -- same _load_model() call, next statement.

**Env conditions:** Identical to the torch.load#0 site: PART_CLASSIFIER_PROVIDER_ENABLED=true AND PART_CLASSIFIER_PROVIDER_NAME=v6 AND file_format in PART_CLASSIFIER_PROVIDER_SHADOW_FORMATS AND upload size <= PART_CLASSIFIER_PROVIDER_MAX_MB AND CAD_CLASSIFIER_MODEL path exists AND classify_parts=true. This statement executes unconditionally right after the torch.load#0 statement inside the same _load_model() call (no additional gate), and only after self.model is built via _build_v2_model/_build_v6_model/_build_v8_model based on checkpoint["version"] (analyzer.py:64-77).

**Caller chain (top-down):**

1. (identical chain through src/ml/part_classifier.py:57 def _load_model(self) as the torch.load#0 site on this file)
1. src/ml/part_classifier.py:62 checkpoint = torch.load(self.model_path, map_location=self.device)
1. src/ml/part_classifier.py:79 self.model.load_state_dict(checkpoint["model_state_dict"])  <-- LOAD SITE

**Notes:** Same reachability analysis and same two dead alternate paths (CADAnalyzer._classify_with_ml, classify_tool.py's ClassifyTool) as the torch.load#0 sibling site -- see that entry's notes for the full dead-path verification. Both statements belong to the same PartClassifier._load_model() call, so they share one activation.

</details>

<details><summary><code>src/ml/part_classifier.py::PartClassifierV16._load_models::torch.load#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request (env-conditioned): PartClassifierV16.__init__ does NOT call _load_models -- loading is deferred to the first .predict() call. Module-level singleton `_v16_classifier` in src/core/analyzer.py is constructed on first reachable /analyze request; _load_models() itself is idempotent (short-circuits via self.loaded flag) so the actual torch.load only fires once per process, on the first request that reaches predict().

**Env conditions:** PART_CLASSIFIER_PROVIDER_ENABLED=true (default false, hard gate) AND file_format in PART_CLASSIFIER_PROVIDER_SHADOW_FORMATS (default dxf,dwg) AND upload size <= PART_CLASSIFIER_PROVIDER_MAX_MB (default 10.0) AND DISABLE_V16_CLASSIFIER not true (default unset = enabled) AND both models/cad_classifier_v6.pt and models/cad_classifier_v14_ensemble.pt exist on disk AND analysis_options.classify_parts=true (default true). provider_name defaults to "v16" so no PART_CLASSIFIER_PROVIDER_NAME override is needed for this branch (unlike the V6 sites).

**Caller chain (top-down):**

1. Dockerfile CMD: python -m uvicorn src.main:app (re-verified, 3 occurrences)
1. src/main.py:410 app.include_router(api_router, prefix="/api")
1. src/api/__init__.py:288-294 analyze router mounted into v1_router at prefix /analyze -> POST /api/v1/analyze/
1. src/api/v1/analyze.py:89 @router.post("/") async def analyze_cad_file(...)
1. src/api/v1/analyze.py:110 return await run_analysis_live_pipeline(...)
1. src/core/analysis_live_pipeline.py:12 run_analysis_live_pipeline(...) -> calls run_parallel_pipeline_fn(..., content=content, ...)
1. src/core/analysis_parallel_pipeline.py:20 run_analysis_parallel_pipeline(...); :50 `if analysis_options.classify_parts:` (default True)
1. src/core/analysis_parallel_pipeline.py:54 cls_payload = await classify_pipeline(..., content=content, ...) [== run_classification_pipeline]
1. src/core/classification/classification_pipeline.py:33 run_classification_pipeline(...) -> :63 shadow_context = await build_shadow_classification_context(..., content=content, ...)
1. src/core/classification/shadow_pipeline.py:613 build_shadow_classification_context(...) -> :637 cls_payload = await _run_part_classifier_shadow(..., content=content)
1. src/core/classification/shadow_pipeline.py:481 _run_part_classifier_shadow(...); gate :489-491 env PART_CLASSIFIER_PROVIDER_ENABLED must be "true" (default "false"); gate :495-501 file_format in env PART_CLASSIFIER_PROVIDER_SHADOW_FORMATS (default "dxf,dwg"); gate :518 size_mb <= env PART_CLASSIFIER_PROVIDER_MAX_MB (default 10.0); :532 writes uploaded bytes to a real NamedTemporaryFile -> tmp_path; :507 provider_name = os.getenv("PART_CLASSIFIER_PROVIDER_NAME", "v16") -- this is the DEFAULT, no override needed; :534 provider = _get_classifier_provider(provider_name); :537 part_result = await asyncio.wait_for(provider.process(_make_classifier_request(filename=file_name, file_path=tmp_path)), timeout=...)
1. src/core/classification/shadow_pipeline.py:329-332 _get_classifier_provider -> bootstrap_core_provider_registry(); ProviderRegistry.get("classifier", "v16")
1. src/core/providers/classifier.py:357-419 bootstrap_core_classifier_providers() registers classifier/v16 -> V16CoreProvider(V16PartClassifierProviderAdapter) (also called eagerly at src/main.py:152 startup)
1. src/core/providers/base.py:91-92 BaseProvider.process(request) -> self._process_impl(request)
1. src/core/providers/classifier.py:265 V16PartClassifierProviderAdapter._process_impl(request); :270 requires request.file_path truthy (satisfied); :275 clf = _get_v16_classifier(); :278 result = clf.predict(str(request.file_path))
1. src/core/analyzer.py:54 _get_v16_classifier(speed_mode="fast"); :58-59 gated by env DISABLE_V16_CLASSIFIER not true; :73 gated by os.path.exists("models/cad_classifier_v6.pt") and os.path.exists("models/cad_classifier_v14_ensemble.pt"); :75 _v16_classifier = PartClassifierV16(speed_mode=env_speed_mode, enable_cache=True, cache_size=int(os.getenv("V16_CACHE_SIZE","1000")))
1. src/ml/part_classifier.py:517 PartClassifierV16.__init__ constructs the object only; does NOT call _load_models (confirmed no such call in __init__ body lines 517-556)
1. back at src/core/providers/classifier.py:278 clf.predict(...) -> src/ml/part_classifier.py:981 def predict(self, file_path): -> :996 self._load_models()  [lazy-first-predict trigger]
1. src/ml/part_classifier.py:637 def _load_models(self): :639 `if self.loaded: return` (guards re-entry on later requests)
1. src/ml/part_classifier.py:655 v6_ckpt = torch.load(v6_path, map_location=self.device, weights_only=False)  <-- LOAD SITE

**Notes:** _load_models() loads TWO checkpoint files (cad_classifier_v6.pt at :655 and cad_classifier_v14_ensemble.pt at :695) as one logical V16 activation -- same two-distinct-files-one-logical-activation shape called out in the mission brief; owner decision pending on tuple-vs-per-file-pin-vs-bundle-KIND modeling, same open question as classifier_api.py::V16Classifier. A second wired-but-dead sub-path exists: CADAnalyzer._classify_with_v16 (src/core/analyzer.py:128-175) calls the SAME _get_v16_classifier() singleton, but always short-circuits before reaching it because doc.file_path/source_path is never set on any CadDocument (see torch.load#0-on-part_classifier.py entry for the PartClassifier family for the full verification -- same root cause applies here, CadDocument has no such field and no adapter sets one). /health does not reach this load call: V16PartClassifierProviderAdapter._health_check_impl (classifier.py:252-262) only checks os.path.exists() for both model files (_models_present), never constructs PartClassifierV16 or calls .predict().

</details>

<details><summary><code>src/ml/part_classifier.py::PartClassifierV16._load_models::load_state_dict#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request (env-conditioned), identical trigger to the torch.load#0 sibling above -- same _load_models() call, a few statements later (after building ImprovedClassifierV6 architecture at :657-680).

**Env conditions:** Identical to the torch.load#0 site on this file: PART_CLASSIFIER_PROVIDER_ENABLED=true AND file_format in PART_CLASSIFIER_PROVIDER_SHADOW_FORMATS AND upload size <= PART_CLASSIFIER_PROVIDER_MAX_MB AND DISABLE_V16_CLASSIFIER not true AND both v6/v14 model files exist AND classify_parts=true. Executes unconditionally right after torch.load#0 in the same _load_models() call.

**Caller chain (top-down):**

1. (identical chain through src/ml/part_classifier.py:637 def _load_models(self) as the torch.load#0 site on this file)
1. src/ml/part_classifier.py:655 v6_ckpt = torch.load(v6_path, ...)
1. src/ml/part_classifier.py:657-680 local class ImprovedClassifierV6(nn.Module) defined; :682 self.v6_model = ImprovedClassifierV6(48, 256, 5, 0.5)
1. src/ml/part_classifier.py:683 self.v6_model.load_state_dict(v6_ckpt['model_state_dict'])  <-- LOAD SITE

**Notes:** Part of the same single _load_models() invocation as the other 3 PartClassifierV16 sites -- one logical V16 activation loading two files (v6.pt loaded/applied here; v14_ensemble.pt loaded/applied at torch.load#1/load_state_dict#1 below). See the torch.load#0 sibling entry for the full dead-alternate-path verification (CADAnalyzer._classify_with_v16 wired-but-dead; /health does not reach this call).

</details>

<details><summary><code>src/ml/part_classifier.py::PartClassifierV16._load_models::torch.load#1</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request (env-conditioned), identical trigger to the other _load_models sites -- executes after the V6 branch completes, inside `if v14_path.exists():` at :694.

**Env conditions:** Same as the other three PartClassifierV16 sites, PLUS the local guard at :694 `if v14_path.exists()` -- redundant in practice because src/core/analyzer.py:73 already required os.path.exists(v14_path) as a precondition to construct PartClassifierV16 at all in the only live caller (_get_v16_classifier). So this local re-check always passes when the outer chain is reached; it only matters as defense-in-depth if PartClassifierV16 were ever constructed by a caller that skips the analyzer.py existence gate (none found in this checkout).

**Caller chain (top-down):**

1. (identical chain through src/ml/part_classifier.py:637 def _load_models(self) as the other three PartClassifierV16 sites on this file)
1. src/ml/part_classifier.py:692-694 v14_path = self.model_dir / "cad_classifier_v14_ensemble.pt"; if v14_path.exists():
1. src/ml/part_classifier.py:695 v14_ckpt = torch.load(v14_path, map_location=self.device, weights_only=False)  <-- LOAD SITE

**Notes:** Second file of the same one-logical-V16-activation _load_models() call. See torch.load#0 sibling entry for full chain verification and dead-alternate-path notes (CADAnalyzer._classify_with_v16 wired-but-dead due to doc.file_path never set; /health does not reach this call, only checks file existence).

</details>

<details><summary><code>src/ml/part_classifier.py::PartClassifierV16._load_models::load_state_dict#1</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request (env-conditioned), identical trigger to the other _load_models sites -- executes inside the `for fold_state in v14_ckpt['fold_states']:` loop at :696-698, once per ensemble fold.

**Env conditions:** Same as the other three PartClassifierV16 sites, executing only when v14_path.exists() (see torch.load#1 sibling) and once per fold_state entry found in v14_ckpt['fold_states'].

**Caller chain (top-down):**

1. (identical chain through src/ml/part_classifier.py:637 def _load_models(self) as the other three PartClassifierV16 sites on this file)
1. src/ml/part_classifier.py:695 v14_ckpt = torch.load(v14_path, ...)
1. src/ml/part_classifier.py:696 for fold_state in v14_ckpt['fold_states']: :697 model = _FusionModelV14(48, 5)
1. src/ml/part_classifier.py:698 model.load_state_dict(fold_state)  <-- LOAD SITE (once per fold; number of folds set by SPEED_MODES[speed_mode]['v14_folds'] used only for inference-time truncation elsewhere, not for this load loop which iterates all folds present in the checkpoint)

**Notes:** Final statement of the same one-logical-V16-activation _load_models() call as the other 3 PartClassifierV16 sites in this family. See torch.load#0 sibling entry for the complete chain verification and dead-alternate-path analysis (CADAnalyzer._classify_with_v16 wired-but-dead; /health does not reach any of these 4 load statements).

</details>

### pointnet — LIVE×5

| Site | Label | Trigger | kind | activation id | conf. |
|---|---|---|---|---|---|
| `PointNet3DAnalyzer._try_load_model::torch.load#0` | **LIVE** | lazy-first-request, env-conditioned | single-file | `pointnet/main` | high |
| `PointNet3DAnalyzer._try_load_model::load_state_dict#0` | **LIVE** | lazy-first-request, env-conditioned | single-file | `pointnet/main` | high |
| `PointNet3DAnalyzer._try_load_model::load_state_dict#1` | **LIVE** | lazy-first-request, env-conditioned | single-file | `pointnet/main` | high |
| `PointNet3DAnalyzer._try_load_model::load_state_dict#2` | **LIVE** | lazy-first-request, env-conditioned | single-file | `pointnet/main` | high |
| `PointNet3DAnalyzer._try_load_model::load_state_dict#3` | **LIVE** | lazy-first-request, env-conditioned | single-file | `pointnet/main` | high |

<details><summary><code>src/ml/pointnet/inference.py::PointNet3DAnalyzer._try_load_model::torch.load#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request, env-conditioned

**Env conditions:** Route mounts unconditionally (pointcloud import chain touches only numpy unguarded; torch/trimesh are try/except-guarded in inference.py:20-26 and preprocessor.py:17-31). Two runtime gates must both hold for the load to fire: (1) HAS_TORCH -- torch importable (inference.py:20-24); (2) POINTNET_MODEL_PATH set (no default; os.getenv at pointcloud.py:34) to a path os.path.exists() confirms at construction time (inference.py:88). Gate (1) is build-dependent and verified directly by reading both Dockerfiles: torch is NOT in base requirements.txt (commented at :21) and IS in requirements-l3.txt:2 (x86_64-only). Root/verified-anchor Dockerfile default `production` stage (:37-90, CMD uvicorn src.main:app :90) copies deps from a builder that installs ONLY requirements.txt (:31-32) => torch ABSENT; torch is present in that file only in the GPU stage, which explicitly `pip install torch` (:137). The alternate deployments/docker/Dockerfile installs requirements-l3.txt by DEFAULT (ARG INSTALL_L3_DEPS=1, :31-35) => torch PRESENT on x86_64, and its CMD `python -m src.main` (:60) serves the same src.main:app via uvicorn.run (main.py:635-637). So whether gate (1) is already satisfied at build depends on which image/stage is deployed; on a default deployments/docker (L3) x86_64 build the sole remaining gate is POINTNET_MODEL_PATH. Label is LIVE either way. POINTNET_ENABLED (default True) is read only by the passive readiness registry (readiness_registry.py:303-308), which introspects the already-constructed `_analyzer` and never constructs it -- it does NOT gate this load.

**Caller chain (top-down):**

1. Dockerfile:90 CMD ["python","-m","uvicorn","src.main:app",...] (root Dockerfile production stage; also deployments/docker/Dockerfile:60 CMD ["python","-m","src.main"] -> main.py:635-637 uvicorn.run("src.main:app") -- both serve the same ASGI app)
1. src/main.py:20 `from src.api import api_router` -> imports src/api/__init__.py
1. src/api/__init__.py:269 `pointcloud = _import_router("pointcloud", "src.api.v1.pointcloud")` -> _load_router_module (:150-186) imports the module; import succeeds unconditionally because inference.py:20-26 and preprocessor.py:25-31 both guard `import torch` in try/except and only numpy (a base dep) is imported unguarded
1. src/api/__init__.py:522-530 `if pointcloud is not None: _include_router(v1_router, name="pointcloud", module=pointcloud, prefix="/pointcloud", ...)` -> _include_router (:189-223) calls v1_router.include_router(pointcloud.router, prefix="/pointcloud")
1. src/api/__init__.py:532 `api_router.include_router(v1_router)` (v1_router has prefix "/v1")
1. src/main.py:410 `app.include_router(api_router, prefix="/api")` -> final routes: POST /api/v1/pointcloud/{classify,features,similar}
1. src/api/v1/pointcloud.py:101-114 classify_pointcloud (also :117-130 extract_features, :133-149 find_similar) call `_get_analyzer()` at :106 / :122 / :141
1. src/api/v1/pointcloud.py:31-36 `_get_analyzer()`: module-level lazy singleton (`_analyzer` is None at boot -- no startup preload); first call reads `model_path = os.getenv("POINTNET_MODEL_PATH", None)` (:34) then `_analyzer = PointNet3DAnalyzer(model_path=model_path)` (:35)
1. src/ml/pointnet/inference.py:76 `PointNet3DAnalyzer.__init__` -> `self._try_load_model()`
1. src/ml/pointnet/inference.py:82-110 `_try_load_model()`: gate `if not HAS_TORCH: return` (:84), gate `if self.model_path is None or not os.path.exists(self.model_path): return` (:88), then LOAD CALL `checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)` (:108-110)

**Notes:** Verified hop-by-hop against source. torch.load#0 (line 108) always executes once both gates pass -- unconditional inside the try block, unlike the load_state_dict calls below it. Single .pt checkpoint file (single-file kind); all load_state_dict sites below read from this same checkpoint object, so one logical activation = pointnet/main (not an open-question two-file case; those are classifier_api.py V16 and part_classifier.py V16). Second caller src/core/assistant/tools/pointcloud_tool.py:51 constructs PointNet3DAnalyzer() with model_path=None (default) so it always short-circuits at inference.py:88 and never reaches this load -- correctly excluded. readiness_registry.py:306 only reads the `_analyzer` module attribute (introspection), never constructs. No startup/lifespan preload exists (grep of PointNet3DAnalyzer and _get_analyzer across src/ shows construction only at pointcloud.py:35 route path + the dead-end tool).

</details>

<details><summary><code>src/ml/pointnet/inference.py::PointNet3DAnalyzer._try_load_model::load_state_dict#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request, env-conditioned

**Env conditions:** Same two gates as torch.load#0 (HAS_TORCH importable -- absent from base requirements.txt, present via requirements-l3.txt on x86_64 which the deployments/docker Dockerfile installs by default (INSTALL_L3_DEPS=1) or the root Dockerfile's GPU stage; and POINTNET_MODEL_PATH set to an existing file). Additionally this branch fires only when the loaded checkpoint dict contains key 'classifier_state_dict'; mutually exclusive with load_state_dict#1 and #2 via if/elif/else.

**Caller chain (top-down):**

1. Dockerfile:90 CMD python -m uvicorn src.main:app (root production stage; deployments/docker/Dockerfile:60 python -m src.main also serves src.main:app via main.py:635-637)
1. src/main.py:20 `from src.api import api_router` -> src/api/__init__.py
1. src/api/__init__.py:269 `pointcloud = _import_router("pointcloud", "src.api.v1.pointcloud")` (guarded-torch import; unconditional success)
1. src/api/__init__.py:522-530 `_include_router(v1_router, name="pointcloud", module=pointcloud, prefix="/pointcloud")`
1. src/api/__init__.py:532 `api_router.include_router(v1_router)`
1. src/main.py:410 `app.include_router(api_router, prefix="/api")` -> POST /api/v1/pointcloud/{classify,features,similar}
1. src/api/v1/pointcloud.py:106/:122/:141 route handlers call `_get_analyzer()`
1. src/api/v1/pointcloud.py:31-36 `_get_analyzer()` -> `_analyzer = PointNet3DAnalyzer(model_path=os.getenv("POINTNET_MODEL_PATH", None))`
1. src/ml/pointnet/inference.py:76 `__init__` -> `self._try_load_model()`
1. src/ml/pointnet/inference.py:82-116 `_try_load_model()`: gates :84 (HAS_TORCH) and :88 (model_path exists); `checkpoint = torch.load(...)` (:108-110); `self._classifier = PointNetClassifier(num_classes=...)` (:112-114); LOAD CALL `if "classifier_state_dict" in checkpoint: self._classifier.load_state_dict(checkpoint["classifier_state_dict"])` (:115-116)

**Notes:** Line 116, first branch of a 3-way if/elif/else keyed on checkpoint dict shape (:115 if / :117 elif / :119 else). Exactly one of load_state_dict#0/#1/#2 always executes once torch.load succeeds -- alternative paths through the same reached statement group, not independently-optional calls. Same caller chain as torch.load#0, diverging only at the final hop. Verified against source.

</details>

<details><summary><code>src/ml/pointnet/inference.py::PointNet3DAnalyzer._try_load_model::load_state_dict#1</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request, env-conditioned

**Env conditions:** Same two build/runtime gates as torch.load#0 (torch importable via L3/GPU build + POINTNET_MODEL_PATH set to an existing file). This branch additionally requires the checkpoint dict to lack 'classifier_state_dict' but contain 'state_dict'; mutually exclusive with #0 and #2.

**Caller chain (top-down):**

1. Dockerfile:90 CMD python -m uvicorn src.main:app (root production stage; deployments/docker/Dockerfile:60 python -m src.main also serves src.main:app via main.py:635-637)
1. src/main.py:20 `from src.api import api_router` -> src/api/__init__.py
1. src/api/__init__.py:269 `pointcloud = _import_router("pointcloud", "src.api.v1.pointcloud")`
1. src/api/__init__.py:522-530 `_include_router(v1_router, name="pointcloud", module=pointcloud, prefix="/pointcloud")`
1. src/api/__init__.py:532 `api_router.include_router(v1_router)`
1. src/main.py:410 `app.include_router(api_router, prefix="/api")` -> POST /api/v1/pointcloud/{classify,features,similar}
1. src/api/v1/pointcloud.py:106/:122/:141 route handlers call `_get_analyzer()`
1. src/api/v1/pointcloud.py:31-36 `_get_analyzer()` -> `_analyzer = PointNet3DAnalyzer(model_path=os.getenv("POINTNET_MODEL_PATH", None))`
1. src/ml/pointnet/inference.py:76 `__init__` -> `self._try_load_model()`
1. src/ml/pointnet/inference.py:82-118 `_try_load_model()`: gates :84/:88; `checkpoint = torch.load(...)` (:108-110); classifier constructed (:112-114); LOAD CALL `elif "state_dict" in checkpoint: self._classifier.load_state_dict(checkpoint["state_dict"])` (:117-118)

**Notes:** Line 118, second branch (elif) of the same 3-way checkpoint-shape dispatch as #0. Same caller chain, diverging only at the final hop. Verified against source.

</details>

<details><summary><code>src/ml/pointnet/inference.py::PointNet3DAnalyzer._try_load_model::load_state_dict#2</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request, env-conditioned

**Env conditions:** Same two build/runtime gates as torch.load#0 (torch importable via L3/GPU build + POINTNET_MODEL_PATH set to an existing file). This else branch fires whenever the checkpoint dict contains neither 'classifier_state_dict' nor 'state_dict' -- the whole checkpoint object is passed directly as the state dict. Mutually exclusive with #0 and #1.

**Caller chain (top-down):**

1. Dockerfile:90 CMD python -m uvicorn src.main:app (root production stage; deployments/docker/Dockerfile:60 python -m src.main also serves src.main:app via main.py:635-637)
1. src/main.py:20 `from src.api import api_router` -> src/api/__init__.py
1. src/api/__init__.py:269 `pointcloud = _import_router("pointcloud", "src.api.v1.pointcloud")`
1. src/api/__init__.py:522-530 `_include_router(v1_router, name="pointcloud", module=pointcloud, prefix="/pointcloud")`
1. src/api/__init__.py:532 `api_router.include_router(v1_router)`
1. src/main.py:410 `app.include_router(api_router, prefix="/api")` -> POST /api/v1/pointcloud/{classify,features,similar}
1. src/api/v1/pointcloud.py:106/:122/:141 route handlers call `_get_analyzer()`
1. src/api/v1/pointcloud.py:31-36 `_get_analyzer()` -> `_analyzer = PointNet3DAnalyzer(model_path=os.getenv("POINTNET_MODEL_PATH", None))`
1. src/ml/pointnet/inference.py:76 `__init__` -> `self._try_load_model()`
1. src/ml/pointnet/inference.py:82-120 `_try_load_model()`: gates :84/:88; `checkpoint = torch.load(...)` (:108-110); classifier constructed (:112-114); LOAD CALL `else: self._classifier.load_state_dict(checkpoint)` (:119-120)

**Notes:** Line 120, else branch of the same 3-way checkpoint-shape dispatch as #0/#1 -- the fallback branch, guaranteed to fire for any checkpoint lacking the two wrapper keys (e.g. a raw state_dict() saved directly). Same caller chain, diverging only at the final hop. Verified against source.

</details>

<details><summary><code>src/ml/pointnet/inference.py::PointNet3DAnalyzer._try_load_model::load_state_dict#3</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request, env-conditioned

**Env conditions:** Same two build/runtime gates as torch.load#0 (torch importable via L3/GPU build + POINTNET_MODEL_PATH set to an existing file). Independently of which classifier branch (#0/#1/#2) fired, this call additionally requires the checkpoint dict to contain key 'extractor_state_dict'. Unlike #0/#1/#2 this is a standalone `if` (:127), NOT part of the elif chain -- so a fully successful classifier load can still skip this call if the checkpoint was saved without an extractor sub-state-dict.

**Caller chain (top-down):**

1. Dockerfile:90 CMD python -m uvicorn src.main:app (root production stage; deployments/docker/Dockerfile:60 python -m src.main also serves src.main:app via main.py:635-637)
1. src/main.py:20 `from src.api import api_router` -> src/api/__init__.py
1. src/api/__init__.py:269 `pointcloud = _import_router("pointcloud", "src.api.v1.pointcloud")`
1. src/api/__init__.py:522-530 `_include_router(v1_router, name="pointcloud", module=pointcloud, prefix="/pointcloud")`
1. src/api/__init__.py:532 `api_router.include_router(v1_router)`
1. src/main.py:410 `app.include_router(api_router, prefix="/api")` -> POST /api/v1/pointcloud/{classify,features,similar}
1. src/api/v1/pointcloud.py:106/:122/:141 route handlers call `_get_analyzer()`
1. src/api/v1/pointcloud.py:31-36 `_get_analyzer()` -> `_analyzer = PointNet3DAnalyzer(model_path=os.getenv("POINTNET_MODEL_PATH", None))`
1. src/ml/pointnet/inference.py:76 `__init__` -> `self._try_load_model()`
1. src/ml/pointnet/inference.py:82-128 `_try_load_model()`: gates :84/:88; `checkpoint = torch.load(...)` (:108-110); classifier loaded (:112-120); `self._feature_extractor = PointNetFeatureExtractor(feature_dim=self.feature_dim)` (:124-126); LOAD CALL `if "extractor_state_dict" in checkpoint: self._feature_extractor.load_state_dict(checkpoint["extractor_state_dict"])` (:127-130)

**Notes:** Lines 127-130, sole call inside its own `if` guard (:127), independent of the classifier's 3-way branch above it. Reads the same checkpoint object produced by torch.load#0 (still single-file). Checkpoint-content-conditioned (not always-fires), unlike the mutually-exclusive #0/#1/#2 group where exactly one always fires. Same caller chain, diverging only at the final hop. Verified against source.

</details>

**Verifier corrections (1):**
- `src/ml/pointnet/inference.py::PointNet3DAnalyzer._try_load_model::torch.load#0` — Rewrote env_conditions (family-wide; the same refinement propagated to all 5 pointnet sites, which share the HAS_TORCH + POINTNET_MODEL_PATH gate set). Corrected the torch-packaging characterization after independently reading BOTH Dockerfiles rather than inheriting the auditor's text. Kept the LIVE

### vision3d-uvnet — LIVE×2

| Site | Label | Trigger | kind | activation id | conf. |
|---|---|---|---|---|---|
| `UVNetEncoder._load_model::torch.load#0` | **LIVE** | lazy-first-request | single-file | `vision3d-uvnet/main` | high |
| `UVNetEncoder._load_model::load_state_dict#0` | **LIVE** | lazy-first-request | single-file | `vision3d-uvnet/main` | high |

<details><summary><code>src/ml/vision_3d.py::UVNetEncoder._load_model::torch.load#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request

**Env conditions:** Requires HAS_TORCH==True (numpy, torch, and src.ml.train.model.UVNetGraphModel all importable; guarded by try/except ImportError at vision_3d.py:14-23; torch ships only in optional requirements-l3.txt line 2, `torch>=2.0.0; platform_machine=="x86_64"`). Requires a checkpoint file to exist at UVNET_MODEL_PATH (env var, default "models/uvnet_v1.pth"; VERIFIED ABSENT in this checkout -- deployment/volume-provisioned, same pattern by which v6.pt/v2.pt/extratrees ARE present in models/), else vision_3d.py:185-190 `if not os.path.exists(self.model_path): ... return` fires before reaching torch.load. Requires the /analyze request to carry file_format in {step,stp,iges,igs}, AnalysisOptions.extract_features truthy (Form default options JSON sets extract_features:true), a successful geo_engine.load_step() returning a truthy shape (feature_pipeline.py:88), and a 3D feature-cache MISS for that content hash (a cache HIT at line 82-84 skips the encoder branch). Endpoint additionally requires a valid API key via Depends(get_api_key) (imported from src.api.dependencies at analyze.py:15; authn, not a logical-reachability gate). Per mission's ratified rule these file-present / optional-import gates keep the site LIVE-but-env-conditioned, they do not demote it.

**Caller chain (top-down):**

1. Dockerfile:90/109/152 CMD ["python","-m","uvicorn","src.main:app",...] -- production ASGI app is src.main:app (re-verified: 3 occurrences)
1. src/main.py:20 `from src.api import api_router` -- eager import at process start
1. src/main.py:410 app.include_router(api_router, prefix="/api")
1. src/api/__init__.py:243 analyze = _import_router("analyze", "src.api.v1.analyze") -- eager import of src/api/v1/analyze.py at process-start (importlib.import_module inside _import_router)
1. src/api/__init__.py:288-296 _include_router(v1_router, name="analyze", module=analyze, prefix="/analyze", ...) -- mounts analyze.router under v1_router at /v1/analyze
1. src/api/__init__.py:532 api_router.include_router(v1_router) -- v1_router has prefix "/v1", so full path is /api/v1/analyze
1. src/api/v1/analyze.py:88 @router.post("/", response_model=AnalysisResult) async def analyze_cad_file(...) -- POST /api/v1/analyze/
1. src/api/v1/analyze.py:110-123 return await run_analysis_live_pipeline(..., run_feature_pipeline_fn=run_feature_pipeline, ...)
1. src/core/analysis_live_pipeline.py:101 feature_context = await run_feature_pipeline_fn(extract_features=..., file_format=..., content=..., ...)
1. src/core/feature_pipeline.py:73 if extract_features and file_format in ["step","stp","iges","igs"]:
1. src/core/feature_pipeline.py:87-91 shape = geo_engine.load_step(content, ...) (line 88); if shape: (line 88) ... encoder = (encoder_3d_factory or _default_encoder_3d_factory)() (line 91) -- reached on 3D feature-cache miss with a successful STEP/IGES shape
1. src/core/feature_pipeline.py:44-47 _default_encoder_3d_factory(): `from src.ml.vision_3d import get_3d_encoder` (line 45, function-local) `return get_3d_encoder()` (line 47) -- FIRST import of src.ml.vision_3d triggers module body execution
1. src/ml/vision_3d.py:376 _encoder = UVNetEncoder() -- module-level singleton constructed on first import
1. src/ml/vision_3d.py:164-180 UVNetEncoder.__init__: if HAS_TORCH: ...(device selection)... self._load_model() (line 180)
1. src/ml/vision_3d.py:182-196 UVNetEncoder._load_model(): after `if not os.path.exists(self.model_path): return` (lines 185-190), checkpoint = torch.load(self.model_path, map_location=self.device) (line 196)  <-- TARGET SITE

**Notes:** ADVERSARIALLY RE-VERIFIED, label CONFIRMED LIVE. Every hop read directly in source. Key refutation attempts, all failed: (1) Is vision_3d imported at app-startup? NO -- repo-wide grep of src/ shows exactly two references to src.ml.vision_3d: the function-local lazy import at feature_pipeline.py:45 (live path) and readiness_registry.py:285. (2) Does the startup lifespan trigger the load? main.py:146 lifespan -> main.py:183 await load_models() -> loader.py:22 build_model_readiness_snapshot() -> readiness_registry.py:371 _uvnet_item() -> readiness_registry.py:285 _module_object("src.ml.vision_3d","_encoder"). But _module_object (readiness_registry.py:93-97) is `module = sys.modules.get(module_name); if module is None: return None; return getattr(module, attr_name, None)` -- a bare read of sys.modules, it NEVER imports. So startup readiness reporting cannot construct the singleton or trigger torch.load. (3) Dynamic import? readiness_registry's only importlib use is importlib.util.find_spec("torch") at line 88 (spec probe, not a vision_3d import). So the ONLY production trigger is the lazy function-local import at feature_pipeline.py:45, fired by the /analyze STEP/IGES encoder branch. Trigger is genuinely lazy-first-request, NOT app-startup. Paired 1:1 with the load_state_dict#0 site -- both calls load the SAME single checkpoint file inside one _load_model() try block (torch.load reads the file at :196, load_state_dict applies checkpoint["model_state_dict"] at :220). This is ONE file via two sequential calls -> kind='single-file', NOT the two-distinct-files open-question pattern the mission calls out (that pattern is V16Classifier v6.pt+v14.pt / PartClassifierV16 two files). Load fires exactly once per process (subsequent requests reuse the cached singleton via get_3d_encoder()). in_phase_a_denominator kept true per literal per-site rule (label==LIVE); logical dedup to one activation is downstream via shared proposed_logical_activation_id.

</details>

<details><summary><code>src/ml/vision_3d.py::UVNetEncoder._load_model::load_state_dict#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request

**Env conditions:** Same chain and same gates as torch.load#0 (they execute sequentially inside one _load_model() try block): HAS_TORCH==True (optional torch, requirements-l3.txt, platform_machine=="x86_64"); checkpoint present at UVNET_MODEL_PATH (default "models/uvnet_v1.pth", VERIFIED ABSENT in checkout -> deployment/volume-provisioned); /analyze request with file_format in {step,stp,iges,igs}; extract_features truthy; successful geo_engine.load_step() shape; 3D feature-cache miss; valid API key via Depends(get_api_key). This call additionally requires the earlier statements in the same try block to have succeeded: torch.load (line 196) returned a dict, and UVNetGraphModel(...) construction. Any failure earlier in the try raises and is caught by `except Exception as e:` at vision_3d.py:226-228 (sets self._load_error, logs, never re-raises to caller), so load_state_dict is skipped on such failure. These are runtime provisioning conditions, not chain breaks -> LIVE-but-env-conditioned per the ratified rule.

**Caller chain (top-down):**

1. Dockerfile:90/109/152 CMD python -m uvicorn src.main:app -- production ASGI app (re-verified 3 occurrences)
1. src/main.py:20 `from src.api import api_router` (eager) -> src/main.py:410 app.include_router(api_router, prefix="/api")
1. src/api/__init__.py:243 analyze = _import_router("analyze", "src.api.v1.analyze") -- eager import at process-start
1. src/api/__init__.py:288-296 _include_router(v1_router, name="analyze", module=analyze, prefix="/analyze", ...)
1. src/api/__init__.py:532 api_router.include_router(v1_router) (v1_router prefix "/v1")
1. src/api/v1/analyze.py:88 @router.post("/") async def analyze_cad_file(...) -- POST /api/v1/analyze/
1. src/api/v1/analyze.py:110-123 return await run_analysis_live_pipeline(..., run_feature_pipeline_fn=run_feature_pipeline, ...)
1. src/core/analysis_live_pipeline.py:101 feature_context = await run_feature_pipeline_fn(...)
1. src/core/feature_pipeline.py:73 if extract_features and file_format in ["step","stp","iges","igs"]:
1. src/core/feature_pipeline.py:87-91 geo_engine.load_step(...) returns truthy shape -> encoder = (encoder_3d_factory or _default_encoder_3d_factory)() (line 91)
1. src/core/feature_pipeline.py:44-47 _default_encoder_3d_factory(): `from src.ml.vision_3d import get_3d_encoder` (line 45); `return get_3d_encoder()` (line 47) -- first import triggers module body
1. src/ml/vision_3d.py:376 _encoder = UVNetEncoder() -- module-level singleton on first import
1. src/ml/vision_3d.py:164-180 UVNetEncoder.__init__: if HAS_TORCH: ... self._load_model() (line 180)
1. src/ml/vision_3d.py:194-220 UVNetEncoder._load_model(): checkpoint = torch.load(...) (line 196); self.model = UVNetGraphModel(...) (line ~208-218); self.model.load_state_dict(checkpoint["model_state_dict"]) (line 220)  <-- TARGET SITE

**Notes:** ADVERSARIALLY RE-VERIFIED, label CONFIRMED LIVE. Same module-level-singleton-on-lazy-import mechanism as torch.load#0 (see that site's notes for the full startup-does-not-import proof: readiness_registry._module_object is a bare sys.modules.get, never imports; the only src/ import of vision_3d is the function-local feature_pipeline.py:45). This call and torch.load#0 are two calls loading ONE checkpoint file (uvnet_v1.pth) within a single _load_model() invocation -> single logical activation, shared proposed_logical_activation_id "vision3d-uvnet/main", kind='single-file'. NOT the two-distinct-files open-question pattern (V16Classifier v6.pt+v14.pt / PartClassifierV16). Both LIVE sites kept in_phase_a_denominator=true per literal per-site rule; downstream dedup by shared logical-activation id collapses them to one activation.

</details>

### embedding — LIVE×1 / LATENT×1 / OFFLINE×1

| Site | Label | Trigger | kind | activation id | conf. |
|---|---|---|---|---|---|
| `EmbeddingProvider._init_model::ctor:SentenceTransformer#0` | **LIVE** | lazy-first-request: fires on the first POST /api/v1/assistant/query after process start (t | bundle | `embedding/assistant-retriever-minilm-l12` | high |
| `SentenceTransformerProvider._load_model::ctor:SentenceTransformer#0` | **LATENT** | none -- no caller in src/ or scripts/ ever reaches this load call under any condition | bundle | `embedding/semantic-retrieval-sentence-transformer` | high |
| `DomainEmbeddingModel._try_load_sentence_transformer::ctor:SentenceTransformer#0` | **OFFLINE** | CLI-only: scripts/train_domain_embeddings | bundle | `embedding/domain-manufacturing-v2` | high |

<details><summary><code>src/core/assistant/embedding_retriever.py::EmbeddingProvider._init_model::ctor:SentenceTransformer#0</code> — LIVE (evidence)</summary>

**Trigger:** lazy-first-request: fires on the first POST /api/v1/assistant/query after process start (three nested singleton caches -- api/v1/assistant.py's module-level _assistant_instance, embedding_retriever.py's module-level _semantic_retriever, and the per-CADAssistant KnowledgeRetriever._semantic_retriever -- make it a one-time load per process). Reached on every default-mode query because KnowledgeRetriever's default RetrievalMode is HYBRID and QueryRequest exposes no mode field to opt out.

**Env conditions:** (1) Default RetrievalMode.HYBRID always exercises the semantic branch; QueryRequest has no mode field, so every /query call reaches this path -- no opt-in needed. (2) Load call is wrapped in try/except ImportError (embedding_retriever.py:56-64). VERIFIED against the Dockerfile: all three CMD stages that run `uvicorn src.main:app` (production line 37-90, development line 95-109, gpu line 114-152) install only requirements.txt (dev stage adds requirements-dev.txt); `sentence-transformers` is absent from both files and appears only as a *commented-out* optional line in requirements-assistant.txt (`# sentence-transformers>=2.2.0`), which the Dockerfile never installs at all. So in the as-shipped image the ImportError branch fires every time, _fallback_mode=True, and the SentenceTransformer(...) ctor itself does NOT execute -- the provider silently serves n-gram TF-IDF vectors instead. The call graph is fully live and reached by default on every query; only the (currently absent) optional dependency gates whether the ctor actually runs. (3) The /assistant router is registered best-effort (required=False, unlike health's required=True), but its own top-level imports are lightweight (fastapi/pydantic/stdlib only; the heavy `src.core.assistant` import is deferred inside get_assistant()), so normal registration is not fragile.

**Caller chain (top-down):**

1. src/main.py:410 app.include_router(api_router, prefix="/api")  [VERIFIED ANCHOR]
1. src/api/__init__.py:266 assistant = _import_router("assistant", "src.api.v1.assistant")  (best-effort, required=False)
1. src/api/__init__.py:494-501 _include_router(v1_router, module=assistant, prefix="/assistant", ...)
1. src/api/__init__.py:532 api_router.include_router(v1_router)  -> final route POST /api/v1/assistant/query
1. src/api/v1/assistant.py:463-464 @router.post("/query") async def query_assistant(request: QueryRequest)
1. src/api/v1/assistant.py:482 assistant = get_assistant()
1. src/api/v1/assistant.py:155-165 get_assistant(): lazily constructs CADAssistant(config=config) on first call
1. src/core/assistant/assistant.py:123 CADAssistant.__init__: self._knowledge_retriever = KnowledgeRetriever()  (mode defaults to RetrievalMode.HYBRID, knowledge_retriever.py:81)
1. src/api/v1/assistant.py:486-491 response = assistant.ask_with_context(...) [if request.context] else assistant.ask(request.query)
1. src/core/assistant/assistant.py:234-237 (ask()) / :307 (ask_with_context()): results = self._knowledge_retriever.retrieve(analyzed, max_results=...)  -- no mode override, effective_mode=HYBRID
1. src/core/assistant/knowledge_retriever.py:141-142 retrieve(): effective_mode in [SEMANTIC,HYBRID] -> semantic_results = self._retrieve_by_semantic(query)
1. src/core/assistant/knowledge_retriever.py:178-182 _retrieve_by_semantic(): semantic_retriever = self._get_semantic_retriever()
1. src/core/assistant/knowledge_retriever.py:104-109 _get_semantic_retriever(): lazy import + call get_semantic_retriever()
1. src/core/assistant/embedding_retriever.py:535-539 get_semantic_retriever(): lazy singleton -> _semantic_retriever = SemanticRetriever(config)
1. src/core/assistant/embedding_retriever.py:281-283 SemanticRetriever.__init__: self.embedding_provider = EmbeddingProvider(self.config)
1. src/core/assistant/embedding_retriever.py:48-52 EmbeddingProvider.__init__: self._init_model()
1. src/core/assistant/embedding_retriever.py:54-59 _init_model(): from sentence_transformers import SentenceTransformer; self._model = SentenceTransformer(self.config.model_name)  <-- LOAD CALL

**Notes:** Seed hypothesis ('SentenceTransformer -- reached via the assistant') is directionally correct and is now backed by a fully-read, hop-by-hop chain terminating at POST /api/v1/assistant/query (default HYBRID mode). Important correction to the seed: the actual load is defeated in the as-shipped Docker image because sentence-transformers is not installed (commented out even in the assistant-specific requirements file, which itself is never installed) -- so today this is 'LIVE but currently always falls into its own fallback', not a confirmed-executing model load. Per the ratified rule this is still LIVE/env-conditioned (try/except optional import is explicitly named as an env-condition case, not a LATENT case), but the operational reality (fallback-only in current image) should weigh into any Phase-A remediation/verification priority. Also worth flagging for the design-lock owners: embedding_retriever.py:283 and knowledge_retriever.py's KnowledgeRetriever are a completely separate, parallel implementation from the same-named SemanticRetriever/EmbeddingProvider classes in semantic_retrieval.py (site #2's file) -- two unrelated class hierarchies share names across the two files, which likely explains why the seed hypotheses for sites #1 and #2 read almost identically despite very different reachability outcomes.

</details>

<details><summary><code>src/core/assistant/semantic_retrieval.py::SentenceTransformerProvider._load_model::ctor:SentenceTransformer#0</code> — LATENT (evidence)</summary>

**Trigger:** none -- no caller in src/ or scripts/ ever reaches this load call under any condition.

**Env conditions:** Doubly gated even in the hypothetical case where SimilarityTool.execute() were somehow invoked: (a) create_semantic_retriever() must be called with use_transformers=True (no such call exists in src/ or scripts/; the sole real call site uses defaults); (b) constructing SentenceTransformerProvider alone does not call _load_model() -- only SemanticRetriever.__init__'s eager `.dimension` property access (semantic_retrieval.py:400) or a subsequent embed_text()/embed_batch() call would trigger it. tests/unit/assistant/test_semantic_retrieval.py exercises SentenceTransformerProvider directly but that is test-only, not a production or script caller.

**Caller chain (top-down):**

1. src/core/assistant/tools/__init__.py:24 TOOL_REGISTRY = {..., SimilarityTool(), ...}  (construction only, at import time; does not execute the tool)
1. src/core/assistant/tools/similarity_tool.py:43 async def execute(self, params): ...  -- NEVER INVOKED: TOOL_REGISTRY has exactly two consumers in src/ -- FunctionCallingEngine (function_calling.py:91 self._tools=dict(TOOL_REGISTRY); :372 tool.execute(params)) and AnalysisReportGenerator (report_generator.py:22 self._tools=dict(TOOL_REGISTRY)). FunctionCallingEngine is never instantiated anywhere in src/ (grep across all of src/ for 'FunctionCallingEngine(' returns zero hits outside its own class definition); its only other repo appearance is scripts/run_performance_baseline.py:299-316 _bench_function_calling_init(), which benchmarks only __init__(llm_provider='offline') timing and never drives a tool-use loop or calls .execute() on anything. AnalysisReportGenerator is never instantiated anywhere in src/, and even its own generate_full_report() (report_generator.py:34-38) only calls the classify_part/extract_features/recommend_process/estimate_cost/assess_quality tools by literal key -- 'search_similar' (SimilarityTool.name) is not among them.
1. src/core/assistant/tools/similarity_tool.py:55,58 (hypothetical, unreached) from .semantic_retrieval import create_semantic_retriever; retriever = create_semantic_retriever()  -- called with NO arguments, so use_transformers=None
1. src/core/assistant/semantic_retrieval.py:675 if provider is None and use_transformers:  -- False when use_transformers is None/falsy, so this branch is skipped even in the hypothetical case above
1. src/core/assistant/semantic_retrieval.py:677 provider = SentenceTransformerProvider(model_name=model_name)  -- this is the ONLY construction site of SentenceTransformerProvider anywhere in src/; it only executes when a caller explicitly passes create_semantic_retriever(use_transformers=True), which no caller anywhere in src/ or scripts/ does (grep for 'use_transformers' outside semantic_retrieval.py itself returns zero hits in src/)
1. src/core/assistant/semantic_retrieval.py:156-162 SentenceTransformerProvider._load_model(): from sentence_transformers import SentenceTransformer; self._model = SentenceTransformer(self.model_name, device=self.device)  <-- LOAD CALL, itself only invoked via the .dimension property, embed_text(), or embed_batch() -- never from the ctor directly

**Notes:** Seed hypothesis ('SentenceTransformer -- reached via the assistant') does not hold up: this class lives in a completely separate, parallel SemanticRetriever/EmbeddingProvider implementation (semantic_retrieval.py) from the one actually wired into the assistant (embedding_retriever.py, site #1) -- the two files share class names (EmbeddingProvider, SemanticRetriever) but are unrelated hierarchies. The only would-be entry point (SimilarityTool, a function-calling tool) is fully wired end-to-end in source but has zero live invokers anywhere in the repo: its two possible drivers (FunctionCallingEngine, AnalysisReportGenerator) are each defined but never instantiated in src/, and the one script that references FunctionCallingEngine only times its constructor, never runs a tool-use loop. This is squarely gate-before-wired/dormant code, matching LATENT rather than OFFLINE (no script anywhere actually reaches this specific load call, unlike site #3).

</details>

<details><summary><code>src/ml/embeddings/model.py::DomainEmbeddingModel._try_load_sentence_transformer::ctor:SentenceTransformer#0</code> — OFFLINE (evidence)</summary>

**Trigger:** CLI-only: scripts/train_domain_embeddings.py's argparse-invoked demo() path (dispatched by the `--demo` flag -- argparse `action="store_true"`, routed via `if args.demo: demo(args)` at line 324, NOT an argparse subcommand) constructs DomainEmbeddingModel directly (with or without a model_path). This is a training/demo script run manually (`python scripts/train_domain_embeddings.py --demo`), never part of the ASGI app.

**Env conditions:** Requires manual CLI invocation of scripts/train_domain_embeddings.py's --demo flag (`python scripts/train_domain_embeddings.py --demo`); requires sentence-transformers installed in whatever environment runs the script (not verified against any requirements file used by that script, but irrelevant to the OFFLINE determination since the script itself is never run by the service). Separately, and unlike site #1's fallback, this constructor is not wrapped by a caller that silently no-ops on failure at the DomainEmbeddingModel level for the script's own control flow -- _try_load_sentence_transformer itself still catches ImportError/Exception and returns False, letting _load() fall through to the TF-IDF fallback (src/ml/embeddings/model.py:67-74) if sentence-transformers is absent.

**Caller chain (top-down):**

1. scripts/train_domain_embeddings.py:229 def demo(args): ...  (argparse CLI entry point, run only via direct script invocation, never imported/reached from src/main.py or src/api/)
1. scripts/train_domain_embeddings.py:238 model = DomainEmbeddingModel(model_path=model_path)  [if a fine-tuned model dir exists]  OR  scripts/train_domain_embeddings.py:241 model = DomainEmbeddingModel()  [fallback]
1. src/ml/embeddings/model.py:36-42 DomainEmbeddingModel.__init__: self._load(model_path)
1. src/ml/embeddings/model.py:48-65 _load(): tries _try_load_sentence_transformer(model_path, fine_tuned=True) first when model_path is a dir, else falls through to _try_load_sentence_transformer(_BASE_MODEL_NAME, fine_tuned=False)
1. src/ml/embeddings/model.py:76-82 _try_load_sentence_transformer(): from sentence_transformers import SentenceTransformer; self._model = SentenceTransformer(name_or_path)  <-- LOAD CALL

**Notes:** Seed hypothesis ('DomainEmbeddingModel SentenceTransformer -- production embedding path') OVERCLAIMS: this class is never reached from the production service. Its only production-adjacent wiring is DomainEmbeddingProvider (src/core/assistant/domain_embedding_provider.py:73, the sole other DomainEmbeddingModel(...) construction site in src/), which is itself constructed only inside create_semantic_retriever() (semantic_retrieval.py:652), which is called only from SimilarityTool.execute() (tools/similarity_tool.py:58) -- and SimilarityTool.execute() is never invoked from production, by the identical dead-end proof given for site #2 (FunctionCallingEngine/AnalysisReportGenerator never instantiated in src/). The lazy `from src.ml.embeddings.model import DomainEmbeddingModel` inside DomainEmbeddingProvider.__init__ never fires in the served app, so the enclosing module is genuinely never imported by production -- scripts-only. So the service-side path is dead (same LATENT-shaped gap as site #2), but this specific load call IS demonstrably reached -- just only by scripts/train_domain_embeddings.py's demo() CLI, a training/eval-style script per the OFFLINE definition. Also worth flagging to the L3 owners independent of reachability: domain_embedding_provider.py's own module docstring documents a previously-fixed 'TF-IDF laundering' bug (the provider used to report available=True without disclosing the underlying model was serving an untrained TF-IDF fallback because models/embeddings/manufacturing_v2/ ships no encoder weights) -- that disclosure fix is orthogonal to this reachability audit but is directly adjacent code in the same file.

</details>

**Verifier corrections (1):**
- `src/ml/embeddings/model.py::DomainEmbeddingModel._try_load_sentence_transformer::ctor:SentenceTransformer#0` — Fixed the invocation wording in `trigger` and `env_conditions`: the auditor described a `demo` argparse subcommand (`python scripts/train_domain_embeddings.py demo ...`), but train_domain_embeddings.py has no subparser -- `demo` is exposed as the `--demo` flag (argparse `action="store_true"`), dispa

## 7. What Wave-1 does NOT do

- No loader is touched; no manifest `reason` is edited (that is a separate facts-only PR if the owner wants the corrected labels reflected there); no runtime behavior claim is made beyond static source reachability — no request was replayed, no model was loaded.
- Nothing here authorizes production enablement, model promotion, dynamic reload/retraining, Track E, or Phase B.
- The Phase-A W3 wiring scope implied by this audit is the 23-site / 11-activation LIVE set; per the ratified plan, W2 (`load_pinned_file` / `load_pinned_bundle` / baseline manifest / degraded-503) precedes wiring.
