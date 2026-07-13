# Phase A — retrain/promotion bypass seals — Dev & Verification (2026-07-12)

> **Posture: Safety foundation complete; retraining remains disabled.** The unconditional L3 gate
> (#509, merged `8ff94175`) guards `scripts/auto_retrain.sh` — but two sinks bypassed it. This PR
> seals them, proves zero side effects, and proves the emergency ROLLBACK path is NOT harmed.
> No Track E completion is claimed; re-enabling anything is Phase B (real §8.1.4 metrics + the
> two-stage release gate binding manifest/split digests, candidate SHA-256, evaluator version,
> thresholds) behind its own owner-gated enablement PR.

## 1. Promotion/reload sink inventory (the audit this PR is built on)

| Sink | Kind | Disposition |
|---|---|---|
| `scripts/auto_retrain.sh` | retrain + promote pipeline | already gated by the unconditional L3 gate (#509) |
| `scripts/finetune_from_feedback.py` | trains AND promotes (`reload_model(..., force=True)`) | **SEALED (this PR)** — calls the SAME unconditional gate before ANY side effect (export/train/save/reload); a gate that is missing or *returns* (subverted) also blocks |
| `POST /api/v1/model/reload` | loads an ARBITRARY model path into serving | **SEALED (this PR)** — 403 fail-closed; no payload/credential/flag opens it; pre-seal handler body removed (not dead code); Phase B reintroduces it accepting ONLY an approved-artifact-bound model hash |
| `auto_remediation._action_rollback_model` | emergency ROLLBACK to the KNOWN previous model (`_MODEL_PREV_PATH`), in-process | **NOT sealed (deliberate)** — never traverses the API route; verified by test to still work. Distinction: rollback restores a previously-served model; it does not promote a new one |
| `src/ml/classifier.reload_model` | the loader mechanism + its security validation (magic/size/hash/opcode) | unchanged; still fully covered by direct-call unit tests (route tests converted to direct calls) |
| research/offline training scripts (`train_classifier_*.py`, `finetune_graph2d_*`, `finetune_v11_on_real.py`, `finetune_agent_llm.py`, `distill_ensemble.py`, …) | save model FILES offline; do not reload/promote into serving | **REGISTERED, not blocked** (per owner ruling); they become promotion sinks only via the sealed paths above |
| `scripts/finetune_from_feedback_e2e.py` | offline demo (memory store, synthetic vectors); no reload | registered; additionally inherits the seal wherever it drives the sealed script |
| `scripts/stress_concurrency_reload.py` | stress tool that POSTs the API route | registered; now receives 403 — usable again when Phase B re-opens the route |
| `deploy_production.sh` / `deploy_staging.sh` | operator deployment (model chosen via env at startup) | registered; operator-controlled deployment, not an automated promotion path |

## 2. Changes

- **`scripts/finetune_from_feedback.py`** — first act of `main()` (before `get_active_learner`, i.e.
  before feedback export, training, save, reload): import the L3 gate and `check()`. Three
  fail-closed branches: gate raises `GateBlocked` → exit 1; gate module missing/renamed → exit 1;
  gate RETURNS (subverted — it has no pass path by construction) → "invariant breach" → exit 1.
- **`src/api/v1/model.py::model_reload`** — 403 before any loader work, with the §5.2/§8.1 pointer
  and the explicit note that rollback is unaffected. The unreachable pre-seal body (status envelope
  handling) was removed rather than left as dead code; a Phase-B note documents what to reintroduce.
- **Route tests converted, coverage preserved** — the reload security tests that exercised
  magic/size/hash/opcode/rollback semantics THROUGH the route now call `reload_model` directly
  (identical result-dict assertions), so the loader's security machinery keeps its coverage while
  the route itself is pinned to 403.

## 3. Verification (`tests/unit/test_retrain_bypass_seals.py` — 5 tests, all green)

| Proof | Test |
|---|---|
| finetune: gate fires FIRST; `get_active_learner` never reached → zero export/train/save/reload | `test_finetune_from_feedback_blocks_before_export` |
| finetune: a SUBVERTED gate (stubbed to return) still blocks — exit 1 | `test_finetune_from_feedback_distrusts_a_subverted_gate` |
| API: 403 + `reload_model` never invoked (spy) | `test_model_reload_route_is_sealed_403_and_never_loads` |
| API: no payload shape (path/None/'', force, expected_version) opens the seal | `test_model_reload_seal_has_no_payload_bypass` |
| rollback: `_action_rollback_model` still reaches `reload_model(prev_path)` in-process | `test_auto_remediation_rollback_still_works` |

Plus the three converted security-test files stay green (direct-call coverage of the loader).

## 4. What this deliberately does NOT do

No §8.1.4 metrics, no model-hash binding, no evaluator/threshold config, no enablement. Those are
Phase B, gated on a real data/model environment and owner threshold decisions, delivered as a
separate enablement PR. Until then: **Safety foundation complete; retraining remains disabled.**
