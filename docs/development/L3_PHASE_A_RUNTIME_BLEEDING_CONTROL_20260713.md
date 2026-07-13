# L3 Phase A — runtime bleeding-control — Dev & Verification (2026-07-13)

> **Posture: Producer disabled; runtime activation remains untrusted.** This is fail-closed,
> risk-REDUCING bleeding-control ONLY — it seals a reproduced live RCE surface, refuses the insecure
> default credentials in production, and makes activation-surface completeness enforced *by
> construction*. It builds **no proof membrane** and enables **nothing**: every change here can only
> refuse, never green-light an activation. Grounded on `origin/main@e2facd99`. Fresh from latest
> main — does NOT revive the closed #514. The membrane itself is the ratification-gated follow-up
> (design-lock #513).

## 0. Why (the reproduced live risk on `e2facd99`)

- `POST /api/v1/model/reload` hot-reloads an **arbitrary caller-supplied path** into the serving
  process. The reload deserializes **before** it checks: `src/ml/classifier.py:535` runs
  `pickle.loads(data)` ahead of the whitelist/hash check, and the compared hash is truncated to 16
  hex — a reproduced arbitrary-code-execution-on-load. Guarded only by `api_key` + `admin_token`
  that **both default to the literal `"test"`** (`src/api/dependencies.py`).
- A hand-maintained "these are the activation points" list has been wrong repeatedly; the real
  surface is **≥30 model-load sites across ≥9 families** (design-lock #513 §1).

## 1. Changes (all fail-closed)

### 1.A `POST /api/v1/model/reload` — SEALED (403)
`src/api/v1/model.py::model_reload` now `raise HTTPException(403, "…fail-closed…")` before any loader
work; `status_code=403`, no success `response_model`; the caller path is **not logged**. The pre-seal
status envelope (success/not_found/version_mismatch/size_exceeded/magic_invalid/hash_mismatch/
opcode_blocked/opcode_scan_error/rollback) was removed with the seal, not left as dead code. Per the
design-lock §3.2 the route may re-open ONLY when BOTH the production-identity gate AND the proof
membrane hold; neither exists, so the interim membrane default here is #509's: refuse unconditionally.
Emergency rollback runs in-process via auto-remediation and does not use this route.

### 1.B Default `"test"` credentials — fail-closed in a production posture
`get_api_key` / `get_admin_token` (`src/api/dependencies.py`) now refuse the insecure default in a
production posture (`REQUIRE_STRONG_AUTH=1`, or `ENVIRONMENT`/`APP_ENV`/`ENV` ∈
{production, prod, staging}): a missing/`"test"` `ADMIN_TOKEN` → **500 fail-closed** (refuse to
operate), and `X-API-Key: test` → **401**; an optional `API_KEY` env is validated when set. Dev/CI
leave the posture unset, so the historical `"test"` default is preserved and the suite is not bricked.

### 1.C Activation-surface enumerator — completeness by construction
`scripts/ci/activation_surface_enumerator.py` (stdlib, AST-based) discovers **every** model-load site
(`torch.load` / `pickle.load(s)` / `joblib.load` / `*.load_state_dict` / a `reload_model(` call) in
`src/` + `scripts/` and requires each to be classified in `scripts/ci/activation_surface.json`
(`gated | producer | offline | unmounted | infra`). Keys are `<file>::<enclosing symbol>::<kind>#<n>`
— **stable across line-number drift** (AST, not grep-by-line). A **new un-annotated load site reds
CI**, so a new activation surface cannot land silently. It is discovery + fail-closed bookkeeping
only: it can never emit a "green that enables" — it only passes (all classified) or reds.

Current classification (`e2facd99` after the seal): **114 sites** — `gated`=30, `producer`=44,
`offline`=33, `unmounted`=3, `infra`=4. The **30 gated** production-reachable activation points span
9 families: pickle-classifier, graph2d, pointnet, part, part-v16, hybrid, history, vision3d-uvnet,
anomaly-monitor. Each `gated` site MUST route through the L3 proof membrane (`verify_and_load`) once
it exists; until then the membrane default is #509's unconditional raise.

## 2. Verification (local)

| Check | Result |
|---|---|
| enumerator: current tree fully classified; `main()` exit 0 | pass |
| enumerator: a NEW unclassified load site → **RED** (exit 1) | pass |
| enumerator: a STALE manifest entry → RED; invalid class rejected; every `gated` names a family | pass |
| `/model/reload` sealed 403; `reload_model` **never** invoked (spy); no payload bypass | pass |
| default `ADMIN_TOKEN` unset/`"test"` in prod posture → **500**; `X-API-Key: test` in prod → **401** | pass |
| a strong `ADMIN_TOKEN` in prod authenticates; wrong token still 403 | pass |
| dev posture (no prod flag) preserves the `"test"` default (suite not bricked) | pass |
| OpenAPI snapshot regenerated: `/model/reload` responses = **403 + 422**, 193 paths / 198 ops | pass (CI view; local shows a known +`/metrics` delta) |
| reload route tests converted to direct `reload_model` calls — loader security coverage preserved | pass |

`tests/unit/test_activation_surface_enumerator.py` (5) + `tests/unit/test_l3_phase_a_runtime_bleeding_control.py` (7).
CI wired in `ci.yml` + `ci-tiered-tests.yml` (enumerator + these suites run as an L3 step).

## 3. What this deliberately does NOT do

No proof membrane, no `verify_and_load`, no signed proofs, no Track E, no re-enablement of any
activation. The route stays sealed and the `gated` sites stay proof-unbound (the membrane is the
ratification-gated follow-up, design-lock #513). **Producer disabled; runtime activation remains
untrusted.** For owner review — not for self-merge.
