# L3 Design-Lock — Production Identity, Fail-Closed

**Date**: 2026-07-13 · **Status**: PROPOSED (for-review; do NOT self-merge; owner ratifies)
**Rigor**: L3 (`PRODUCT_STRATEGY.md` §7.1) · **Grounded on**: `origin/main@e2facd99`
**Authority**: `PRODUCT_STRATEGY.md` §8.4 Days 0-30 ("close production-auth defaults") · **Ordering**: this is the **first runtime L3** after #509, ahead of Track E (owner-**directed** ordering 2026-07-13). **This design-lock is itself PROPOSED — the owner has NOT ratified it; "ordering" ≠ ratification of this contract.**

> **This is a proposal, not an implementation.** It changes no runtime and closes no surface. It defines
> the contract the implementation must satisfy so it can be ratified before code is written — "propose,
> don't build". Unattended routines may not author or merge this runtime surface. Solo-maintainer L3 review
> protocol per `L3_MODEL_ACTIVATION_MEMBRANE_DESIGNLOCK_20260712.md` §"Solo-maintainer L3 review protocol"
> applies verbatim (isolated critic supplies evidence; the human owner alone ratifies and authorizes a
> pinned head; `require_code_owner_reviews` stays false).

---

## 0. Why this is the P0 (empirically, at `e2facd99`)

The production authentication boundary is **fail-open by default and unconditionally so** — there is no
environment-gated guard anywhere. Verified against `origin/main@e2facd99` (and reproduced on a running
server by the owner 2026-07-13):

| # | Observed behavior | Root cause (`file:line`) |
|---|---|---|
| 1 | No `X-API-Key` → **200**, identity `"test"` | `src/api/dependencies.py:8` — `Header(default="test")`; rejects only empty |
| 2 | **Any** attacker-chosen `X-API-Key` → **200** | `dependencies.py:8-11` — returns the header value; **no comparison to any expected key** |
| 3 | Valid JWT `sub=alice` + `x-user-id=bob` → identity `user_id=bob` | `src/api/middleware/integration_auth.py:104` — `user_id = header or str(subject)`; **tenant is checked (`:95`), user is not** |
| 4 | Default `ADMIN_TOKEN=test` lets a caller's `path`+`force` reach the model loader | `dependencies.py:38` `os.getenv("ADMIN_TOKEN","test")` + `src/api/v1/model.py:46,71` → `reload_model(payload.path, force=…)` |
| 5 | `INTEGRATION_AUTH_MODE` default `disabled` → **all auth skipped**; identity set from **raw headers** | `src/core/config.py:27` / `config/__init__.py:50` default `"disabled"`; `integration_auth.py:56-58` skip + `_set_state_from_headers` |
| 6 | JWT accepted with **no `exp` / no `aud` / no `iss`** | `integration_auth.py:84` `jwt.decode(token, secret, algorithms=[alg])` — no `audience=`, no `issuer=`, no `require:["exp"]` |
| 7 | **A test locks in the vulnerability** | `tests/unit/test_integration_auth_middleware.py:111` asserts `user_id == "user-header"` (spoofed); `:112` asserts `auth_subject == "user-1"` |

There is **no** startup/runtime guard that inspects `ENVIRONMENT`/`APP_ENV=production` to reject default
`test` credentials or `disabled` auth — `src/main.py:61-141` (`_validate_optional_feature_flags`, the only
startup validator) checks GRAPH2D/FUSION flags only; `config/feature_flags.py:79-81` merely `warnings.warn`,
checks emptiness not the `test` literal, and has no production condition. **Fail-open is unconditional, not
merely under-tested.** #509 stopped contaminated-eval-driven retrain, but this identity boundary and the
caller-path reload are live.

Scope — precise (corrected by the isolated critic 2026-07-13): there is **no live consumer of a spoofed
identity** today. The `request.state.user_id` sink's only reader (`create_api_actor_from_request`,
`audit/service.py:531`) is uncalled; and the three raw-`x-user-id` readers cited in an earlier draft are
**all dormant** — `audit/logger.py:614` sits in the **unmounted** `AuditMiddleware`,
`feature_flags/decorators.py:136` in the **unmounted** `FeatureFlagMiddleware`, and
`request_context/__init__.py:213` (`RequestContext.from_headers`) has **no `src` caller**. Only CORS /
TrustedHost / `IntegrationAuthMiddleware` are mounted (`src/main.py:395-407`). The defect is nonetheless a
**fix-now latent trap**, not an active exploit: the **live** `IntegrationAuthMiddleware` sets
`request.state.user_id` from the header (`integration_auth.py:104`) and via `_set_state_from_headers`
(`:113-122`), **and a test asserts that spoofed contract** — so it becomes exploitable the instant any
consumer is wired. This lock closes it now and does **not** claim a live exploit. (Earlier phrasing said
"read live by …" — that was wrong; the readers are dormant.)

---

## 1. Invariants the implementation must establish (the contract)

### A. No credential grants access by default. Any/attacker-chosen API key must NOT pass.
- `get_api_key` must compare against a **configured expected key set** (not merely non-empty). Unset / `test` /
  unknown value → **401**. There is **no default value that authenticates**.
- Applies to `X-API-Key` (fixes #1/#2) and the admin path (`ADMIN_TOKEN` must not authenticate at its `test`
  default — fixes #4).

### B. Production mode is explicit and fail-closed.
- **Decidable production signal, precedence defined:** treat the deployment as **production UNLESS**
  `ENVIRONMENT`/`APP_ENV` is exactly `development`/`test`; any conflicting/unknown value resolves to
  **production** (fail-closed). NB the implementation MUST pick one canonical rule and apply it everywhere,
  because today these disagree: `ENVIRONMENT` defaults to `development` by *absence* in
  `observability/__init__.py:188` / `metrics.py:94` / `tracing.py:101`, while `APP_ENV=production` is baked
  into the image (`Dockerfile:50`, `docker-compose.yml:17`, which also configures **no** creds).
- When production, the service **refuses to boot** if any of: `X-API-Key` unconfigured or `test`; `ADMIN_TOKEN`
  unset/`test`; `INTEGRATION_AUTH_MODE=disabled`; or — decidable rule — `mode=required` without
  secret+audience+issuer, or `mode=optional` with any `INTEGRATION_*` set but no secret (fixes #1/#4/#5;
  replaces the undecidable "JWT config absent while integration is expected").
- **The permissive dev/test defaults require an explicit opt-in AND a harness migration in the SAME change**
  (load-bearing — without it the whole test suite + ≥5 CI workflows go red, since none sets any signal today):
  set the opt-in (`ENVIRONMENT=development`) **autouse** in `tests/conftest.py` (and add it to
  `_ENV_VARS_TO_ISOLATE`); set it in every server-booting workflow (`ci.yml`, `self-check.yml`,
  `observability-checks.yml`, `ci-tiered-tests.yml`, `stress-tests.yml`); document the now-required production
  credentials in `docker-compose.yml` / `.env.example` / README. A golden asserts **pytest *without* the
  opt-in fails closed**, so the wiring is deliberate, not accidental.

### C. JWT must carry and verify issuer, audience, and expiry.
- `jwt.decode(..., audience=<INTEGRATION_JWT_AUDIENCE>, issuer=<INTEGRATION_JWT_ISSUER>,
  options={"require": ["exp","iat","sub","tenant_id"]})`. Missing/wrong `aud`, `iss`, or `exp` → **401**
  (fixes #6). A token that never expires is not accepted.
- **New config keys:** add `INTEGRATION_JWT_AUDIENCE` / `INTEGRATION_JWT_ISSUER` to `src/core/config.py`
  (they exist nowhere today) and to B's production boot checklist. These requirements apply in `required` mode
  and whenever a token is presented in `optional` mode.
- Hardening (note, not blocking): prefer an **asymmetric alg** (RS256/ES256, verify-only public key) so the
  service holds no signing-capable secret; if HS256 remains, the secret is production-required (per B) and
  never defaulted.

### D. The actor is derived ONLY from the validated token `sub`.
- `request.state.user_id` (and every identity used for authz **or** attribution) = the validated `sub`. The
  `x-user-id` header **never** sets or overrides identity (fixes #3). Collapse the `user_id` (spoofable) /
  `auth_subject` (authentic) duplication into one authenticated identity.
- **Fail-first flips the locked contract, concretely** (so it does not collide with C/E): rewrite
  `test_required_valid_token_sets_state` to (a) mint a token carrying `exp`/`iat`/`aud`/`iss` matching the new
  settings **and** `sub=user-1`, (b) send it **without** a conflicting `x-user-id` header, and assert
  `user_id == "user-1"` on 200. The mismatching-header case becomes a **separate** test asserting **401** (per
  E). The old `:111` assertion (`user_id == "user-header"`) and any other test encoding the override are
  updated in the same change.

### E. Forged identity headers are rejected, never trusted.
- In authenticated (`required`) mode, an `x-user-id`/`x-tenant-id`/`x-org-id` header that **disagrees** with a
  token claim → **401** (tenant already does this at `:95`; extend to user; org already at `:98`). A header may
  never *establish* identity when a token is present.
- In non-authenticated paths (`disabled`, public, `optional`-without-token), identity headers must **not**
  populate a trusted identity: `_set_state_from_headers` must not set an authenticated `user_id`/`tenant_id`
  from raw headers (fixes #5). Untrusted hints, if kept at all, must be clearly non-authoritative and unusable
  for authz/attribution.

### F. Identity readers use the validated identity, not the raw header (forward guard).
- These readers are **dormant today** — `audit/logger.py:614` (unmounted `AuditMiddleware`),
  `feature_flags/decorators.py:136` (unmounted `FeatureFlagMiddleware`), `request_context/__init__.py:213`
  (`from_headers`, no `src` caller), `audit/service.py:531` (uncalled). **Requirement:** each must be
  **fixed-or-deleted before it is ever mounted/wired** — a mounted reader derives identity from validated
  request state (set by D), never `request.headers.get("x-user-id")`. This is a forward guard, not an
  active-exploit fix (see §0).
- **`RequestContext.from_headers` special case:** a unit test (`test_enterprise_p52_p55.py:961-983`) asserts it
  reads `X-User-ID` → `ctx.user_id`. The lock must pick one: (a) add it to the fail-first flip inventory, or
  (b) **exempt** it as an internal trace-propagation utility (it also carries `X-Request-ID`/`X-Trace-ID`
  between trusted services and has no access to a validated token). **Default: exempt-and-document** — it is
  not an authenticated-identity path; do not silently break its test.

---

## 2. Fail-first golden matrix (observed-RED, executed; production mode unless noted)

| Case | Required result | Today (`e2facd99`) |
|---|---|---|
| no `X-API-Key` | **401** | 200, id=`test` |
| attacker-chosen `X-API-Key` | **401** | 200 |
| `ADMIN_TOKEN` unset (default `test`) on `/model/reload` | **refuse (boot or 401)** | reachable with `X-Admin-Token: test` |
| `INTEGRATION_AUTH_MODE=disabled` in production | **boot refuses** | silently skips all auth |
| valid JWT `sub=alice`, **no** conflicting id header | identity = **alice** (from `sub`) | override path unexercised |
| valid JWT `sub=alice` + `x-user-id=bob` (mismatch, `required` mode) | **401** (per E) | **200**, id=`bob` (locked by a test) |
| JWT with **no `exp`** | **401** | accepted |
| JWT with wrong / missing `aud` | **401** | accepted (no aud check) |
| JWT with wrong / missing `iss` | **401** | accepted (no iss check) |
| forged `x-tenant-id` ≠ token | 401 | 401 (keep) |
| identity/audit reader **if/when mounted** (all such readers are **dormant** today, §0/§F) | derives actor from validated `sub`, never the header | **no live reader today** — forward guard (F), not a live-today assertion |
| dev/test with explicit insecure opt-in | unchanged (permissive) | permissive |
| production with all creds configured non-default | serves normally | — |

Each RED must be produced against the running app with a positive control (a request that legitimately
succeeds), and the flipped `test_required_valid_token_sets_state` committed in the same change.

---

## 3. Ordering relationship with the activation membrane (#513)

`/model/reload` re-enablement HARD-DEPENDS on **both** gates (see #513 §3.2):
1. **This identity gate** — no default creds, explicit production, actor from `sub`, forged headers rejected.
2. **#513 Phase A** — the route is production-disabled with no env bypass **now**; #513 Phase B (proof membrane)
   later binds *which* model may load.

Neither alone is sufficient. #513 Phase A freezes the route immediately (emergency containment); this gate is
the *who-may-ask* half required before it could ever be re-opened. Build order (owner-**directed** 2026-07-13;
none of these locks is yet ratified): **#513 Phase A freeze → this identity gate → Track E → #513 Phase B (pilot-gated).**

## 4. Non-goals / exclusions

- Not building the proof membrane or Track E here (separate locks). Not touching model-activation code.
- Not adding new authn providers (OAuth/OIDC) — this hardens the existing API-key + integration-JWT surface.
- No runtime is built by this document. Implementation lands **default-off relative to production** in the
  sense that it *adds* fail-closed checks; dev/test behavior is preserved behind an explicit opt-in.

## 5. Two-actor note (do not conflate)

This gate defends the **runtime API caller** (anonymous/forged/defaulted identity). It does **not** defend
against the code-generating routine (governed by branch protection + the solo-maintainer protocol; live
protection: 0 required approvals, 11 required strict checks, `enforce_admins` — a checks-passing PR is not
review-gated, so branch protection is not a substitute for stopping an unsafe routine). No unattended routine
may author or merge this L3 runtime.

## Appendix A — reproduction (read-only, at repo root against `origin/main@e2facd99`)

```sh
git show origin/main:src/api/dependencies.py | sed -n '8,45p'                 # get_api_key (any key) + admin default test
git show origin/main:src/api/middleware/integration_auth.py | sed -n '55,113p' # disabled=skip; jwt.decode no aud/iss/exp; user_id=header or sub
git show origin/main:tests/unit/test_integration_auth_middleware.py | sed -n '100,113p' # asserts user_id=="user-header"
git show origin/main:src/main.py | sed -n '61,141p'                          # only GRAPH2D/FUSION validated; no auth guard
git grep -n "headers.get(self.user_header)\|x-user-id\|x_user_id" origin/main -- 'src/**/*.py' # live raw-header readers
```
