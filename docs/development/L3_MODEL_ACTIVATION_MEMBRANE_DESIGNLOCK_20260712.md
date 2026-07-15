# L3 Design-Lock — Model-Release & Activation Proof Membrane

**Date**: 2026-07-12 (rev 2026-07-15 — canonical-strategy alignment) · **Status**: PROPOSED (for-review; do NOT self-merge; owner ratifies)
**Rigor**: L3 (`PRODUCT_STRATEGY.md` §7.1) · **Grounded on**: `origin/main@e84fea2d`; activation-map count on `origin/main@f2ebe2fa` (2026-07-14; #516 enumerator + #519 hardening merged). The import-aware CI enumerator — the count authority, NOT a hand-count — executes on current `main` as **128 sites / 38 marked `gated` / 11 families**. Its current summary still calls those 38 "production-reachable" and presents rejected option (a), blanket hard-refuse, as valid; those labels are stale and are **not** inherited here. #521 is the open, unmerged output-truth correction. This lock uses the executed count while treating `gated` as a conservative AST classification: per-site logical reachability is a **Wave-1 audit**, NOT asserted here, and several sites are latent or not-yet-proven-reachable (§1.B(cont)); a hand-count has been wrong ≥4 times.
**Authority**: `PRODUCT_STRATEGY.md` §4 (AI safety), §5.2 (evaluation integrity not release-grade),
§8.1 (Track E). Scheduled deliverable for the 7/20–7/26 week; pulled forward because the runtime
work is P0-blocked and a design-lock is a doc, not runtime.

> **This is a proposal, not an implementation.** It changes no runtime, touches no model-activation
> code, and does not itself close any surface. It defines the contract the future implementation must
> satisfy so it can be ratified before any code is written — precisely the "propose, don't build"
> mode L3 requires. Unattended routines may not author or merge this runtime surface.

### Solo-maintainer L3 review protocol

This repository currently has one human developer. That makes a second human GitHub approval
unavailable; it does **not** justify fabricating one or silently lowering L3 rigor. For this repository,
`PRODUCT_STRATEGY.md` §4.7 and §7.1 are satisfied by the following compensating protocol:

1. An isolated critic that did not author the patch first derives the load-bearing discriminators
   from this ratified design lock, then reviews the implementation. Use a different model when
   available; at minimum use a fresh context with no inherited implementation rationale.
2. The PR records the critic findings, fail-first golden, observed-RED run, positive controls for the
   verifier itself, and the exact final head reviewed. A material post-review change invalidates that
   evidence and requires another pass.
3. The critic provides **evidence, not approval**. The sole human owner explicitly ratifies the design
   and separately authorizes the pinned implementation head. A second account controlled by the same
   person is not independent review.
4. Merge, deployment, and enablement are separate decisions. The implementation lands default-off.
   **Dynamic** activation and any **re-enablement** of `/model/reload` remain blocked on the proof
   membrane, the production-identity gate, and the Track E evidence. Ratifying this lock authorizes
   building **Phase-A baseline containment**, not enabling it and not promoting a model: before Track E,
   Phase A may load only the exact already-in-service
   `(logical_activation_id, artifact_id, kind, digest)` tuple (§0.5/§5).

`.github/CODEOWNERS` may inventory this surface, but `require_code_owner_reviews` remains false while
the repository has only one developer; enabling it would create an impossible approval gate, not an
independent critic. If an independent human reviewer becomes available later, their review strengthens
this protocol but is not simulated in the meantime.

---

## 0. Why this exists (and why the first attempt was insufficient)

#509 made `scripts/auto_retrain.sh` unconditionally fail-closed. That was correct **but narrow**, and
an earlier claim that it "closed retraining on main" was **overstated**. Corrected here (second
review): `auto_retrain.sh` is a **producer** (it prints a deploy command, it does not activate a
running service), so #509 closes **none of the runtime activation points** (the import-aware CI
enumerator executes as 38 entries marked `gated` across 11 model families; this lock does not inherit
its current, stale "production-reachable" label — reachability is a **Wave-1 audit** (§1.B(cont)) —
see §1.B; a hand-count has been wrong ≥4 times, hence the CI-enumerator contract). It is
bleeding-control one layer upstream; the runtime membrane is unbuilt.

Corrections from review, all load-bearing:

1. **Model families are distinct release paths, not one path's bypasses.** `POST /api/v1/model/reload`
   hot-reloads the **pickle classifier** (`CLASSIFICATION_MODEL_PATH`, `src/ml/classifier.py:227,85`).
   `auto_retrain.sh`/#509 promotes the **Graph2D checkpoint** (`GRAPH2D_MODEL_PATH`, resolved in
   `Graph2DClassifier.__init__` (`src/ml/vision_2d.py:40-41`) and loaded via `torch.load` at
   `src/ml/vision_2d.py:136`). These are **different models with
   different activation surfaces**. The membrane must be **cross-family**.

2. **A file-reference count is a discovery list, not a boundary.** The 15 `*_MODEL_PATH` / `load` hits
   are not 15 production entry points. The acceptance goal is *"every production-reachable activation
   point passes the **phase-appropriate activation gate**"* — Phase A: a fixed-`SHA-256` pin; Phase B:
   the signed proof — **not** mechanically wrapping every `load()`.

---

## 0.5 Phasing — pin first (Phase A: static fixed-hash), prove later (Phase B: signed proof)

**PROPOSED phasing — no ratification has occurred.** This doc is for-review; the owner alone
ratifies it (no owner review/comment/pinned-head ratification exists as of this writing). The
proposal is that the membrane ships in **two separately-ratifiable phases**.
Phase A is cheap containment that needs **no** cryptographic proof store; Phase B is the full proof
membrane and is **deferred until a real pilot needs dynamic model-swap**. Building the signed store now
(Phase B) would spend ~2–3 weeks and add **no customer evidence**. The external `/model/reload` boundary is now **sealed (#516, 403)**; the remaining gap is
the **internal `gated` loaders** still unpinned (of the 38 conservatively-`gated` sites; per-site reachability is a **Wave-1 audit** and several are latent/unproven — §1.B(cont)). Pin the reachable ones first (owner decision (b)).

### Phase A — Static-artifact activation (fixed-`SHA-256` pin; build next, after **this** lock ratifies — the internal fixed-hash loaders do NOT depend on caller identity; the production-identity gate only gates a future re-open of `/model/reload`)

**Goal:** no arbitrary, caller-path, or hot-swapped model activation can occur in production — with
**no signing keys / no signed proof store** (a plain content-hash comparison only; the signed proof
is Phase B). Per the owner's decision (b), models still load from pinned server-owned artifacts, but
before Track E that means **only the exact already-in-service
`(logical_activation_id, artifact_id, kind, digest)` tuple**. Phase A is a containment migration, not
a model-promotion path.
1. **Production-disable the external `POST /api/v1/model/reload` route — no environment-variable bypass.**
   It was LIVE (§1.A/§3.2); **#516 has already sealed it (403)**. It refuses unconditionally; **no flag or env var re-opens it** (re-opening is a Phase B + identity-gate
   decision, §3.2). A request carrying a valid `path`+`force`+default `test` creds must not reach
   `reload_model`. (This route is DONE — #516.)
2. **Static fixed-hash activation at every §1.A/1.B `gated` site (owner decision (b)).** Each `gated`
   loader may activate a model **only** from a *server-owned* artifact resolved from a controlled store:
   **no caller-influenced path, no env-var path-swap, no dynamic replacement, and no network fetch at
   load** (offline-only). An unclassified/unpinned load cannot activate. This is a subset of §3's
   `verify_and_load` **without** the signed proof store (Phase B).

   **Shared bounded pre-read (Phase A AND Phase B).** Before a full file read, bundle copy, digest, or
   framework loader runs, reject an artifact whose declared KIND/type is wrong, whose single-file size
   exceeds its family limit, or whose bundle exceeds a bounded file-count, per-file-size, or aggregate-
   byte limit. Validate file type / a bounded magic prefix where the family has one; for an unpacked
   bundle, use `lstat` metadata plus only bounded prefixes, never read the full tree to decide whether it
   is safe to copy. A malformed/unreadable entry or metadata overflow is a refusal, not a best-effort
   skip. These limits are server policy, not caller or environment input.

   Two artifact KINDS — because several `gated` families are NOT single files (review 7c):

   - **single-file** (`torch.load`/`pickle.load`/`joblib.load`/`onnx.load` of one file — graph2d, part,
     hybrid, history, vision3d, pickle-classifier, anomaly-monitor): after the bounded pre-read passes,
     read the resolved bytes **once**, `SHA-256(bytes) ==` the pinned value, and load **THOSE** bytes
     (TOCTOU-safe). This is the original contract.
   - **bundle / tree** (`from_pretrained` / `SentenceTransformer` / `PaddleOCR` — ocr, embedding —
     which load a **directory of many files**, and may otherwise fetch from a network hub): the pin is
     a **deterministic, versioned tree digest** (`tree-digest-v1`) = `SHA-256` over a **canonical
     encoding** of the **sorted** list of `(posix-relpath, SHA-256(file-bytes))` for **every** file under
     the artifact root — canonical encoding fixed by the version id: UTF-8 posix relpaths, each record
     length-prefixed and NUL-delimited (`len(relpath)` · relpath · file-sha256-hex), records sorted
     bytewise by relpath — so Phase A and Phase B compute the **identical** digest for the same tree (a
     different encoding is a new version id, never a silent change). The site (a) resolves a
     server-owned artifact id to a controlled, **read-only** (access-control only — NOT a load-duration
     immutability guarantee, which is why (b) copies), **already-unpacked** local directory that has
     passed the file-count / per-file / aggregate-byte pre-read bounds —
     **never** a hub id, and with `HF_HUB_OFFLINE=1` / `local_files_only=True` so no network load can
     occur; (b) **freezes an immutable snapshot** of that directory that nothing else can mutate for
     the load's duration — a per-process **copy** (or an FS-level read-only snapshot) into a
     service-private freeze dir, **never a bind-mount of the still-mutable source** (a bind re-exposes
     the source, so a mid-load mutation would show through and NOT close the TOCTOU) — so the
     bytes that are digested are the SAME bytes the framework later reads (the tree analog of the
     single-file "hash and load the same bytes"; this is what closes the **bundle TOCTOU** — a file
     changing between digest and the framework's read is impossible on the frozen snapshot);
     (c) recomputes the tree digest over that **frozen** directory, `resolve()`-contained to the store
     root (reject any symlink/`..`/absolute escaping the root, per-file); (d) `== the pinned tree
     digest`, then hands the framework loader the **frozen snapshot's local path** (never the mutable
     original); else refuses fail-closed. A controlled unpack (if the release ships an archive) verifies
     the archive digest first and unpacks into the store with path-traversal rejection, before the
     frozen snapshot and its tree digest are taken.

   Either KIND refuses fail-closed on mismatch / unknown-id / missing / any containment escape or
   shared bounded-pre-read failure.
3. **CI activation-surface enumerator (§1/§3) — the completeness authority FOR DECLARED loader idioms** (import-aware torch/pickle/joblib/onnx `.load`, `load_state_dict`, `from_pretrained`, curated constructors, `reload_model(`; NOT a proof of every possible Python load — a novel framework escapes until its pattern is added). Marks every
   `torch.load`/`pickle.load(s)`/`joblib.load` (import-alias-aware) / `load_state_dict` / `from_pretrained` (HF) / model constructors (`SentenceTransformer`/`PaddleOCR`/…) / `reload_model(` site `gated|producer|offline|unmounted|infra` and
   REDS when a new un-annotated load MATCHING A DECLARED IDIOM appears, or a `gated` site is neither fixed-hash-checked (Phase A, owner decision (b)) nor routed through
   `verify_and_load` (Phase B). This replaces the hand-count (wrong ≥4×). Seed = the §1 map; authority = the
   IMPORT-AWARE enumeration (f2ebe2fa): **128 load sites total, 38 `gated`** across 11 families
   (pickle-classifier, graph2d, pointnet, part, part-v16, hybrid, history, vision3d-uvnet, **ocr** —
   DeepSeek HF `from_pretrained`+PaddleOCR via mounted /ocr — **embedding** — SentenceTransformer — and
   **anomaly-monitor** — the conservatively-gated production metrics model). The 38 is a **conservative
   count of AST load sites**, NOT a proven-live count: several are latent or not-yet-proven-reachable
   (the `auto_remediation` rollback; the `_reload_model_impl` hot-reload deserialization now that
   `/model/reload` is sealed; the two `MetricsAnomalyDetector.load_models` sites with no `src/` caller).
   **Per-site logical reachability is a Wave-1 audit**, not hand-asserted here (§1.B(cont)). The
   current merged script's final summary still says "production-reachable" and offers blanket
   hard-refuse; those are known output-truth defects, not this lock's contract. #521 proposes the
   wording correction but is open and unmerged, so this document relies on current `main` only for
   the executable count/classification data.
   A name-only matcher (review 5) missed the ocr/embedding families and
   import aliases entirely — the enumerator is now import-aware (+onnx/ort, review 6) and reds on any new un-annotated load matching a declared idiom.

4. **Guard-verification contract — the enumerator must PROVE each guard is wired, not just classify
   (review 7c).** #516's enumerator only classifies a site (`class`/`family`/`reason`); it is GREEN
   even with **zero** guards wired — that is correct for **Phase A0** (discovery), but full Phase A
   must make "the guard is present" a machine-checked fact. The manifest schema therefore gains, per
   **logical activation** (NOT per AST call site — see below):

   - `logical_activation_id` — one stable id per *logical* model activation. The 38 AST call sites are
     **not** 38 activations: a single `_load_model()` often has one `torch.load` **plus** several
     `load_state_dict` calls that together are **one** activation; the contract binds the id to the
     activation (its entry function), and the enumerator groups the member AST sites under it.
   - **Every `gated` load must be a raw `torch.load`/`pickle.load`/`from_pretrained`/… that appears
     ONLY inside one of the canonical wrappers** `load_pinned_file(artifact_id, family, env)` /
     `load_pinned_bundle(artifact_id, family, env)` (Phase B: `verify_and_load`). The wrappers are the
     **single sanctioned home** of every raw loader idiom; the enumerator whitelists the raw load lines
     *inside* those wrapper bodies and REDS on a raw `gated` load **anywhere else**. A call site
     activates a model only by calling the wrapper — deleting/renaming/inlining the wrapper makes a raw
     `torch.load`/`from_pretrained` reappear at a non-wrapper site → **observed-RED** (a test ships that
     deletes the wrapper and asserts CI reds). This is a structure the AST scanner CAN verify; an
     arbitrary inter-procedural "guard dominates the load" claim is one it CANNOT, so the contract does
     not depend on a manifest-self-reported `guard_symbol` string.
   - **Escape hatch, same-function only:** where a raw load genuinely cannot move into a wrapper, it is
     accepted ONLY if an `assert_fixed_hash`/`assert_bundle_digest` call on the **same artifact**
     precedes it **lexically in the same function body** — a local dominance the AST can check with no
     inter-procedural reasoning. Anything relying on a guard in a *different* function is rejected.
   - `guard_mode` ∈ `{ sealed | fixed-hash | bundle-digest | verify_and_load | unbuilt }` — what
     protects this activation. `sealed` = the route is 403 (`/model/reload`, #516); `fixed-hash` /
     `bundle-digest` = the Phase-A (b) single-file / tree wrapper; `verify_and_load` = Phase B;
     `unbuilt` = no guard yet (all 38 conservatively-`gated` sites today).

   **The enumerator's Phase-A assertion:** for every `gated` `logical_activation_id`, `guard_mode` must
   be a real guard (not `unbuilt`) **and** the load must be structurally inside a canonical wrapper (or
   same-function-lexically guarded, per above) — verified from the AST, **never** from a self-reported
   symbol string; **deleting or bypassing the wrapper reds CI (an observed-RED test ships with it).**
   Until full Phase A wires the wrappers, the internal loaders are honestly recorded `guard_mode:
   unbuilt` and the assertion is **advisory** (A0) → **blocking** (full Phase A). This closes both the
   "38 sites all unguarded but CI green" gap and the "guard_symbol domination is unverifiable" gap.

**Phase A exit criteria (observed-RED, REQUIRED — NOT claimed executed here):** external
`/model/reload` refuses in prod with no env bypass; a new un-annotated prod loader REDS CI; **every**
`gated` §1.A/1.B site activates ONLY via the §0.5 step-2 per-KIND check (single-file fixed-`SHA-256` / bundle tree-digest) over a server-owned artifact (owner
decision (b)) — no caller path, no env path-swap, no dynamic replacement; before Track E, only the
exact already-in-service `(logical_activation_id, artifact_id, kind, digest)` tuple may load; any
tuple-field change, mismatch, unknown-id, or bounded-pre-read failure refuses fail-closed; and the
enumerator asserts every `gated` site is either fixed-hash-checked
(Phase A) or routed through `verify_and_load` (Phase B).

> **What #516 actually delivers = Phase A0 only, not full Phase A.** #516 seals the external
> `/model/reload` route (403), fail-closes the default `test` creds in a production posture, and ships
> the enumerator as a *discovery/classification* gate. It does **NOT** yet **pin** (fixed-hash /
> bundle-digest-check) the internal `gated` loaders (of the 38 conservatively-`gated` sites; they still load), and the enumerator does
> **NOT** yet assert each `gated` site is fixed-hash/bundle-digest-checked (Phase A, owner (b)) or
> routed through `verify_and_load` (Phase B). So the middle exit criterion above (every gated
> site fixed-hash/bundle-digest-checked, owner decision (b)) and the enumerator guard-assertion are **still unbuilt**. Honest #516 closeout: *external
> reload sealed; producer disabled; internal runtime activation remains proof-unbound.* Full Phase A
> (fixed-hash-check every gated site per (b) + enumerator asserts hashed-or-verified) is the next build
> after this lock ratifies.

**Phase A does NOT build:** the proof schema (§2.2), the signed proof store / issuer / key-custody (§2.3),
`verify_and_load`'s proof lookup (§3 steps 4–5), revocation/expiry, the LKG re-validation readiness probe,
or the append-only activation audit (§3.3). All of those are Phase B.

**Phase A — default-off, pin authority, and failure product-semantics (review 7c).**

- **Default-off at land.** When the Phase-A code merges it ships with **no production pin configured**;
  the guard defaults to *refuse* (`degraded`, §below) until a pin manifest is supplied. **Enabling a
  pin is a separate owner/deployment decision — a controlled release asset, not a code flag** (there
  is no `ENABLE_PIN=1`; a pin exists or it does not). `merged != enabled != safe`.
- **Pin authority & immutability.** The pin manifest (`logical_activation_id / artifact_id → {kind,
  SHA-256 | tree-digest-v1}`) lives in a **controlled release asset that the running service cannot
  modify** (read-only mount / signed release bundle / deploy-time-baked config). It is **not**
  runtime-writable and **not** env-swappable. The service reads it once at startup into immutable state.
- **Pre-Track-E baseline lock — model promotion stays fail-closed.** Phase A accepts only the exact
  `(logical_activation_id, artifact_id, kind, digest)` tuple captured from the already-in-service target
  deployment and explicitly owner-reviewed as the migration baseline. The baseline record contains no
  filesystem path. **Any change to `logical_activation_id`, `artifact_id`, `kind`, or `digest` is a
  model promotion (or contract migration) and is REFUSED before Track E**; there is no "re-pin the same
  baseline" concept and no generic "new pin = deploy" permission. If
  the target environment cannot prove the exact baseline tuple, no pin is issued and the family remains
  `degraded`/503. After Track E exits, changing the tuple requires a separately ratified model-promotion
  contract bound to its versioned reproducible evaluation artifact; this design-lock does not authorize
  that promotion merely because a deployment can replace a file.
- **Failure = an explicit, defined product state — NO silent stub.** On missing pin / hash-miss /
  bundle-digest-miss / containment-escape, each family MUST enter a **defined `degraded` state with a
  `503` (or family-appropriate) health/readiness contract** — the endpoint tells the client the model
  is unavailable; it does **not** silently serve a stand-in. This is a **hard change from today**:
  `src/core/ocr/providers/deepseek_hf.py` currently **falls back to a stub** on load failure
  (`deepseek_hf.py:93`), which is *fail-open* (a caller silently gets a stub answer). Phase A replaces
  every such silent-stub/best-effort fallback with the explicit `degraded`/503 contract; a family's
  Phase-A shard is not accepted until its silent fallbacks are gone and its degraded contract is tested.

### Phase B — Proof membrane (DEFERRED — build **only** when a real pilot needs dynamic model-swap)

The full contract in §2–§3: content-bound split digest (§2.1), the signed proof envelope (§2.2), the trust
source + key-custody + revocation + expiry + LKG re-validation (§2.3), the `verify_and_load` choke-point with
**server-owned artifact IDs** and TOCTOU-safe hash-and-load (§3), and the append-only audit (§3.3). Phase B
**replaces** each Phase-A fixed-hash body with real signed-proof-gated activation — re-enablement is replacing the body,
not adding a flag (§7.2). Phase B depends on **Track E** (§8.1) existing (the split/manifest/metrics the proof
binds to) and on a **HSM / human-gated signer outside CI** (§2.3 key-custody).

**Do not start Phase B while no pilot requires dynamic model-swap.** Sections §2, §2.3, and §3 below
define Phase B; §1 and the enumerator define Phase A.

> **RESOLVED — owner selected (b) Static-artifact-only startup (2026-07-14).** (This records the
> owner's design choice; the owner still ratifies the final pinned head — see the protocol above.)
> Phase A is therefore **static fixed-hash artifact activation**, NOT a blanket hard-refuse of all
> ML: a `gated` loader may load **only** from a *server-owned* artifact resolved from a controlled
> store — **no caller path, no env-var-swap of the path, and dynamic replacement is forbidden**. Models
> work; hot-swap does not. **Before Track E, "models work" is limited to the exact owner-reviewed
> already-in-service baseline tuple; any changed tuple field is a refused promotion/contract migration.** Each
> `gated` site first applies the shared bounded pre-read and then the static check for its §0.5-step-2 **KIND**
> (a subset of §3's `verify_and_load`, with **no signed proof store and no signing keys** — that is
> Phase B): **single-file** — resolve a server-owned artifact id → read once → `SHA-256(bytes)` == the
> pinned value → load THOSE bytes, or refuse; **bundle/tree** — recompute the deterministic tree digest
> over an offline, **frozen**, per-file `resolve()`-contained snapshot → == the pinned tree-digest →
> load from the frozen snapshot, or refuse. The rejected alternative (a) — every `gated` loader hard-refuses, ML
> fully off with a defined `degraded`/health contract — is recorded for the audit trail but NOT the
> chosen path.
>
> #516 (Phase A0) prejudged neither: it sealed only the external `/model/reload` and left the internal
> `gated` loaders loading as-is (the 38 conservatively-`gated` sites; the sealed external `/model/reload`
> route is separate — NOT among the 38, §3 shard; several of the 38 latent/unproven-reachable — §1.B(cont)).
> Full Phase A implements (b) at each `gated` site once its
> reachability is confirmed (Wave-1 audit).

---

## 1. Activation map (verified `file:line` — the boundary this membrane must cover)

Classified by reachability, per the review's taxonomy.

### 1.A External-reachable activation (highest risk)
| Point | Family | Evidence | Current guard |
|---|---|---|---|
| `POST /api/v1/model/reload` (was `reload_model(payload.path, force=…)`) | pickle classifier | `src/api/v1/model.py` | **SEALED 403 by #516.** The externally reachable arbitrary-deserialization / code-execution risk (arbitrary caller `path`; `api_key`+`admin_token` both default `"test"`; no proof binding) is CLOSED. No execution PoC is claimed. Re-open only under Phase B + the identity gate (§3.2). |

### 1.B Startup / runtime-config activation (mutates a RUNNING service)
| Point | Family | Evidence | Current guard |
|---|---|---|---|
| `CLASSIFICATION_MODEL_PATH` → `pickle.load` on **first `predict()`** (lazy, not at startup: `load_model()` is the loader (`classifier.py:47`); `load_models()`-style readiness only builds a snapshot) | pickle classifier | `src/ml/classifier.py:22,85` (load fires via `classifier.py:124`) | **none — no magic-number check and no proof binding.** (The magic-number check is only in the *hot-reload* path `reload_model`, `classifier.py:313`, NOT this startup load — corrected from the first draft.) |
| `GRAPH2D_MODEL_PATH` → `torch.load` in `Graph2DClassifier` | Graph2D | path `src/ml/vision_2d.py:41`, load `src/ml/vision_2d.py:136`; `src/main.py:61` only *reads the flag*, does not load | **no proof binding** |

### 1.C NOT production-reachable — reclassified (corrected from the first draft)
The first draft over-counted these as live "activation points". They are not:
| Point | Real class | Evidence |
|---|---|---|
| `auto_retrain.sh` | **PRODUCER, not activator** | prints `SUCCESS: Ready for deployment` + `export GRAPH2D_MODEL_PATH=…` (`auto_retrain.sh:206,212`) — it generates+quantizes a candidate and prints a deploy command; it does **not** activate a running service. (Still #509 fail-closed, correctly, as a producer of promotable artifacts.) |
| `finetune_from_feedback.py` → `reload_model(force=True)` | **offline CLI** | `scripts/finetune_from_feedback.py:302`; **no `src/` importer** (verified) — the reload happens inside a CLI process that then exits, mutating no running production service. |
| `auto_remediation._action_rollback_model` → `reload_model(prev_path)` | **LATENT (future surface)** | `src/ml/monitoring/auto_remediation.py:301`; `AutoRemediation` is defined/exported but **no live scheduler calls `evaluate_and_act`** (verified). Must gain a proof check *before* it is ever wired to fire. |

### 1.D Explicitly OUT of the membrane
- **Offline tools** — `scripts/quantize_*.py`, `scripts/finetune_*.py`, training scripts: they
  *produce* artifacts; a produced artifact only becomes live by passing 1.A/1.B.
- **Unmounted serving scaffold** — `src/ml/serving/worker.py`, `grpc_service.py`: imported by **0**
  mounted routes (verified). Not a production boundary today; mark `inert` (or delete). If ever
  mounted it is promoted into 1.A and must gain a proof check first.

### 1.B (cont.) MORE `gated` loads — a hand-count kept missing these (reachability = Wave-1 audit)
An earlier draft said "**exactly 3**". That was wrong (the fourth such miscount), because the model zoo
is larger than two families. Additional `gated`, proof-unbound loads (conservatively classified — per-site reachability is a **Wave-1 audit**; the ones flagged below are latent or not-yet-proven-reachable):
| Point | Family | Evidence |
|---|---|---|
| PointNet via the **mounted** pointcloud router | pointnet | router imported+mounted `src/api/__init__.py:269,522`; the endpoint loads the point-cloud model |
| V16 part-classifier ensemble | cad-ensemble | `torch.load` `src/ml/part_classifier.py:62`; reachable via classify / health routes |
| HybridClassifier branch checkpoints | hybrid(stat/text) | `torch.load` `src/ml/hybrid_classifier.py:448,476` |
| PartClassifier / V16 / V14 | part | `torch.load` `src/ml/part_classifier.py:62,655,695` (via `/analyze`, `/health`) |
| HistorySequence | history | `torch.load` `src/ml/history_sequence_classifier.py:162` (via `/analyze`) |
| Vision3D encoder (`UVNET_MODEL_PATH`) | vision3d/uvnet | `torch.load` `src/ml/vision_3d.py:196` (via `/analyze` on 3D/STEP/IGES inputs; format+cache-miss gated but real) |
| DeepSeek OCR (HF) + PaddleOCR — **bundle/tree** | ocr | `from_pretrained` `src/core/ocr/providers/deepseek_hf.py:128,132` + `PaddleOCR` `:86,268`; **mounted** `/ocr` (a directory artifact — bundle-digest KIND) |
| SentenceTransformer embedding — **bundle/tree** | embedding | `SentenceTransformer` `src/core/assistant/embedding_retriever.py:59` (also `semantic_retrieval.py`, `ml/embeddings/model.py`); via the assistant (a directory artifact — bundle-digest KIND) |
| MetricsAnomalyDetector production metrics model | anomaly-monitor | `joblib.load` + `pickle.load` `src/ml/monitoring/anomaly_detector.py` (`load_models`); conservatively `gated` (production monitoring) — **not-yet-proven-reachable**: no `.load_models(` caller exists in `src/` (only class def + exports); **single-file KIND** — `load_models(path)` reads ONE file via a `joblib.load`-or-`pickle.load` **fallback** (verified: both idioms open the same `src`), so the two AST sites are ONE logical activation → one single-file pin |

**Reachability caveat (owner review, 2026-07-15).** The evidence requires `gated` to be interpreted
**conservatively**, not as the current script's stale "production-reachable" summary: the 38 entries
are **AST load sites, not proven-live loaders**. #521 is the open, unmerged output-truth correction.
Known latent / not-yet-proven-reachable among the 38: the
`auto_remediation` rollback (§1.C — no live scheduler); the pickle-classifier `_reload_model_impl`
hot-reload deserialization (`classifier.py:535`), reachable only via the now-sealed `/model/reload`,
the latent `auto_remediation`, and offline `finetune_from_feedback`; and the two
`MetricsAnomalyDetector.load_models` sites (no `.load_models(` caller in `src/`). **Per-site logical
reachability is a Wave-1 audit** — the safe contract is *38 conservatively-`gated` AST entries / 11
families, several latent/unproven, reachability confirmed in Wave 1* — never a hand-asserted live count.

**The recurring lesson — a hand-enumerated count is the wrong contract.** It has been wrong ≥4
times. The membrane's completeness must be enforced **by construction, not by a list**: ship a CI
**activation-surface enumerator** that AST-parses (import-aware) every `torch.load` / `pickle.load(s)` /
`joblib.load` / `onnx.load` / `load_state_dict` / `from_pretrained` / model-constructor / `reload_model(`
call site, marks each `gated | producer | offline | unmounted | infra`, and **fails if any `gated` site
is neither fixed-hash-checked (Phase A) nor routed through `verify_and_load` (Phase B, §3), or if a new
un-annotated load appears.**
That inverts the burden: a new activation surface reds CI until it is gated or explicitly classified
out. This §1 map is the *seed* of that enumerator, not the authority.

**Coverage gap today:** **#509 closes NONE of the runtime activation points** — it closes the
`auto_retrain` *producer* (necessary, but upstream of activation). Every `gated` load (**38 across 11 families** per the enumerator — a conservative classification; per-site reachability is a **Wave-1 audit**, seeded in §1.B/§1.B(cont), the enumerator not the map being the authority) is proof-unbound. The runtime activation membrane is **entirely unbuilt**.

---

## 2. The proof (what "may activate" means) — **Phase B**

An activation is authorized **iff** the model being activated is bound to a *reproducible evaluation
that a fresh clone can re-derive*. The proof has two phases (Track E, `PRODUCT_STRATEGY.md` §8.1).

### 2.1 Pre-training proof — the data is sound
Bind the training/evaluation split to **content**, not paths:
- **portable, versioned manifest** — no workstation absolute paths (today's manifests carry
  `/Users/.../…`); source drawings tracked or content-addressed so provenance is independent.
- **canonical split digest over `content_hash + family + label + side`** — NOT `(file_path → side)`.
  The split digest on the (unmerged) Track E branch `claude/track-e-eval-integrity-splitter-...`
  hashed only `(file_path → side)` — blind to *same-path-changed-bytes* / *same-path-changed-label*,
  the 262/914 content-overlap class it must catch. (That file is NOT on `origin/main@8ff94175`; this
  requirement is what the Track E digest MUST satisfy, not a description of current main.)
- **non-empty both sides** + **largest-component share** reported and bounded (real data: `file:syn`
  is a single dominant component (~50%+ of assignable rows — a PR #510-review figure, not a
  reproducible current metric; re-run gives ~53%) — so a 20% "family holdout" over that is not a family
  sample). Release-quality requires authoritative family/source fields, not the filename heuristic.

### 2.2 Post-training proof — the model is the one that was evaluated
The evaluation result is bound, cryptographically, to the exact artifact being activated:

```
proof = {
  artifact_kind     : "single-file" | "bundle-tree"       # §0.5 step-2 KIND — Phase B binds BOTH (a bundle is not one file)
  artifact_digest   : sha256(file bytes) | tree-digest-v1  # THE candidate — single-file byte-hash OR the versioned tree digest (§0.5 step 2)
  model_family      : "pickle-classifier" | "graph2d"     # families do not share a proof
  split_digest      : content+family+label+side digest    # §2.1
  manifest_digest   : sha256 over source + license + provenance + label-authority (§2.1)
  evaluator_version : pinned                               # which metric code produced the numbers
  thresholds        : the gate values applied              # per-class / macro-F1 / ECE / dup / miss
  environment       : deployment env id                    # a proof for staging ≠ prod
  policy_version    : proof-schema/policy version          # for revocation + forward compat
  not_after         : expiry                               # a stale proof is not a valid proof
  metrics           : per-class, macro-F1, ECE, false-duplicate, missed-reuse, source-stratified
  issuer, signature : who signed it + a signature over all of the above  # see 2.3
}
```

### 2.3 A proof needs a TRUST SOURCE, not just the right fields (review gap)
Binding the fields is necessary but **not sufficient** — the fields alone prove *format*, not
*authority to issue*. The lock therefore also requires:
- **Trusted issuer + signature (or a server-owned, unforgeable record).** A proof is valid only if
  signed by a trusted evaluator identity, or recorded server-side where the requester cannot write
  it. Otherwise anyone who can write a well-formed file has "a proof".
  **Key custody (review gap):** the signing key must **not** be a CI-accessible repo secret — if it
  were, the code-generating routine could mint valid proofs from a branch, collapsing the two-actor
  separation §3.1 relies on. The key lives outside CI (an HSM / a human-gated signer); signing is
  invoked only by the evaluator identity, never by a PR job; and the verifying service **pins** the
  issuer public key (it does not trust whatever key a proof names).
- **`manifest_digest` over provenance**, since the current `split_digest` covers only the split
  assignment — not source, license, or label-authority. Provenance must be in the signed envelope.
- **Policy version + revocation + expiry (`not_after`)** so a compromised or superseded proof stops
  being accepted.
- **Server-owned `family` and `environment`.** These are determined by the activation site and the
  deployment, **never supplied by the requester** (a caller must not label its own model's family or
  claim `environment=staging` to dodge the prod policy).
- **Readiness / fallback on proof-miss.** A failed check does not just raise — it defines the service
  state: fall back to the last-known-good model, or refuse to serve (fail-closed), never silently
  serve the unverified one.
- **The last-known-good must STILL have a valid proof, and needs a re-validation MECHANISM (review
  gap).** "Was good once" is not a licence to keep running. But `verify_and_load` (§3) runs only at
  *activation* time — so revoking the currently-serving model's proof would do nothing without a
  post-activation check. **Mechanism (required, not just asserted):** a readiness probe re-validates
  the *currently-serving* `artifact_digest`'s proof against the store's revocation/expiry list on a bounded
  interval (and on a revocation-push if the store supports it); a serving model whose proof is now
  revoked/expired flips readiness to **red** (drain / refuse to serve). Without this loop, revocation
  only affects the NEXT activation, and the toothless case §2.3 exists to fix persists.

An activation point resolves a **server-owned artifact ID** to bytes from a controlled store (never a
caller path — §3), requires a **signed** proof whose `artifact_digest` (single-file SHA or tree digest) + server-owned `family` +
server-owned `environment` match, is unexpired and unrevoked. **Any miss → fail-closed** per the
readiness rule. This makes the token *bound and authorized* — the defect that sank the first gate (an
artifact for dataset A green-lighting a model on dataset B, and "a passing-format artifact anyone can
emit") cannot recur.

---

## 3. The membrane (one choke-point at every production-reachable activation) — **Phase B** (Phase A fixed-hash-checks these same sites; see §0.5)

A single function — `verify_and_load(artifact_id, family, env)` — where **`artifact_id` is a
server-owned identifier (or model hash), NOT a caller-supplied filesystem path (review gap).** The
first draft let the caller pass `payload.path`, which — even with later verification — first *opens
that path*, creating a path-probe / arbitrary-file-read / memory-IO-DoS surface (a caller could point
it at `/etc/shadow` or a 50 GB file). Instead:
1. **Resolve `artifact_id` in a controlled, read-only, content-addressed store.** The caller names
   *which approved artifact*, the server owns *where its bytes are*. Reject an unknown id.
2. **Apply the shared bounded pre-read from Phase A (§0.5 step 2):** KIND/type and bounded magic-prefix,
   single-file size, and bundle file-count / per-file / aggregate-byte caps. A hostile, corrupt, or
   oversized object is rejected before a full read/copy or framework load.
3. **One immutable read (per KIND):** for a **single-file** artifact, read the resolved bytes into
   memory (or an fd) ONCE and **hash THOSE bytes and load the model from THE SAME bytes**; for a
   **bundle/tree** artifact, **freeze an immutable snapshot** of the directory, take the tree digest
   over the frozen snapshot, and hand the framework **that frozen path** (§0.5 step 2). Either way the
   digested bytes ARE the loaded bytes — no re-open-by-path between check and load, so "verified A,
   loaded B" is impossible (closes both the single-file and the bundle TOCTOU).
4. Look up a **signed** proof for `(artifact_digest, server-owned family, server-owned env)` in a **read-only,
   out-of-band proof store** (a model may not carry its own passing proof — self-attestation again).
   **Store UNREACHABLE ≠ proof ABSENT:** a store timeout/outage is NOT a transient "keep serving" —
   it is an unverifiable state → fail-closed (refuse to activate; hold LKG only while LKG's proof is
   independently cached-valid). A builder must not implement store-down as "skip the check";
   verify the signature/issuer and that the proof is **unexpired and unrevoked** with a current
   `policy_version`/`evaluator_version`/`thresholds`.
5. On **any miss**, apply the readiness rule (§2.3): fall back to LKG **only if LKG's own proof is
   still valid**, else refuse to serve — never load the unverified bytes. Emit an audit record (§3.3).

The set of call sites is **NOT a hand-list** (a hand count has been wrong repeatedly — §1.B(cont)).
The **CI activation-surface enumerator is the completeness authority for the DECLARED loader idioms** (bounded — not a proof of every possible load): the membrane is accepted only
when the enumerator confirms **every `gated` load site is fixed-hash-checked (Phase A) or routed
through `verify_and_load` (Phase B)**. The §1 map is that enumerator's *seed*, not the boundary.
Implementation is **sharded per model family**, each shard wiring the Phase-A fixed-hash check (in
Phase B, `verify_and_load`) **before** the load and shipping its own enumerator entry + golden:

- **pickle-classifier** — the external `/model/reload` route (`model.py:71`) stays **sealed 403** (#516, not fixed-hash-wired; not one of the 38 `gated` load sites); the Phase-A body wires `classifier.py:85` (the lazy
  first-`predict()` `pickle.load`, which today has NO magic/hash/opcode check). The current hot-reload
  path deserializes **before** it checks (`classifier.py:535` `pickle.loads` runs ahead of the
  whitelist/hash check, and the hash is truncated to 16 hex) — `verify_and_load`'s "cheap guards +
  one immutable read, hash-and-load the same bytes" (above) replaces that ordering.
- **graph2d** — `vision_2d.py:136`.
- **hybrid** (its own container auto-enables on file presence, same footing as graph2d) —
  `hybrid_classifier.py:448` (stat branch) and `:476` (text branch).
- **pointnet** — the **mounted** `/pointcloud` router → `pointnet/inference.py:108`.
- **part / v16 / v14** — `part_classifier.py:62,655,695` (reached via `/analyze`, `/health`).
- **history-sequence** — `history_sequence_classifier.py:162` (reached via `/analyze`).
- **vision3d / uvnet** — `vision_3d.py:196` (`UVNET_MODEL_PATH`, reached via `/analyze` on 3D inputs).
- **ocr** (bundle-digest KIND) — DeepSeek HF `from_pretrained` (`deepseek_hf.py:128,132`) + PaddleOCR (`:86,268`), **mounted** `/ocr`; a directory artifact → tree-digest, offline-only, no silent stub (§0.5 step 2 + failure-semantics).
- **embedding** (bundle-digest KIND) — SentenceTransformer (`embedding_retriever.py:59`, `semantic_retrieval.py`, `ml/embeddings/model.py`), via the assistant; directory artifact → tree-digest.
- **anomaly-monitor** — `anomaly_detector.py` `load_models(path)`; the conservatively-gated production metrics model. **single-file KIND** — one `path`, read via a joblib-or-pickle **fallback** (both open the same file; one logical activation → one fixed-hash pin).
- any surface the enumerator later discovers → its own shard before it can go live.

The §1.C **latent activator** `auto_remediation.py:301` gains the same activation guard before it is
ever scheduled. `auto_retrain.sh` is different: it is a **producer, never an activator**, and it MUST
NOT call the runtime activation membrane. #509's unconditional block remains until Track E's exit
condition is satisfied; only then may a separately reviewed **Track E model-promotion gate** produce a
candidate plus its versioned evaluation-integrity artifact for an explicit owner promotion decision.
Producing that evidence does not load or activate the candidate; every later runtime load still passes
the phase-appropriate activation guard. The **unmounted serving scaffold** (`src/ml/serving/*`) is
promoted into a shard automatically if it is ever mounted (the enumerator reds until it is).

Per owner decision (b), the **Phase-A body at each `gated` site is the per-KIND static check of §0.5
step 2** — **single-file** (read once → `SHA-256(bytes)` == the pinned value → load THOSE bytes) or
**bundle/tree** (recompute the deterministic tree-digest over the offline, per-file `resolve()`-contained
**frozen** unpacked snapshot → == the pinned tree-digest → hand the framework the **frozen snapshot's** local path), else refuse to a
defined `degraded`/503 — models load, but **only** from pinned server-owned
artifacts, with no caller path, no env path-swap, no runtime hot-swap. The external `/model/reload`
route is the exception: it stays **sealed (403, done by #516)**, re-opened only under Phase B + the
identity gate (§3.2). Once **Track E** and the signed proof store exist, Phase B **replaces** the
fixed-hash body with the signed-proof `verify_and_load` — re-enablement is replacing the body, not
adding a flag (`merged != enabled != safe to enable`, §7.2). Build order: **Phase A → Track E →
Phase B → enablement.**

---

## 3.1 Two threat actors — do NOT conflate them (review gap)

The first draft blurred two distinct threats; the membrane addresses one, and the other is covered
elsewhere. Keeping them separate keeps the design honest.

- **Runtime API caller** (someone hitting `POST /model/reload`, or influencing a startup config).
  *This* is what the activation membrane defends: an authenticated-but-wrong or malicious caller
  cannot activate a model without passing the **phase-appropriate activation gate** — Phase A: the
  static fixed-hash / bundle-digest check over a server-owned pinned artifact (owner (b)); Phase B: the
  signed proof. The unattended routine is **not** this actor —
  it has no runtime path to `/model/reload`.
- **Code-generating routine** (the unattended loop): it can modify branches and open PRs, but it
  **cannot thereby reach a runtime activation**. It is governed by *different* controls. Current
  branch protection (live facts: **0 required approvals**, **11 required strict status checks**,
  required conversation-resolution, `enforce_admins`, no force-push/deletion) blocks a direct push and
  a checks-failing PR, but **requires NO human approval** — so it is not independent review and must
  not substitute for disabling an unsafe routine (a checks-passing PR is not review-gated).
  `CODEOWNERS` is only a path inventory in this solo-maintainer repository;
  `require_code_owner_reviews=false` and must remain so while the owner cannot approve its own PR.
  The isolated-critic protocol above supplies review evidence, while the human owner alone ratifies
  and authorizes a pinned head. Do not claim CODEOWNERS containment. Conflating
  the two would mis-scope the membrane (it is not a defense against branch edits) and under-protect the
  runtime (which needs the proof regardless of who wrote the code).

## 3.2 The external `/model/reload` HARD-DEPENDS on the production-identity gate (review gap)

The proof membrane authorizes *which model* may activate; it does **not** authenticate *who* is
asking. `POST /model/reload` is **now SEALED (403, #516)**; before the seal it was guarded only by `api_key` +
`admin_token` **both defaulting to `test`** (`dependencies.py:8,38` — #516 also fail-closes that default
in a production posture). Completing the proof membrane must **not** re-open the route while identity is
fail-open — a trusted-but-defaulted caller passing a valid proof would still be an anonymous activation.

**Ordering lock:** the external `/model/reload` route may be enabled **only when BOTH gates hold**:
1. the production-identity gate (separate design-lock): no default `test` credentials, unambiguous
   authenticated tenant/user, `x-user-id` cannot override the token subject (`src/api/middleware/integration_auth.py:104`);
2. the proof membrane (this doc).
The route **was LIVE (the original vulnerability, §1.A/§6); #516 has now SEALED it (403)** and it
stays sealed until BOTH gates hold — re-opening it is a Phase B + identity-gate decision, never a
flag. So for `/model/reload`, "disabled" is now a fact (#516); for the §1.B internal loads it is the
Phase-A fixed-hash body that must still be built. (The §1.B loads are internal and gated by
the proof membrane alone, but they must not read a caller-influenced path either). Neither gate alone
is sufficient; do not ship one and call the surface safe.

## 3.3 Append-only activation audit (review suggestion — adopted)

Every activation *attempt* (success or refusal) writes one **append-only** record:
`{ timestamp, actor (authenticated identity), decision (activated|refused|fell-back-to-LKG),
proof_id, artifact_id, model_family, environment, artifact_kind, candidate_artifact_digest,
previous_artifact_digest, new_artifact_digest (== candidate on 'activated', null on 'refused'),
failure_reason }` (drop the ambiguous bare `model_hash`; the digest is the single-file SHA or the tree
digest per `artifact_kind`; `candidate_artifact_digest` is what was attempted, `new_artifact_digest`
what is now serving).
**No filesystem paths or other sensitive strings** (per the redaction discipline the strategy applies
to logs). The ledger is append-only so a bad activation cannot be erased, and it is what makes
revocation and incident review possible after the fact.

## 4. Non-goals / explicit exclusions

- Not building the proof store or Track E here — this locks their **contract**.
- Not touching auth defaults here — that is a **separate** L3 design-lock (production-identity model:
  `dependencies.py:8` default `test`, `src/api/middleware/integration_auth.py:104` header-overrides-subject). Cross-ref,
  don't merge the two.
- Not covering offline training/quantization (§1.D) — they don't activate.
- Customer corrections are **isolated until Track E lands**; after it, they may enter a
  training-readable store **only** under explicit customer authorization, single-customer isolation,
  and with cross-customer training default-off. (The earlier "never enter a training-readable store"
  overstated the canonical strategy — the rule is conditional, not absolute.)

## 5. Golden matrix the implementation must ship (observed-RED, REQUIRED — not yet executed; the membrane is unbuilt)

**Phasing of this matrix (owner decision (b)).** **Phase A** is *baseline-only static fixed-`SHA-256`
activation* — before Track E, a `gated` site loads ONLY the exact owner-reviewed already-in-service
server-owned artifact tuple, **no proof store, no signing keys**; its
golden cases are the first table. **Phase B** adds the signed-proof binding (the second table); any row
needing a proof store / signature / evaluator / revocation is Phase B **by definition**. Phase B depends
on **Track E** existing, so the build order is **Phase A → Track E → Phase B → enablement**.

### Phase A (b) — baseline-only static fixed-hash activation (no proof store, no signing keys)
| Case | Required result |
|---|---|
| a `gated` site loads a **server-owned** artifact whose bytes' `SHA-256` **== the pinned value** | **GREEN — the Phase-A green** (this is the fixed-hash success case; models work) |
| **pre-Track-E:** any configured tuple field differs from the owner-reviewed already-in-service baseline | RED → REFUSED / `degraded`/503; any difference is a model promotion/contract migration, not a deploy-time re-pin |
| bytes' `SHA-256` **≠** the pinned value, or the pinned artifact is missing/unknown-id | RED → the site enters a defined **`degraded`/503** state (never loads the mismatched bytes) |
| caller supplies a filesystem `path`, or an env var swaps the artifact path | REJECTED — no caller-influenced path is ever opened |
| attempt to **hot-swap / re-point the pinned manifest at runtime** | REJECTED — the baseline manifest is **immutable at runtime**; before Track E, any changed tuple field is refused even across a deploy |
| the resolved artifact **escapes the store root** (symlink / `..` / absolute outside root) | RED — `resolve()`-contained to the store root (same containment discipline as the Track E manifest) |
| single-file exceeds its family size cap or has the wrong KIND/type/magic | REJECTED before full read or framework load |
| bundle exceeds file-count, per-file-size, or aggregate-byte cap, or contains a malformed/unreadable entry | REJECTED during bounded metadata/prefix scan, before freeze/copy/digest/framework load |
| **bundle/tree** family (HF/SentenceTransformer/PaddleOCR): the deterministic tree digest over the **frozen** unpacked snapshot **== the pinned tree-digest** | **GREEN** (bundle Phase-A green; loads from the **frozen snapshot's** local path) |
| a file is added/removed/changed inside the bundle dir | RED — the tree digest changes → `degraded`/503 |
| the loader would **fetch from a network hub** (hub id / online) | REJECTED — offline-only (`HF_HUB_OFFLINE`/`local_files_only`); a hub id is never a valid artifact id |
| any file in the bundle **escapes the store root** (per-file symlink/`..`/absolute) | RED — per-file `resolve()`-containment |
| load fails and the provider tries a **silent stub / best-effort fallback** (e.g. deepseek_hf.py:93) | FORBIDDEN — must enter the explicit `degraded`/503, never serve a stub |
| **no provable exact baseline pin** at land (default-off) | the guard refuses → `degraded`/503 (baseline capture + owner review is separate from merge; no code flag opens it) |
| **swap a single-file artifact between hash and load** (TOCTOU) | RED — read once, hash and load THE SAME bytes |
| **change a bundle file AFTER the tree digest, before/during the framework's read** (bundle TOCTOU) | RED — the digested tree is a **frozen immutable snapshot**; the framework reads the digested files, never the mutable original (§0.5 step 2) |
| a new **un-annotated** prod loader appears | CI RED (the enumerator, §1/§3) |

### Phase B — signed-proof binding (needs Track E + the signed store; replaces the Phase-A body)
| Case | Required result |
|---|---|
| activate a model whose bytes have **no** proof | RED at every §1.A/1.B point |
| **swap the file (single-file) or a bundle file (tree) between digest and load** (TOCTOU) | RED — the loaded bytes/tree are the digested ones; the bundle loads from a frozen snapshot (§3) |
| **unsigned / wrongly-signed** proof (well-formed but not from a trusted issuer) | RED (no authority — §2.3) |
| **expired / revoked** proof (`not_after` passed, or policy revoked) | RED (stale proof is not valid) |
| requester supplies `family` or `environment=staging` to dodge policy | IGNORED — both are server-owned (§2.3) |
| caller supplies a filesystem `path` (not a server-owned artifact id) | REJECTED — no caller path is opened (§3) |
| shared bounded pre-read fails (wrong KIND/type/magic, oversized file/bundle, malformed entry) | REJECTED before full read/copy or framework load (§0.5/§3) |
| fall back to a last-known-good model whose proof is REVOKED/EXPIRED | RED — readiness refuses; LKG needs a valid proof (§2.3) |
| complete the proof membrane but leave `ADMIN_TOKEN=test` on `/model/reload` | route stays DISABLED — needs the identity gate too (§3.2) |
| proof for family A, activate a family-B model of same bytes-len | RED (family mismatch) |
| proof for `staging`, activate in `prod` | RED (environment mismatch) |
| change one byte of the model file after the proof was issued | RED (`artifact_digest` mismatch) |
| add/remove/change a file in a **bundle** artifact after the proof was issued | RED (tree-digest → `artifact_digest` mismatch) |
| change a **label** in the manifest, re-derive split_digest | RED (split digest changes — §2.1) |
| rewrite an existing file's **bytes** to duplicate another's content | RED (content in digest — §2.1) |
| bump `evaluator_version` / a threshold | RED (stale proof) |
| a genuine, signed, unexpired, fresh-clone-reproducible proof matching (`artifact_digest`, family, env) — single-file OR bundle | GREEN — the Phase-B green (the Phase-A green is the fixed-hash match above) |

## 6. What this changes about the current risk statement

Accurate post-#509 status, by threat actor (§3.1 — do not conflate them):

- **The unattended routine** is constrained by branch protection (live: **0 required approvals**,
  **11 required strict checks**, conversation-resolution, `enforce_admins`, no force-push/deletion) —
  which stops a direct push and a checks-failing PR but does **not require a human approval**, so it
  does not by itself prevent a checks-passing self-merge; cannot retrain via `auto_retrain.sh` (#509);
  and — being a code-generating actor — **has no runtime path to any activation point** (it cannot call
  `/model/reload`). It is **not** contained by CODEOWNERS (#512 is an unarmed ownership map), and branch
  protection is not a substitute for stopping it. No unattended routine may author or merge this L3
  runtime.
- **A runtime API caller / operator** can still reach the `gated` AST load sites (**38 across 11 model
  families** — a conservative classification; per-site reachability is a **Wave-1 audit** and several are
  latent/unproven (§1.B(cont)); these AST sites group into fewer *logical* activations — §0.5 step 4 — the guard-coverage denominator) (the CI activation-surface enumerator, not a hand count, is the
  authority — §1.B(cont)/§3). The external `/model/reload` is **now SEALED (403, #516)** and is no
  longer reachable; the remaining gap is the **internal `gated` loaders** (of the 38 conservatively-`gated`
  sites (the sealed external `/model/reload` route is separate — NOT among the 38, §3 shard); several latent/unproven — §1.B(cont)) — the pickle-classifier startup load
  and the graph2d / hybrid-branch (stat, text) / pointnet / part / history-sequence / vision3d(uvnet) /
  ocr / embedding / anomaly-monitor surfaces (per-site reachability confirmed in the Wave-1 audit) — which still load **unpinned** (no Phase-A fixed-hash check
  yet). *That* is what full Phase A must close, and it is **unbuilt**.

**So: strong bleeding-control on the code-gen actor; the runtime activation membrane is not built.**
This design-lock defines that membrane. Stopping the routine and owner-ratifying this design under the
solo-maintainer protocol gate *building* it. The isolated critic is mandatory evidence but cannot
ratify or merge. The runtime membrane remains a separate, additional need.

---

## Appendix A — reproduction

```sh
# activation map (run at repo root)
grep -rn 'reload_model(' src scripts --include='*.py' | grep -v 'def reload_model'
grep -rn 'CLASSIFICATION_MODEL_PATH\|GRAPH2D_MODEL_PATH' src --include='*.py' | grep -v test
sed -n '46,72p' src/api/v1/model.py            # external /reload + auth deps
sed -n '38,42p;136,136p' src/ml/vision_2d.py   # __init__ path + real torch.load
sed -n '60,72p' src/main.py                    # flag-check only, NOT a load
# serving scaffold is unmounted:
grep -rln 'ml.serving' src/api --include='*.py' | wc -l     # -> 0
```
