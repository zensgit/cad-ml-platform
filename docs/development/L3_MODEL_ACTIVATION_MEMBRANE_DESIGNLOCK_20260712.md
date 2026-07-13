# L3 Design-Lock — Model-Release & Activation Proof Membrane

**Date**: 2026-07-12 · **Status**: PROPOSED (for-review; do NOT self-merge; owner ratifies)
**Rigor**: L3 (`PRODUCT_STRATEGY.md` §7.1) · **Grounded on**: `origin/main@8ff94175`
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
4. Merge, deployment, and enablement are separate decisions. The implementation lands default-off;
   production activation remains blocked on the proof membrane, the production-identity gate, the
   Track E evidence, and the explicit enablement decision in this document.

`.github/CODEOWNERS` may inventory this surface, but `require_code_owner_reviews` remains false while
the repository has only one developer; enabling it would create an impossible approval gate, not an
independent critic. If an independent human reviewer becomes available later, their review strengthens
this protocol but is not simulated in the meantime.

---

## 0. Why this exists (and why the first attempt was insufficient)

#509 made `scripts/auto_retrain.sh` unconditionally fail-closed. That was correct **but narrow**, and
an earlier claim that it "closed retraining on main" was **overstated**. Corrected here (second
review): `auto_retrain.sh` is a **producer** (it prints a deploy command, it does not activate a
running service), so #509 closes **none of the production-reachable activation points** (there are
≥5, across ≥4 model families — see §1.B; a hand-count has been wrong three times, hence the
CI-enumerator contract). It is bleeding-control one layer upstream; the runtime membrane is unbuilt.

Corrections from review, all load-bearing:

1. **Model families are distinct release paths, not one path's bypasses.** `POST /api/v1/model/reload`
   hot-reloads the **pickle classifier** (`CLASSIFICATION_MODEL_PATH`, `src/ml/classifier.py:227,85`).
   `auto_retrain.sh`/#509 promotes the **Graph2D checkpoint** (`GRAPH2D_MODEL_PATH`, resolved in
   `Graph2DClassifier.__init__` (`src/ml/vision_2d.py:40-41`) and loaded via `torch.load` at
   `src/ml/vision_2d.py:136`). These are **different models with
   different activation surfaces**. The membrane must be **cross-family**.

2. **A file-reference count is a discovery list, not a boundary.** The 15 `*_MODEL_PATH` / `load` hits
   are not 15 production entry points. The acceptance goal is *"every production-reachable activation
   point passes the same proof"* — **not** mechanically wrapping every `load()`.

---

## 1. Activation map (verified `file:line` — the boundary this membrane must cover)

Classified by reachability, per the review's taxonomy.

### 1.A External-reachable activation (highest risk)
| Point | Family | Evidence | Current guard |
|---|---|---|---|
| `POST /api/v1/model/reload` → `reload_model(payload.path, force=payload.force)` | pickle classifier | `src/api/v1/model.py:46,71` | `api_key` + `admin_token` **both default `"test"`** (`src/api/dependencies.py:8,38`). **No proof binding.** Client supplies an arbitrary `path`. |

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

### 1.B (cont.) MORE production-reachable loads — a hand-count kept missing these
An earlier draft said "**exactly 3**". That was wrong (the third such error), because the model zoo
is larger than two families. Verified additional production-reachable, proof-unbound loads:
| Point | Family | Evidence |
|---|---|---|
| PointNet via the **mounted** pointcloud router | pointnet | router imported+mounted `src/api/__init__.py:269,522`; the endpoint loads the point-cloud model |
| V16 part-classifier ensemble | cad-ensemble | `torch.load` `src/ml/part_classifier.py:62`; reachable via classify / health routes |
| HybridClassifier branch checkpoints | hybrid(stat/text) | `torch.load` `src/ml/hybrid_classifier.py:448,476` |
| PartClassifier / V16 / V14 | part | `torch.load` `src/ml/part_classifier.py:62,655,695` (via `/analyze`, `/health`) |
| HistorySequence | history | `torch.load` `src/ml/history_sequence_classifier.py:162` (via `/analyze`) |
| Vision3D encoder (`UVNET_MODEL_PATH`) | vision3d/uvnet | `torch.load` `src/ml/vision_3d.py:196` (via `/analyze` on 3D/STEP/IGES inputs; format+cache-miss gated but real) |

**The recurring lesson — a hand-enumerated count is the wrong contract.** It has been wrong three
times. The membrane's completeness must be enforced **by construction, not by a list**: ship a CI
**activation-surface enumerator** that greps every `torch.load` / `pickle.load` / `load_state_dict` /
`reload_model(` call site, marks each `gated | producer | offline | unmounted`, and **fails if any
`gated` site does not route through `verify_and_load` (§3), or if a new un-annotated load appears.**
That inverts the burden: a new activation surface reds CI until it is gated or explicitly classified
out. This §1 map is the *seed* of that enumerator, not the authority.

**Coverage gap today:** **#509 closes NONE of the runtime activation points** — it closes the
`auto_retrain` *producer* (necessary, but upstream of activation). Every load in §1.A/1.B (now ≥5,
across ≥4 families) is proof-unbound. The runtime activation membrane is **entirely unbuilt**.

---

## 2. The proof (what "may activate" means)

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
  model_hash        : sha256(model file bytes)            # THE candidate, not "a" model
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
  the *currently-serving* `model_hash`'s proof against the store's revocation/expiry list on a bounded
  interval (and on a revocation-push if the store supports it); a serving model whose proof is now
  revoked/expired flips readiness to **red** (drain / refuse to serve). Without this loop, revocation
  only affects the NEXT activation, and the toothless case §2.3 exists to fix persists.

An activation point resolves a **server-owned artifact ID** to bytes from a controlled store (never a
caller path — §3), requires a **signed** proof whose `model_hash` + server-owned `family` +
server-owned `environment` match, is unexpired and unrevoked. **Any miss → fail-closed** per the
readiness rule. This makes the token *bound and authorized* — the defect that sank the first gate (an
artifact for dataset A green-lighting a model on dataset B, and "a passing-format artifact anyone can
emit") cannot recur.

---

## 3. The membrane (one choke-point, enforced at every production-reachable activation)

A single function — `verify_and_load(artifact_id, family, env)` — where **`artifact_id` is a
server-owned identifier (or model hash), NOT a caller-supplied filesystem path (review gap).** The
first draft let the caller pass `payload.path`, which — even with later verification — first *opens
that path*, creating a path-probe / arbitrary-file-read / memory-IO-DoS surface (a caller could point
it at `/etc/shadow` or a 50 GB file). Instead:
1. **Resolve `artifact_id` in a controlled, read-only, content-addressed store.** The caller names
   *which approved artifact*, the server owns *where its bytes are*. Reject an unknown id.
2. **Cheap guards before reading the whole thing:** file type / magic-number and a max-size cap, so a
   hostile or corrupt object is rejected before it is fully loaded.
3. **One immutable read:** read the resolved bytes into memory (or an fd) ONCE; **hash THOSE bytes and
   load the model from THE SAME bytes.** Closes the TOCTOU — no re-open-by-path between check and load,
   so "verified A, loaded B" is impossible.
4. Look up a **signed** proof for `(hash, server-owned family, server-owned env)` in a **read-only,
   out-of-band proof store** (a model may not carry its own passing proof — self-attestation again).
   **Store UNREACHABLE ≠ proof ABSENT:** a store timeout/outage is NOT a transient "keep serving" —
   it is an unverifiable state → fail-closed (refuse to activate; hold LKG only while LKG's proof is
   independently cached-valid). A builder must not implement store-down as "skip the check";
   verify the signature/issuer and that the proof is **unexpired and unrevoked** with a current
   `policy_version`/`evaluator_version`/`thresholds`.
5. On **any miss**, apply the readiness rule (§2.3): fall back to LKG **only if LKG's own proof is
   still valid**, else refuse to serve — never load the unverified bytes. Emit an audit record (§3.3).

The set of call sites is **NOT a hand-list** (a hand count has been wrong repeatedly — §1.B(cont)).
The **CI activation-surface enumerator is the completeness authority**: the membrane is accepted only
when the enumerator confirms **every `gated` load site routes through `verify_and_load`**. The §1 map
is that enumerator's *seed*, not the boundary. Implementation is **sharded per model family**, each
shard wiring `verify_and_load` **before** the load and shipping its own enumerator entry + golden:

- **pickle-classifier** — `model.py:71` (external `/model/reload`) and `classifier.py:85` (the lazy
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
- any surface the enumerator later discovers → its own shard before it can go live.

The §1.C producer/latent points gain the same check **before they are wired to activate**:
`auto_remediation.py:301` (before it is ever scheduled), and `auto_retrain.sh` (swap #509's
unconditional block for the membrane call — the *only* place the permanent-closed logic changes, and
only once the signed store exists). The **unmounted serving scaffold** (`src/ml/serving/*`) is promoted
into a shard automatically if it is ever mounted (the enumerator reds until it is).

Until the proof store and Track E exist, **the membrane's default implementation is #509's: raise
unconditionally.** Re-enablement is replacing the body, not adding a flag. (`merged != enabled !=
safe to enable`, §7.2.)

---

## 3.1 Two threat actors — do NOT conflate them (review gap)

The first draft blurred two distinct threats; the membrane addresses one, and the other is covered
elsewhere. Keeping them separate keeps the design honest.

- **Runtime API caller** (someone hitting `POST /model/reload`, or influencing a startup config).
  *This* is what the activation membrane defends: an authenticated-but-wrong or malicious caller
  cannot activate a model without a valid signed proof. The unattended routine is **not** this actor —
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
asking. `POST /model/reload` today is guarded by `api_key` + `admin_token`, **both defaulting to
`test`** (`dependencies.py:8,38`). Completing the proof membrane must **not** re-open that route while
identity is fail-open — a trusted-but-defaulted caller passing a valid proof is still an anonymous
activation.

**Ordering lock:** the external `/model/reload` route may be enabled **only when BOTH gates hold**:
1. the production-identity gate (separate design-lock): no default `test` credentials, unambiguous
   authenticated tenant/user, `x-user-id` cannot override the token subject (`src/api/middleware/integration_auth.py:104`);
2. the proof membrane (this doc).
The route is **LIVE today** (that is precisely the vulnerability, §1.A/§6) — so "disabled" is an
ACTION the implementation MUST take (disable `/model/reload` until both gates hold), not a current
fact. Do not read §3.2/§5 as "already disabled". (The §1.B loads are internal and gated by
the proof membrane alone, but they must not read a caller-influenced path either). Neither gate alone
is sufficient; do not ship one and call the surface safe.

## 3.3 Append-only activation audit (review suggestion — adopted)

Every activation *attempt* (success or refusal) writes one **append-only** record:
`{ timestamp, actor (authenticated identity), decision (activated|refused|fell-back-to-LKG),
proof_id, artifact_id, model_family, environment, candidate_model_hash, previous_model_hash,
new_model_hash (== candidate on 'activated', null on 'refused'), failure_reason }` (drop the ambiguous
bare `model_hash`; `candidate_model_hash` is what was attempted, `new_model_hash` what is now serving).
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

## 5. Golden matrix the implementation must ship (observed-RED, executed)

| Case | Required result |
|---|---|
| activate a model whose bytes have **no** proof | RED at every §1.A/1.B point |
| **swap the file between hash and load** (TOCTOU) | RED — the loaded bytes are the hashed bytes (§3) |
| **unsigned / wrongly-signed** proof (well-formed but not from a trusted issuer) | RED (no authority — §2.3) |
| **expired / revoked** proof (`not_after` passed, or policy revoked) | RED (stale proof is not valid) |
| requester supplies `family` or `environment=staging` to dodge policy | IGNORED — both are server-owned (§2.3) |
| caller supplies a filesystem `path` (not a server-owned artifact id) | REJECTED — no caller path is opened (§3) |
| artifact exceeds max-size / wrong file type | REJECTED before full load (§3) |
| fall back to a last-known-good model whose proof is REVOKED/EXPIRED | RED — readiness refuses; LKG needs a valid proof (§2.3) |
| complete the proof membrane but leave `ADMIN_TOKEN=test` on `/model/reload` | route stays DISABLED — needs the identity gate too (§3.2) |
| proof for family A, activate a family-B model of same bytes-len | RED (family mismatch) |
| proof for `staging`, activate in `prod` | RED (environment mismatch) |
| change one byte of the model file after the proof was issued | RED (model_hash mismatch) |
| change a **label** in the manifest, re-derive split_digest | RED (split digest changes — §2.1) |
| rewrite an existing file's **bytes** to duplicate another's content | RED (content in digest — §2.1) |
| bump `evaluator_version` / a threshold | RED (stale proof) |
| a genuine, signed, unexpired, fresh-clone-reproducible proof matching (hash, family, env) | GREEN — the only green |

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
- **A runtime API caller / operator** can still reach the **≥5 production-reachable activation points
  across ≥4 model families** (the CI activation-surface enumerator, not a hand count, is the
  authority — §1.B(cont)/§3) — the external `/model/reload` (admin token defaulted to `test`), the
  pickle-classifier startup load, and the graph2d / hybrid-branch (stat, text) / pointnet / part /
  history-sequence / vision3d(uvnet) `torch.load` surfaces — none of which have proof binding. *This*
  is what the membrane defends, and it is **entirely unbuilt**.

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
