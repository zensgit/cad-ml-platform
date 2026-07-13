# L3 Design-Lock — Model-Release & Activation Proof Membrane

**Date**: 2026-07-12 · **Status**: PROPOSED (for-review; do NOT self-merge; owner ratifies)
**Rigor**: L3 (`PRODUCT_STRATEGY.md` §7.1) · **Grounded on**: `origin/main@8ff94175`
**Authority**: `PRODUCT_STRATEGY.md` §4 (AI safety), §5.2 (evaluation integrity not release-grade),
§8.1 (Track E). Scheduled deliverable for the 7/20–7/26 week; pulled forward because the runtime
work is P0-blocked and a design-lock is a doc, not runtime.

> **This is a proposal, not an implementation.** It changes no runtime, touches no model-activation
> code, and does not itself close any surface. It defines the contract the future implementation must
> satisfy so it can be ratified before any code is written — precisely the "propose, don't build"
> mode L3 requires while an unattended reviewer is unavailable.

---

## 0. Why this exists (and why the first attempt was insufficient)

#509 made `scripts/auto_retrain.sh` unconditionally fail-closed. That was correct **but narrow**, and
an earlier claim that it "closed retraining on main" was **overstated**. Corrected here (second
review): `auto_retrain.sh` is a **producer** (it prints a deploy command, it does not activate a
running service), so #509 closes **none of the three production-reachable activation points** — it is
bleeding-control one layer upstream. The runtime activation membrane is still entirely unbuilt.

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
| `CLASSIFICATION_MODEL_PATH` → `pickle.load` at import/startup | pickle classifier | `src/ml/classifier.py:22,85` | **none — no magic-number check and no proof binding.** (The magic-number check is only in the *hot-reload* path `reload_model`, `classifier.py:313`, NOT this startup load — corrected from the first draft.) |
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

**Coverage gap today (corrected):** there are **3 production-reachable activation points** — the
external `/model/reload` (1.A) and the two startup loads (1.B). **#509 closes NONE of them** — it
closes the `auto_retrain` *producer*, which is necessary but is not one of the three. So the runtime
activation membrane is **entirely unbuilt**; the auto_retrain closure is bleeding-control one layer
upstream. (The producers/latent points in 1.C must gain the check too, but before they are wired,
not as today's live gap.)

---

## 2. The proof (what "may activate" means)

An activation is authorized **iff** the model being activated is bound to a *reproducible evaluation
that a fresh clone can re-derive*. The proof has two phases (Track E, `PRODUCT_STRATEGY.md` §8.1).

### 2.1 Pre-training proof — the data is sound
Bind the training/evaluation split to **content**, not paths:
- **portable, versioned manifest** — no workstation absolute paths (today's manifests carry
  `/Users/.../…`); source drawings tracked or content-addressed so provenance is independent.
- **canonical split digest over `content_hash + family + label + side`** — NOT `(file_path → side)`.
  The current digest (`track_e_eval_integrity.py`) is blind to *same-path-changed-bytes* and
  *same-path-changed-label* — exactly the 262/914 content-overlap class it must catch.
- **non-empty both sides** + **largest-component share** reported and bounded (real data: `file:syn`
  is ≈50.9% of assignable rows — a single component; a 20% "family holdout" over that is not a family
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
- **`manifest_digest` over provenance**, since the current `split_digest` covers only the split
  assignment — not source, license, or label-authority. Provenance must be in the signed envelope.
- **Policy version + revocation + expiry (`not_after`)** so a compromised or superseded proof stops
  being accepted.
- **Server-owned `family` and `environment`.** These are determined by the activation site and the
  deployment, **never supplied by the requester** (a caller must not label its own model's family or
  claim `environment=staging` to dodge the prod policy).
- **Readiness / fallback on proof-miss.** A failed check does not just raise — it defines the service
  state: stay on the last-known-good model, or refuse to serve (fail-closed), never silently serve
  the unverified one.

An activation point computes `sha256(model-about-to-load)`, requires a **signed** proof whose
`model_hash` + server-owned `family` + server-owned `environment` match, is unexpired and unrevoked.
**Any miss → fail-closed** per the readiness rule. This makes the token *bound and authorized* — the
defect that sank the first gate (an artifact for dataset A green-lighting a model on dataset B, and
"a passing-format artifact anyone can emit") cannot recur.

---

## 3. The membrane (one choke-point, enforced at every production-reachable activation)

A single function — call it `verify_and_load(model_path, family, env)` — that:
1. **Opens the file ONCE and reads the bytes into memory (or holds an fd); hashes THOSE bytes and
   loads the model from THE SAME bytes.** This closes a TOCTOU (review gap): the first draft hashed
   *by path* and let the loader re-open the path, so the file could be swapped between check and load
   — "verified A, loaded B". Hash and load must be one immutable read (bytes/fd), or the artifact must
   be a content-addressed, read-only object the loader fetches by hash.
2. looks up a **signed** proof for `(hash, server-owned family, server-owned env)` in a **read-only,
   out-of-band proof store** (a model may not carry its own passing proof — self-attestation again);
3. verifies the signature/issuer, and that the proof is **unexpired and unrevoked** and its
   `policy_version`/`evaluator_version`/`thresholds` are current;
4. on **any miss**, applies the readiness rule (§2.3): keep last-known-good or refuse to serve —
   never load the unverified bytes.

Every **production-reachable** point in §1.A/1.B calls it **before** the model is loaded/served:
- `model.py:71` — the external `/model/reload`;
- `classifier.py:85` — the startup `pickle.load`;
- `vision_2d.py:136` — the startup Graph2D `torch.load`.
The §1.C producer/latent points gain the same check **before they are wired to activate**:
`auto_remediation.py:301` (before it is ever scheduled), and `auto_retrain.sh` (swap #509's
unconditional block for the membrane call — the *only* place the permanent-closed logic changes, and
only once the signed store exists).

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
  **cannot thereby reach a runtime activation**. It is contained by *different* controls —
  branch-protection (can't merge to main) + CODEOWNERS on L3 paths (#512) + human review. Conflating
  the two would mis-scope the membrane (it is not a defense against branch edits) and under-protect the
  runtime (which needs the proof regardless of who wrote the code).

## 4. Non-goals / explicit exclusions

- Not building the proof store or Track E here — this locks their **contract**.
- Not touching auth defaults here — that is a **separate** L3 design-lock (production-identity model:
  `dependencies.py:8` default `test`, `integration_auth.py:104` header-overrides-subject). Cross-ref,
  don't merge the two.
- Not covering offline training/quantization (§1.D) — they don't activate.
- No cross-customer training; customer corrections never enter a training-readable store (§8 arc).

## 5. Golden matrix the implementation must ship (observed-RED, executed)

| Case | Required result |
|---|---|
| activate a model whose bytes have **no** proof | RED at every §1.A/1.B point |
| **swap the file between hash and load** (TOCTOU) | RED — the loaded bytes are the hashed bytes (§3) |
| **unsigned / wrongly-signed** proof (well-formed but not from a trusted issuer) | RED (no authority — §2.3) |
| **expired / revoked** proof (`not_after` passed, or policy revoked) | RED (stale proof is not valid) |
| requester supplies `family` or `environment=staging` to dodge policy | IGNORED — both are server-owned (§2.3) |
| proof for family A, activate a family-B model of same bytes-len | RED (family mismatch) |
| proof for `staging`, activate in `prod` | RED (environment mismatch) |
| change one byte of the model file after the proof was issued | RED (model_hash mismatch) |
| change a **label** in the manifest, re-derive split_digest | RED (split digest changes — §2.1) |
| rewrite an existing file's **bytes** to duplicate another's content | RED (content in digest — §2.1) |
| bump `evaluator_version` / a threshold | RED (stale proof) |
| a genuine, signed, unexpired, fresh-clone-reproducible proof matching (hash, family, env) | GREEN — the only green |

## 6. What this changes about the current risk statement

Accurate post-#509 status, by threat actor (§3.1 — do not conflate them):

- **The unattended routine** cannot merge to main (`required_reviews=1 + enforce_admins`), cannot
  retrain via `auto_retrain.sh` (#509), and — being a code-generating actor — **has no runtime path to
  any activation point** (it cannot call `/model/reload`). It is contained by branch-protection +
  CODEOWNERS (#512) + review, not by this membrane.
- **A runtime API caller / operator** can still reach the **3 production-reachable activation points**
  — external `/model/reload` (admin token defaulted to `test`) and the two startup loads — none of
  which have proof binding. *This* is what the membrane defends, and it is **entirely unbuilt**.

**So: strong bleeding-control on the code-gen actor; the runtime activation membrane is not built.**
This design-lock defines that membrane. The two owner P0s (stop the routine; establish an independent
L3 reviewer) gate *building* it, but note they address the code-gen actor — the runtime membrane is a
separate, additional need.

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
