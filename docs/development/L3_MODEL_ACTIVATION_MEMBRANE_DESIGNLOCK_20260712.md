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

#509 made `scripts/auto_retrain.sh` unconditionally fail-closed. That was correct **but narrow**: it
closed **one activation site of one model family**. An earlier claim that it "closed retraining on
main" was **overstated** — corrected here with the full activation map.

Two independent corrections from review, both load-bearing:

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

### 1.B Startup / runtime-config activation
| Point | Family | Evidence | Current guard |
|---|---|---|---|
| `CLASSIFICATION_MODEL_PATH` → `pickle.load` | pickle classifier | `src/ml/classifier.py:22,57,85,288` | pickle magic-number check only; **no proof binding** |
| `GRAPH2D_MODEL_PATH` → `torch.load` in `Graph2DClassifier` | Graph2D | path `src/ml/vision_2d.py:41`, load `src/ml/vision_2d.py:136`; `src/main.py:61` only *reads the flag*, does not load | **no proof binding** |

### 1.C Internal auto-activation (no human in the loop)
| Point | Family | Evidence | Current guard |
|---|---|---|---|
| `auto_remediation._action_rollback_model` → `reload_model(prev_path)` | pickle classifier | `src/ml/monitoring/auto_remediation.py:301` | **none.** Latent: `AutoRemediation` is defined and exported (`src/ml/monitoring/__init__.py:52`); a live scheduler calling `evaluate_and_act` was **not** confirmed on main — the membrane must cover it **before** it is ever scheduled. |
| `finetune_from_feedback` → `reload_model(force=True)` after retrain | pickle classifier | `scripts/finetune_from_feedback.py:302` | **none** |
| `auto_retrain.sh` → Graph2D checkpoint promote | Graph2D | `scripts/auto_retrain.sh` | ✅ **#509 fail-closed** (the only currently-closed point) |

### 1.D Explicitly OUT of the membrane (documented, not covered)
- **Offline tools** — `scripts/quantize_*.py`, `scripts/finetune_graph2d_*.py`, training scripts:
  they *produce* artifacts, they do not *activate* a model in a running production service. Out of
  scope by definition; a produced artifact only becomes live by passing 1.A–1.C.
- **Unmounted serving scaffold** — `src/ml/serving/worker.py`, `grpc_service.py`: imported by **0**
  mounted routes (verified). **Not a production boundary today.** Action: mark `inert` (or delete),
  and if ever mounted it is automatically promoted into 1.A and must gain a proof check first.

**Coverage gap today:** of the 6 real activation points in 1.A–1.C, **exactly 1 is closed** (#509,
Graph2D via auto_retrain). The pickle-classifier family has **4 open activation points** (one of them
external, admin-token-defaulted); Graph2D has **1 open** startup-config activation.

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
  evaluator_version : pinned                               # which metric code produced the numbers
  thresholds        : the gate values applied              # per-class / macro-F1 / ECE / dup / miss
  environment       : deployment env id                    # a proof for staging ≠ prod
  metrics           : per-class, macro-F1, ECE, false-duplicate, missed-reuse, source-stratified
}
```

An activation point computes `sha256(model-about-to-load)` and requires a proof whose `model_hash`,
`model_family`, and `environment` all match. **No match → fail-closed.** This is what makes the token
*bound* — the defect that sank the first gate (an artifact for dataset A green-lighting a model on
dataset B) cannot recur, because the proof names the model bytes and family.

---

## 3. The membrane (one choke-point, enforced at every production-reachable activation)

A single function — call it `assert_release_proof(model_path, family, env)` — that:
1. hashes the bytes at `model_path`;
2. looks up the proof for `(model_hash, family, env)` in a **read-only, out-of-band proof store**
   (a model may not carry its own passing proof — that is self-attestation again);
3. re-validates the proof's `evaluator_version`/`thresholds` are current;
4. **raises on any miss** (fail-closed), with a `§`-pointer message.

Every point in §1.A–1.C calls it **before** the model is loaded/served:
- `model.py:71` before `reload_model`;
- `classifier.py` reload/startup load before `pickle.load`;
- `vision_2d.py:136` before the Graph2D `torch.load`;
- `auto_remediation.py:301` before the rollback reload (a *previous* model still needs a valid proof);
- `finetune_from_feedback.py:302` before its reload;
- `auto_retrain.sh` — replace #509's unconditional block with the membrane call (this is the *only*
  place the current permanent-closed logic is swapped, and only once the store exists).

Until the proof store and Track E exist, **the membrane's default implementation is #509's: raise
unconditionally.** Re-enablement is replacing the body, not adding a flag. (`merged != enabled !=
safe to enable`, §7.2.)

---

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
| activate a model whose bytes have **no** proof | RED at every §1.A–1.C point |
| proof exists for family A, activate a family-B model of same bytes-len | RED (family mismatch) |
| proof exists for `staging`, activate in `prod` | RED (environment mismatch) |
| change one byte of the model file after the proof was issued | RED (model_hash mismatch) |
| change a **label** in the manifest, re-derive split_digest | RED (split digest changes — §2.1) |
| rewrite an existing file's **bytes** to duplicate another's content | RED (content in digest — §2.1) |
| bump `evaluator_version` / a threshold | RED (stale proof) |
| a genuine, fresh-clone-reproducible proof matching (hash, family, env) | GREEN — the only green |

## 6. What this changes about the current risk statement

Accurate post-#509 status: the unattended routine **cannot merge to main** (`required_reviews=1 +
enforce_admins`) and **cannot retrain via `auto_retrain.sh`** (#509). It can still, in principle,
reach the **4 open pickle-classifier activation points and the open Graph2D startup load** — none of
which have proof binding, and one of which (`/model/reload`) is external with a defaulted admin token.
**Strong bleeding-control, not a closed membrane.** This design-lock defines the membrane; the two
owner P0s (stop the routine; establish an independent L3 reviewer) remain prerequisites to building it.

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
