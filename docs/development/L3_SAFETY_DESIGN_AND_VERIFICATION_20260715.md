# L3 Model-Activation Safety — Design & Verification (2026-07-15)

> **Scope & honesty frame.** This documents **only what is delivered and CI-verified on `main`**, plus
> the **design-locked-but-unratified** boundary (#513) and the **gated-and-unbuilt** remainder. It does
> **NOT** claim verification for any runtime that is not yet built. The whole L3 effort exists because
> an unattended loop once built a gate ahead of ratification and produced a fake-green; gated
> implementation is listed as **gated**, never as done. `merged != enabled != safe`.

Grounded on `origin/main@c6625831`. Model-routing key (owner directive): **fable5** low/mechanical ·
**sonnet5** medium pattern-following · **opus4.8** high/security-critical.

---

## Part 1 — DELIVERED & CI-VERIFIED ON `main`

| # | PR (merge) | What it delivers | On-`main` evidence |
|---|---|---|---|
| A0-1 | **#518** `4b421103` | Action Pin Guard runs on **every** PR | `.github/workflows/…` (required check) |
| A0-2 | **#512** `e41528de` | CODEOWNERS L3 ownership inventory | `.github/CODEOWNERS` (inventory only — `require_code_owner_reviews=false`, solo-maintainer) |
| A0-3a | **#516** `d4b200ba` | `POST /api/v1/model/reload` → **403 fail-closed** | `src/api/v1/model.py:48` (`status_code=403`), `:78-82` (`raise HTTPException(403, "…disabled (fail-closed)…")`) — the externally reachable arbitrary-deserialization / code-execution risk is **CLOSED**; no execution PoC is claimed |
| A0-3b | **#516** | default `test` creds **fail-close in a production posture** | `src/api/dependencies.py:11` `_production_posture()` (true for `production/prod/staging`), `:29-32` refuse the `test` default in that posture |
| A0-4 | **#516** | **activation-surface enumerator** = CI gate (AST/import-aware) | `scripts/ci/activation_surface_enumerator.py` + `activation_surface.json`; invoked in `.github/workflows/ci.yml` + `ci-tiered-tests.yml` — a new un-annotated load matching a declared idiom **reds CI** |
| A0-5 | **#519** `f2ebe2fa` | enumerator **fail-closed on unparseable files (exit 2)** | same enumerator |

**A0-6 — #521 `a0e517e8` (MERGED):** the enumerator output-truth correction is now on `main`. Its
summary emits "conservatively-classified AST load site(s), reachability audited separately" and
owner-selected (b), fixed-hash/bundle-digest-check or `verify_and_load` — replacing the earlier
"production-reachable" plus rejected-(a) blanket-hard-refuse wording; counts unchanged.

**A0-7 — #522 `c6625831` (MERGED 2026-07-16):** the manifest-truth CI fix is now on `main`. The LIVE
pickle-classifier `load_model` reason was corrected from "startup" to **lazy-first-predict**
(`classifier.py load_model:47`, called only from `predict:124`), with a pinpoint regression guard
forbidding the "startup" over-claim from returning to that site; enumerator counts unchanged
(128/38/11); no runtime change.

**Manifest/count ground truth (executed on `main`):** 128 sites = `gated` 38, `producer` 44,
`offline` 39, `infra` 4, `unmounted` 3. The 38 `gated` across 11 families: pickle-classifier, graph2d, pointnet,
part, part-v16, hybrid, history, vision3d-uvnet, ocr, embedding, anomaly-monitor.

**Reachability is a Wave-1 audit — NOT a hand-asserted live count.** The merged enumerator now prints
the conservative-AST / (b) wording (#521, merged `a0e517e8`). The delivered fact here is the executed count and manifest
classification. The design interprets `gated` **conservatively**, because known latent /
not-yet-proven-reachable among the 38: the `auto_remediation` rollback (no live scheduler); the
pickle-classifier `_reload_model_impl` hot-reload deserialization (reachable only via the now-sealed
`/model/reload`, the latent `auto_remediation`, and offline `finetune_from_feedback`); the two
`MetricsAnomalyDetector.load_models` sites (no `.load_models(` caller in `src/`). The safe contract is
**38 conservatively-`gated` AST entries / 11 families, several latent/unproven, reachability to be
confirmed in Wave 1.**

**Honest production posture today:** external `/model/reload` **sealed (403)**; the `auto_retrain`
**producer disabled (#509)**; the internal `gated` loaders still load **UNPINNED** (proof-unbound); **no
activation membrane; retraining not enabled.**

---

## Part 2 — DESIGN-LOCKED, AWAITING OWNER RATIFICATION (not runtime)

**#513** — the L3 model-activation proof-membrane **design-lock** (changes no runtime).

- Pinned head: the ratifiable SHA is the latest green #513 head (owner ratifies it; the agent does not).
- **Owner decision (b) — static fixed-hash artifact activation.** A `gated` loader activates ONLY from
  a server-owned artifact resolved from a controlled store: no caller path, no env path-swap, no
  dynamic replacement, offline-only. **Two artifact KINDs:** *single-file* (read once → `SHA-256(bytes)`
  == pinned → load THOSE bytes, TOCTOU-safe) and *bundle/tree* (versioned **`tree-digest-v1`** — a fixed
  canonical encoding so Phase A & Phase B agree — over a **frozen** copy/RO-snapshot, never a bind of
  the mutable source → == pinned → load from the frozen snapshot).
- **Track-E-before-promotion invariant.** Phase A is baseline containment only: before Track E, only the
  exact owner-reviewed already-in-service `(logical_activation_id, artifact_id, kind, digest)` tuple
  passes. Any tuple-field change is a promotion/contract migration and is refused; an unprovable
  baseline stays degraded.
- **Shared bounded pre-read.** Phase A and Phase B reject wrong KIND/type/magic, an oversized single
  file, or a bundle exceeding file-count / per-file / aggregate-byte limits before full read/copy/load.
- **Producer/activator separation.** `auto_retrain.sh` stays #509 fail-closed until Track E and never
  calls the activation membrane; Track E may produce a candidate + versioned evaluation artifact,
  while runtime activation remains a separate guarded action.
- **Guard-verification is a mechanically-checkable structure** (not AST-unverifiable call-graph
  domination): every `gated` raw load lives ONLY inside a canonical `load_pinned_file`/`load_pinned_bundle`
  wrapper (or a same-function lexical assert); deleting the wrapper is **observed-RED**.
- **Phase B** binds a versioned `artifact_kind`+`artifact_digest` (single-file SHA or `tree-digest-v1`)
  so ocr/embedding keep coverage; signed proof envelope + key-custody (HSM/human-gated, outside CI) +
  revocation/expiry + LKG re-validation + append-only audit.
- **Verification status of the DESIGN (not runtime):** prior critic evidence was invalidated by later
  material edits. The ratifiable closing head requires a fresh three-lens review recorded in the PR:
  canonical `PRODUCT_STRATEGY.md` cross-check; internal consistency/self-contradiction; execution and
  source-fact verification. No runtime observed-RED is claimed by this docs-only PR.
- **Status: PROPOSED.** The owner alone ratifies; the critic is **evidence, not approval**.

**This is the gate for all of Part 3.**

---

## Part 3 — GATED & UNBUILT (each ships its OWN Dev&V MD after its gate clears)

| Phase | Items (model routing) | **Gate** | Verification-to-ship (observed-RED) |
|---|---|---|---|
| **Wave 1 (audit + build prep)** | Per-site **logical-reachability audit** of the 38 conservatively-`gated` sites (confirm live vs latent/unproven) (**sonnet5** ∥ **opus4.8** for the auth-adjacent ones) | #513 ratified | each `gated` site labelled live/latent/unreachable with the caller path; latent/unreachable ones gated-before-wired |
| **Phase A** baseline-only static fixed-hash | C1 `assert_fixed_hash`/`load_pinned_*` core (**opus4.8**) ∥ C2 per-family wiring (**sonnet5**) ∥ C3 baseline manifest (**sonnet5**) ∥ C4 degraded/503 (**sonnet5**) ∥ C5 enumerator guard-assertion (**sonnet5**/opus AST) ∥ C6 golden matrix (**opus4.8**) | #513 ratified + Wave-1 audit | §5 Phase-A golden: exact baseline match GREEN; pre-Track-E tuple-field change RED; hash-miss→degraded/503; caller-path/env-swap/hot-swap REJECTED; shared size/type bounds RED; symlink/`..` escape RED; single-file & bundle TOCTOU RED; new un-annotated loader CI RED |
| **Track E** rebuild | leakage-safe split + versioned manifest + real §8.1.4 metrics (**opus4.8** ∥ **sonnet5** ∥ **fable5** fixtures) | **real data + model-run environment** | tamper/containment/binding suite re-derived on `main` + real metrics reproduced on a holdout run |
| **Phase B** signed-proof | `verify_and_load` + signed envelope + key custody + revocation/expiry + LKG re-validation + audit (**opus4.8**) | **Track E + signing-key custody (HSM / human-gated signer outside CI)** | §5 Phase-B golden: no-proof/unsigned/expired-revoked/TOCTOU/family-env-mismatch RED; single signed match GREEN |
| **§8.1.7 + enablement** | full tamper observed-RED, cross-family matrix, staging replay; **separate enablement PR** | **A–E done + owner enablement decision** | staging replay + complete cross-family matrix green; re-enable = replacing a body, never a flag |

**Macro-order (reviewer-locked):** Phase A → Track E → Phase B → enablement.

**Two hard external constraints that cannot be manufactured around:** Track E's five metrics need
**real data + a model-run environment** (producing them without = fabricated metrics); Phase B's proofs
need **signing keys / HSM custody** (producing them without = forged signatures). Either would be the
literal fake-green.

---

## Part 4 — Owner action map

1. Complete and record #513's fresh **three-lens review on the exact closing head**, then owner-ratify
   that green head → unblocks Wave-1 audit + **Phase A baseline containment**.
2. ~~Merge #521~~ **DONE** (`a0e517e8`); ~~land #522 manifest-truth~~ **DONE** (`c6625831`) — enumerator output-truth + corrected LIVE-load reason on `main`; Part-1 evidence is caliber-accurate.
3. Provision **real data + model-run env** → unblocks **Track E**.
4. Provision **signing-key custody (HSM/human-gated)** → unblocks **Phase B** (after Track E).
5. **Owner enablement decision** (post A–E) → §8.1.7 + the separate enablement PR.

Companion planning doc: the Wave 0–5 remaining-work roadmap (per-item model routing + gates).
