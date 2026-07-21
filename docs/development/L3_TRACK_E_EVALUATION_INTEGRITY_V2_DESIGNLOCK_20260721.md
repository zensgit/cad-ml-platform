# L3 Design-Lock — Track E: evaluation-integrity-v2

**Date**: 2026-07-21 · **Status**: PROPOSED (for-review; do NOT self-merge; owner ratifies)
**Rigor**: L3 (`PRODUCT_STRATEGY.md` §7.1 — model-release gates) · **Grounded on**: `origin/main@7160694d` (#509/#513/#526 merged)
**Authority**: `PRODUCT_STRATEGY.md` §8.1 (Track E: evaluation-integrity-v2) + §8 hard-ordering constraint
**Ordering**: owner half-month plan Day 3–5. This design-lock is the **prerequisite** for Track E Slice E1 (Day 6–9); E1 rebuilds from `main` against **this ratified contract**, it does not revive the closed PRs.

> **This is a proposal, not an implementation.** It changes no runtime, ships no gate, and unlocks no
> retraining. It locks the contract that Slice E1 (dry-run) must satisfy so the contract can be ratified
> before code is (re)written — "propose, don't build". Unattended routines may not author or merge this.
> Solo-maintainer L3 review protocol (`L3_MODEL_ACTIVATION_MEMBRANE_DESIGNLOCK_20260712.md` §"Solo-maintainer
> L3 review protocol") applies verbatim: an isolated critic supplies evidence; the human owner alone ratifies
> and pins a head; `require_code_owner_reviews` stays false.

> **Prior art — this locks what was already settled, it does not rediscover it.** Two adversarially-reviewed
> implementations were built and then **CLOSED** under the L3 hold (design-lock must precede runtime): **#510**
> (`claude/track-e-eval-integrity-splitter-20260712`, slice-1 = leakage-safe split) and **#511**
> (`claude/track-e-manifest-provenance-20260712`, slice-2 = versioned manifest + provenance). They are closed,
> not merged; their branches are the reference, not a base. This lock distils their settled invariants + the
> deferred metrics/release-gate contract into one authority. E1 reimplements from `main`.

---

## 0. Why this is L3 and what it must not do (empirically, at `7160694d`)

The retraining/model-promotion path is L3 (`PRODUCT_STRATEGY.md` §7.1 — "model-release gates"). Today it is
held **fail-closed** by `scripts/eval_integrity_gate.py`, whose `check()` has **no pass path at all** and whose
module docstring records *why*:

- `scripts/auto_retrain.sh` stamps "Ready for deployment" at a 91.5% threshold against
  `data/manifests/golden_val_set.csv`, which carries **262/914 (28.7%)** rows whose *bytes* are identical to
  training rows (`PRODUCT_STRATEGY.md` §5.2). Accuracy on it is **not release-grade**.
- An earlier "evaluation-integrity artifact" gate was itself fake-green two ways: the token was **unbound**
  (carried no digest of the validation manifest actually used, nor the hash of the model actually promoted — a
  confused deputy: an artifact for dataset A green-lights a model on dataset B), and the sanctioned producer
  emitted a *passing* artifact with `holdout_rows: 0`, all-zero metrics, hardcoded `reproducible: true`.

**The lock's non-negotiable floor:** nothing in Track E may re-open that gate. The gate stays unconditional;
re-enablement is a **code change to `check()`**, never an argument, env var, file, artifact, or queue-row count
(§8 hard-ordering). Slice E1 is **dry-run / reporting only** and imports nothing from the gate. `auto_retrain.sh`
continues to exit non-zero before any mutation or training.

---

## 1. Invariants the implementation must establish (the contract)

Each maps to a `PRODUCT_STRATEGY.md` §8.1 clause. "Authoritative" = an explicit manifest column wins; a
heuristic is a **fail-safe fallback**, never an override.

### A. Split integrity — content-hash + normalized-family, not path-only (§8.1.1, §8.1.3)
- **Family key** = the manifest `family`/`source_id` column when present (**authoritative**, §8.1.6). Absent →
  `normalized_family(path)`: Unicode-NFC, then strip augmentation/revision/copy/OS-duplicate markers
  (`_aug*`, `_rot*`, `_flip*`, `_scale*`, `_v\d+`, `_rev\d+`, `_copy`/`- Copy`, `(1)`, bare trailing digits).
  The fallback must **err toward over-collapse**: for a leakage guard, over-collapse (keeping a drawing's
  variants together) is safe; under-collapse leaks. A real `family` column is strictly preferred.
- **Split unit** = a **union-find component** of `(family ∪ byte-identical-content)`. Two differently-named
  byte-identical files merge into one unit and **cannot straddle** train/holdout. Assignment is deterministic
  by hashing `"evaluation-integrity-v2|<component>"` — **no RNG, no dict-order dependence**.
- `content_hash(path)` = sha256 of file bytes, **fail-closed**: unreadable/missing bytes (`OSError`) *and*
  malformed paths (embedded NUL, `ValueError`) → the row is **quarantined**, never silently "distinct" (which
  would let identical content leak).

### B. Conflict quarantine (§8.1.2)
- Identical content with **inconsistent labels** → quarantined, excluded from both split sides; surfaced
  separately for audit. Quarantine is fail-closed (see A) — quarantine, don't guess.

### C. Holdout enforced, not advisory (§8.1.3)
- Customer-family or time-based holdout. No family/component may straddle train and validation (enforced by the
  union-find split unit in A). `holdout_fraction` outside `(0,1)` is **rejected**. `eval_eligible` is true only
  when **both** sides are populated (an empty holdout is not a valid evaluation input).

### D. Versioned manifest — every §8.1.6 field, fresh-clone portable (§8.1.6)
- One enriched record per surviving row carrying **all** §8.1.6 fields: host-independent `sample_id`, a
  **dataset-root-relative** `locator` (+ `cache_locator`), `taxonomy_v2_class`, `family`, `content_hash`,
  `split`, `category`, `source`, `license`, `label_authority`. **No absolute run path may enter the manifest.**
- `source` / `license` / `label_authority` are **non-empty-enforced** (blank → fail-closed).
- **Containment (fail-closed, pre-read):** every file/cache path is `resolve()`-contained **before** any bytes
  are read — an escaping symlink, a `..`-escaping locator, or an absolute locator with no explicit dataset root
  is rejected **without `content_hash` ever opening the out-of-root file**. Holds for both an explicit `--root`
  and the repo-relative default root.

### E. Provenance reporting — real / synthetic / augmented, never inferred as real (§8.1.5)
- An explicit `data_origin`/`provenance`/`category` column is **authoritative**. Else a boundary-anchored marker
  positively identifies "augmented" or "synthetic". **Else "unknown"** — an unmarked, undeclared sample is
  **never inferred to be real**. `provenance_complete` is false whenever any row is unknown, so an
  incomplete-provenance dataset cannot be treated as a clean evaluation input. `report_by_category` maps any
  missing/illegal category to "unknown" (never "real") and surfaces the illegal rows.

### F. Reproducible digests, verified against TRUSTED config — not self-declaration (§8.1 exit condition)
- `split_digest` = sha256 over sorted `(content_hash, side)` pairs — **host-independent** (a fresh clone at a
  different absolute path reproduces it; a pure rename of identical bytes does not change it — correct for split
  integrity, *not* claimed as invariance to anything else, per #510 audit P3).
- `manifest_digest` = sha256 over the **entire canonicalized manifest envelope minus the digest field**, rows
  list-sorted → order-independent (whole envelope, not per-row only).
- **`verify` trusts the caller, never the artifact.** It PINS `schema_version == SCHEMA_VERSION`; takes the
  holdout policy from `expected_holdout_fraction` (caller/default), **not** the artifact's self-declared value;
  and independently **re-derives** every load-bearing field from the rows — `split_digest`, the **full per-row
  projection**, the quarantine `(locator, reason_code)` set, the aggregate provenance verdict — rather than
  trusting the digest self-check alone (a re-digesting attacker defeats a naive self-check). It binds the
  **CLOSED key-set**: top-level / per-row / per-quarantine keys must exactly equal what `build` emits, so a
  re-digested manifest **cannot smuggle an unbound key** (e.g. `unlocks_retraining: true`).

### G. Dry-run first — no gate-conformant artifact, no unlock path (§8.1.7, §8 hard-ordering)
- The split/manifest artifacts are **inspection/audit tooling only**: hardcoded `unlocks_retraining: false`, no
  `reproducible` self-attestation field, **no import of `eval_integrity_gate`**. Nothing E1 emits can mint an
  unlock. The reproducibility `verify` runs **dry-run / reporting** in open PRs (path-filtered); flipping it to
  a blocking pre-retrain check is a **separate, later, owner-gated** step (Day 10 observation precedes any
  required status).

### H. Deferred to the model-run lane — explicitly NOT in E1 (§8.1.4)
- Real per-class / macro / calibration / false-duplicate / missed-reuse **metrics** require running the model
  over the holdout (torch + data) — out of scope for the torch-free dry-run slice. The **two-phase release gate**
  that binds metrics is the future replacement for `eval_integrity_gate.check()`:
  - **pre-training:** validated manifest + content/family/label digest + **non-empty** holdout;
  - **post-training:** result bound to `(candidate-model hash, split digest, evaluator version, thresholds)`
    before any "Ready for deployment" is emitted.
  E1 must **not** claim the full §8.1 exit condition ("a fresh clone reproduces the **evaluation result**"); it
  reproduces the **split** deterministically — the leakage-relevant half — and emits **no metrics**.

---

## 2. Verification contract — the observed-RED discriminators E1 must ship

At **contract altitude** (what must go red, not how). Each is a fail-first test that must fail against a
regressed implementation; a green run of these is the acceptance evidence, and each must be shown non-vacuous
(fails when the guarded logic is removed).

**Exit-condition discriminators (§8.1 exit condition — mandatory):**
1. **Change a split → required (dry-run) `verify` RED.** Move one row to the other side / flip its `split` →
   `verify` re-derives from rows and raises. (Non-vacuous: passes only because the digest covers the moved row.)
2. **Reintroduce duplicate content → RED.** Add a byte-identical row on the opposite side; the union-find unit
   must merge them and `verify` must reject the straddle.
3. **Fresh clone reproduces.** Build the split/manifest at absolute root A; verify the **same artifact** at a
   different root B → PASS. No absolute path in the artifact; A-build → B-verify green.

**Leakage discriminators (§8.1.1–.3):**
4. Family variant collapse: `gear2` / `gear (1)` / `gear - Copy` / NFC-vs-NFD all land in one unit and never
   straddle (the #510 audit HIGH — under-collapse straddle — must stay closed).
5. Byte-identical content across differently-named families lands on **one** side.
6. Identical content + inconsistent labels → **quarantined**, not split.
7. Fail-closed content: unreadable bytes **and** a NUL-byte path → **quarantined**, not "distinct"/crash
   (the #510 audit MEDIUM must stay closed).

**Manifest-integrity discriminators (§8.1.5–.6, re-digested attacker — from #511):**
8. **Containment:** file/cache outside the dataset root, `..`-escaping or absolute locator → fail-closed at
   `build` **and** rejected at `verify` even when re-digested; `content_hash` opens **zero** out-of-root files
   (spy asserts zero calls). Holds under explicit `--root` and the default root.
9. **Re-digested per-row tamper** (`category` unknown→real / `split` train→holdout / forged `taxonomy` /
   redirected `locator` / forged `content_hash`, each with a recomputed digest) → RED via full-row binding.
10. **Re-digested schema tamper** (`schema_version`→hostile) → RED: `verify` PINS `SCHEMA_VERSION`.
11. **Re-digested split-policy tamper** (full rebuild at `holdout_fraction=0.9`) → RED: `verify` uses the
    **trusted** policy, not the artifact's declared fraction; a legit non-default build verifies only when the
    caller declares the matching `expected_holdout_fraction`.
12. **Re-digested key smuggling** (an unbound top-level / row / quarantine key, incl. `unlocks_retraining: true`)
    → RED: `verify` binds the CLOSED key-set to exactly what `build` emits.
13. **Provenance never inferred real:** unmarked/undeclared row classifies as **unknown**; `provenance_complete`
    false; `report_by_category` never defaults to "real".

**Fail-closed floor (§8 hard-ordering — mandatory):**
14. E1 imports **nothing** from `eval_integrity_gate`; `scripts/eval_integrity_gate.py check()` still has **no
    pass path**; `auto_retrain.sh` still exits non-zero before mutation/training. A test asserts the artifact
    carries `unlocks_retraining: false` and that no E1 symbol can mint an unlock.

---

## 3. Scope boundary (honest)

- **In E1 (Day 6–9, dry-run, torch-free):** invariants A–G + discriminators 1–14 above.
- **NOT in E1 (deferred, needs the model run / later owner decision):** real metrics (§8.1.4, invariant H), the
  two-phase release gate that binds them, and any re-enablement of retraining (a code change to `check()`, owner
  authorized, bound to a versioned reproducible artifact — never a flag). **Phase-A posture is unchanged: safety
  foundation complete; retraining remains disabled.**
- **PR-slot discipline:** E1 is **one** implementation PR and the line's **single** L3 runtime PR; it cannot
  start until (a) the owner ratifies this design-lock, and (b) the ≤2 active-implementation-PR budget frees a
  slot (currently held by #528/#529). This design-lock is **docs-only** and does not consume that budget.

---

## 4. Ratification checklist (owner)

- [ ] Triple-mirror this contract: **canonical** (every invariant traces to a `PRODUCT_STRATEGY.md` §8.1 clause),
      **internal-consistency** (no invariant contradicts the §8 fail-closed floor), **execution** (the closed
      #510/#511 branches demonstrate A–G are implementable torch-free and the discriminators fire).
- [ ] Confirm the fail-closed floor: E1 must never import the gate, mint an unlock, or claim the full exit
      condition. Metrics stay deferred.
- [ ] Confirm ordering: E1 starts only after this is ratified **and** a PR slot frees.
- [ ] Ratify and pin a head, or return findings for revision. **The drafting agent does not ratify or merge; a
      genuinely fresh context (or the owner) performs the ratifying review — the author must not "review" its own
      draft as independent evidence.**

---

*Independent-context evidence for the owner's ratification decision. `merged != enabled != safe to enable`.*
