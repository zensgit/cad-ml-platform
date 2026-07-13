# Track E slice-1 — leakage-safe split + dry-run split artifact — Dev & Verification (2026-07-12)

> **Targets `main` (#509 merged as `8ff94175`).** This module is **fully decoupled from the L3
> gate**: it imports nothing from it and cannot mint any unlock (the gate has no pass path by owner
> design). It does NOT enable retraining — Phase-A posture: *Safety foundation complete; retraining
> remains disabled.*

## 0. What this is / is not

**Is (slice-1, torch-free, fully verified here):** the leakage-safe **split integrity** layer of
Track E (the split artifact schema is `evaluation-integrity-split-v1`, PRODUCT_STRATEGY.md §8.1) —

- **content-hash + normalized-family split** so identical drawing content and augmentation/revision
  variants of one drawing never straddle train/holdout (§8.1.1, §8.1.3);
- **conflict quarantine** for identical content with inconsistent labels (§8.1.2);
- a **deterministic `split_digest`** with a `verify` that goes RED when a split is tampered or
  duplicate content is reintroduced — the reproducibility half of the §8.1 exit condition, wired
  **dry-run first** (§8.1.7);
- **dry-run split artifact, fully decoupled from the L3 gate**: `build_split_artifact` emits a split
  summary + `split_digest` for inspection and the §8.1.7 `verify` check. It carries a hardcoded
  `unlocks_retraining: false` and has **no metrics, no `reproducible` field, and no import of the
  gate**. The owner review found an artifact "you may proceed" token is an *unbound attestation*
  (an artifact built for dataset A could green-light dataset B), so the L3 gate
  (`scripts/eval_integrity_gate.py`) was rewritten to be **unconditional — no pass path at all**, and
  this module was decoupled from it: nothing here can mint an unlock. Re-enabling retraining is a
  separate, later, owner-gated mechanism that binds a validation-manifest digest + a promoted-model
  hash. (Supersedes the earlier `build_artifact`/`exit_condition_met` design.)

**Is NOT (out of slice-1, needs the model run / follow-up):** real per-class / macro / calibration /
false-duplicate / missed-reuse **metrics** (they require running the model over the holdout — torch +
data, not executable in this environment), and therefore the *full* §8.1 exit condition ("a fresh
clone reproduces the **evaluation result**"). Slice-1 reproduces the **split** deterministically —
the leakage-relevant half — and emits no metrics at all. Producing real metrics (and the two-stage
release gate that binds them to a candidate model) is a later phase on the model-run lane.

## 1. Design

`scripts/track_e_eval_integrity.py` (stdlib only):

- **Family key** — an explicit manifest `family`/`source_id` column is used when present
  (**authoritative**, §8.1.6). Only when absent does it fall back to `normalized_family(path)`:
  Unicode-NFC normalized, then augmentation/revision/copy/OS-duplicate markers stripped (`_aug*`,
  `_rot*`, `_flip*`, `_scale*`, `_v\d+`, `_rev\d+`, `_copy`/`- Copy`, `(1)`, bare trailing digits
  `gear2`). The fallback deliberately **errs toward over-collapse**: for a leakage guard over-collapse
  is safe (keeps a drawing's variants together, costs holdout diversity), under-collapse leaks — so a
  manifest that provides a real `family` column is strictly preferred over the heuristic.
- `content_hash(path)` — sha256 of file bytes. **Fail-closed**: unreadable/missing bytes (`OSError`)
  *and* malformed paths such as an embedded NUL byte (`ValueError`) raise `QuarantineRow` and the row
  is quarantined — never silently treated as "distinct" (which would let identical content leak).
- `compute_split(rows)` — split unit = a **union-find component** of (family ∪ identical-content), so
  two differently-named files with byte-identical content are merged into one unit and cannot
  straddle. Each component is assigned to `holdout`/`train` deterministically by hashing
  `"evaluation-integrity-v2|<component>"` — no RNG, no dict-order dependence.
- `split_digest(split)` — sha256 over the sorted `(content_hash, side)` pairs — **host-independent**
  (a fresh clone at a different absolute path reproduces the same digest; a pure rename of identical
  bytes does not change it, which is correct for split integrity).
- `build_split_artifact(rows)` — a **dry-run** split report (`evaluation-integrity-split-v1`):
  hardcoded `unlocks_retraining: false`, `eval_eligible` (both sides populated), no metrics, no
  `reproducible` field, and no import of the gate. Nothing here can re-enable retraining.
- `verify_reproducible(rows, artifact)` — re-derives the split and compares the digest; RED on any
  drift. CLI: `split` / `build` / `verify`.

## 2. Verification (local, this environment)

| Check | Result |
|---|---|
| Unit tests (`tests/unit/test_track_e_eval_integrity.py`) | **32 passed** |
| — family variant collapse incl. `gear2` / `gear (1)` / `gear - Copy` / NFC-NFD (10 cases) | pass |
| — declared `family` column is authoritative over the filename | pass |
| — unreadable content **and NUL-byte path** → quarantined, not "distinct"/crash (fail-closed) | pass |
| — no family/component straddles train+holdout (incl. bare-digit variants at default fraction) | pass |
| — byte-identical content across families lands on one side | pass |
| — identical content + inconsistent labels → quarantined | pass |
| — split is deterministic (order-independent digest) | pass |
| — `build_split_artifact` carries `unlocks_retraining=false`; module imports nothing from the gate | pass |
| — **fresh-clone-stable digest**: identical bytes under two different absolute roots → same digest | pass |
| — invalid `holdout_fraction` (≤0, ≥1) rejected; `eval_eligible` reflects both sides populated | pass |
| **reproducibility discrimination** — `verify` RED on: digest tamper / duplicate reintroduced / row moves family | **pass (3 cases)** |
| CLI smoke (`split`→`build`→`verify`) — dry-run split artifact (`unlocks_retraining=false`) | pass |

**Independent adversarial review (Sonnet-5 lane)** — hunted for split-leakage. Found and **fixed**
before push: (HIGH) family under-collapse straddle for `gear2.dxf` / `gear (1).dxf` etc. → strengthened
normalizer + authoritative `family` column + regression tests; (MEDIUM) NUL-byte path escaped the
fail-closed catch → now caught. Could-not-refute: moved-row-hides-in-digest, fraction-mismatch masking
a tamper, cross-run determinism. Documented scope gap (LOW, not fixed): `verify_reproducible`
authenticates the **split** only, not the artifact's `metrics`/counter fields — acceptable because
`verify` is a non-blocking dry-run and the blocking L3 gate never reads those counters; metrics
integrity is the model-run lane's job.

**Discrimination**: the reproducibility tests fail if the digest stops covering a moved row — not
vacuous. **Local ≠ CI**: the repo's fixtures assume Python 3.11; formal CI (added below) is the
authority.

## 3. CI

`ci.yml` + `ci-tiered-tests.yml`: a curated step runs `test_track_e_eval_integrity.py`. The
reproducibility `verify` is **dry-run / reporting only** here (§8.1.7 — "path-filtered dry-run in
open PRs before making the new gate blocking"); flipping it to a blocking pre-retrain check is a
later owner-gated step, not this slice.

## 4. Model routing (per the goal)

Correctness core (the leakage guard + reproducibility digest) built on the strong lane (Opus 4.8) and
NOT fragmented across agents; an **independent adversarial review** ran on the **Sonnet-5** lane
(the split-leakage failure mode is exactly what a second, cheaper lens should attack). No
low-difficulty mechanical bulk was separable without integration risk, so no Fable-5 delegation this
slice.

## 5. Scope boundary (honest)

#508 and #509 are both merged; this PR targets `main` directly. Real Track E **metrics** (§8.1.4),
the two-stage release gate (manifest/split digests + candidate SHA-256 + evaluator version +
thresholds), and any re-enablement of retraining are **Phase B** — they require a real data/model
environment and owner threshold decisions, and are deliberately NOT claimed here. Phase-A posture:
*Safety foundation complete; retraining remains disabled.*
