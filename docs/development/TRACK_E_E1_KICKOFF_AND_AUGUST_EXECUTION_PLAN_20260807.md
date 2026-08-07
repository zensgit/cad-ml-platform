# Track E E1 Kickoff & August Execution Plan (2026-08-07)

- **Status**: FOR REVIEW. Scheduling/execution guidance only — it does not amend
  `PRODUCT_STRATEGY.md`, the Track E design-lock, or the closeout/safe-park plan. Where anything
  here conflicts with `docs/development/L3_TRACK_E_EVALUATION_INTEGRITY_V2_DESIGNLOCK_20260721.md`
  (merged via #531) or `docs/development/REPO_CLOSEOUT_SAFE_PARK_PLAN_20260721.md` (#530), those
  documents win and this one is wrong.
- **Owner**: repository owner (sole ratifier of merges / enablement / KIND decisions referenced
  below). This document proposes an execution order and schedule; it does not authorize anything
  by itself.
- **Code grounding**: `origin/main@0ffe6ce5` (post #531/#527/#523/#539/#525/#538/#537/#536/#535/
  #532/#529/#528). Re-verify PR/CI state before acting on any item below if this head has aged.
- **Relationship to #530**: #530 owns the strategic framing (three end states, Layer 0–4,
  decision ladder) and the authoritative status tables. This document does not duplicate those
  tables — it adds the one thing #530 intentionally leaves at contract-altitude: a concrete,
  invariant-by-invariant build order for Track E Slice E1, plus a week-by-week schedule for the
  remaining August work. If #530's status tables and this document's status summary ever
  disagree, #530 is authoritative (it is refreshed independently and is the canonical Layer-0/1/3
  tracker).
- **Authorship & routing disclosure**: mechanical execution-planning based on already-ratified
  documents (`PRODUCT_STRATEGY.md`, the Track E design-lock, #530) plus live-verified repository
  state. Makes no security-semantics judgment and no completion claim; the design-lock's own
  invariants and discriminators are the only source of truth for what E1 must do.

---

## 1. Where things stand (brief — see #530 for the authoritative table)

As of `0ffe6ce5`: Phase-A model-activation membrane (C1+C2–C6), production-identity fail-closed
(#517/#538), the two Track-1 seal items (#525 DWG fail-closed, #539 OCR/PointNet honesty), the
Track E design-lock (#531), the Wave-1 audit (#527), and the CI runner-routing fix (#523) are all
merged. Open and relevant: #530 (this plan's sibling, in re-review), #507 (portfolio strategy,
parked at Layer 4 — owner's three questions, not merge-driven), #534 (pilot DWG default-disabled,
draft), and seven dependabot PRs + issue #476.

**Practical consequence**: Track E Slice E1's two design-lock start conditions — (1) the
design-lock is ratified, (2) the ≤2-active-implementation-PR budget has a free L3 runtime slot —
are both satisfied. E1 does not need real data or a model-run environment (that's invariant H,
explicitly deferred out of E1's scope). E1 is ready to start.

---

## 2. Non-E1 work — do first, this week, does not block or depend on E1

| Item | What | Why now |
|---|---|---|
| **Dependabot batch** (#497, #481, #394, #393, #392, #391, #389) | Merge the ones that are CI-green as-is; close the rest as superseded/stale (oldest open since 2026-05-18). Fix issue **#476** (dependabot SHA-pin config) so future dependabot PRs pass Action Pin Guard instead of piling up. | Pure hygiene debt, zero risk, frees reviewer attention before E1's review cycles start. |
| **#534** (pilot DWG default-disabled) | Currently draft; last known state showed a merge conflict against current `main`. Rebase, resolve, take out of draft, run through an L2 review (not L3 — it changes a default, not an auth/data/model-release surface). | Small, bounded, unrelated to E1 — clears before E1's reviews start consuming attention. |

Neither item touches the L3 runtime PR-slot budget (docs/CI/config only, or a small L2 fix), so
neither competes with E1 for the single L3 runtime slot.

---

## 3. Track E Slice E1 — suggested build order

The design-lock (§1) defines invariants **A–H** and (§2) 14 observed-RED discriminators, and
requires (§3) that E1 ship as **one** implementation PR — the line's single L3 runtime PR while
open. What follows is not a new requirement; it is a suggested internal sequencing so the first
submission already covers every invariant and discriminator, rather than arriving in E1 several
review rounds after gaps are found. **Reference, not base**: the closed #510 (leakage-safe split)
and #511 (versioned manifest + provenance) branches still exist
(`origin/claude/track-e-eval-integrity-splitter-20260712`,
`origin/claude/track-e-manifest-provenance-20260712`) and were themselves adversarially reviewed —
useful as algorithm reference. E1 reimplements from `main` against the ratified contract; it does
not revive or rebase onto those branches.

1. **Core: content-hash + union-find grouping — invariant A.** `content_hash(path)` = sha256,
   fail-closed on unreadable bytes or a malformed (NUL-embedded) path → quarantine, never
   "distinct." `family`/`source_id` column is authoritative when present; otherwise
   `normalized_family(path)` with properly *anchored* variant matchers (literal parens, `_aug` as
   a stem suffix, not `_au`+`g*`) — over-collapse is the safe failure mode, under-collapse leaks.
   Union-find merges `(family ∪ byte-identical content)` into a split unit that cannot straddle
   train/holdout; deterministic assignment by hashing
   `"evaluation-integrity-v2|<component>"` (no RNG, no dict-order dependence).
   → covers discriminators **#4** (including the `plate_aux`-vs-`plate` negative/anchoring half)
   and **#5**.
2. **Conflict quarantine — invariant B.** Identical content with inconsistent labels →
   quarantined out of both split sides, surfaced separately. Build this alongside step 1 — it
   shares `content_hash`'s fail-closed error path.
   → covers **#6**, and **#7** (unreadable bytes / NUL path → quarantine, not crash or
   "distinct").
3. **Holdout enforcement — invariant C.** Built on step 1's split units.
   `holdout_fraction` outside `(0,1)` rejected; `eval_eligible` true only when both sides are
   non-empty.
4. **Versioned manifest + containment — invariant D.** Every §8.1.6 field (host-independent
   `sample_id`, dataset-root-relative `locator`/`cache_locator`, `taxonomy_v2_class`, `family`,
   `content_hash`, `split`, `category`, `source`, `license`, `label_authority`); `source`/
   `license`/`label_authority` non-empty-enforced. **Containment check runs before any byte is
   read** — an escaping symlink, `..`-escaping locator, or absolute locator without an explicit
   root is rejected pre-read, under both an explicit `--root` and the repo-relative default.
   → covers **#8** (a spy must assert zero out-of-root `content_hash` calls, both root modes).
5. **Provenance reporting — invariant E.** Explicit `data_origin`/`provenance`/`category` column
   is authoritative; else a boundary-anchored marker positively identifies augmented/synthetic;
   else `"unknown"` — never inferred real. `provenance_complete` false whenever any row is
   unknown. Relatively independent of steps 1–4; can be built in parallel with step 4.
   → covers **#13**.
6. **Reproducible digests + `verify` — invariant F. Build this last** — it re-derives every
   load-bearing field from steps 1–5's output rather than trusting self-declaration, so it needs
   them finished first. `split_digest` = sha256 over sorted `(content_hash, side)` pairs
   (host-independent). `manifest_digest` = sha256 over the canonicalized envelope minus the
   digest field, rows sorted. `verify` pins `SCHEMA_VERSION`, takes `expected_holdout_fraction`
   from the **caller**, never the artifact's self-declared value, independently re-derives
   `split_digest` / full per-row projection / quarantine set / provenance verdict, and binds the
   **closed key-set** so a re-digested artifact cannot smuggle an unbound field (e.g.
   `unlocks_retraining: true`).
   → covers **#1/#2/#3** (the exit-condition discriminators — change-a-split RED,
   reintroduce-duplicate RED, fresh-clone-reproduces GREEN — these are the most load-bearing
   discriminators in the whole contract) and **#9/#10/#11/#12** (each re-digested-tamper
   scenario: per-row, schema, split-policy, key-smuggling).
7. **Dry-run / no-gate-import constraint — invariant G. Self-check throughout, not a separate
   build step.** Before submitting: hardcoded `unlocks_retraining: false`, no `reproducible`
   self-attestation field, **zero** imports of `eval_integrity_gate` anywhere in the E1 module —
   grep for it as a pre-submission checklist item, not something to discover in review.
   → covers **#14** (the fail-closed floor — also assert `scripts/eval_integrity_gate.py
   check()` still has no pass path and `auto_retrain.sh` still exits non-zero before mutation, on
   the E1 branch, as a regression check on the existing floor rather than new E1 behavior).

**Explicitly not in E1** (invariant H, deferred): real per-class/macro/calibration/false-duplicate
metrics, and the two-phase release gate that binds them to `(candidate-model hash, split digest,
evaluator version, thresholds)`. These need a real model run over real holdout data — out of
scope for this torch-free slice.

### Pre-submission checklist (self-verify before requesting review)

Every one of the 14 discriminators should be run as fail-first before the first submission: break
the guarded logic, confirm the discriminator goes RED, restore it, confirm GREEN. This is not new
process — it's the same standard applied to every L3 PR reviewed this cycle (#513, #538, #525,
#539) — but stating it explicitly here because §2 of the design-lock itself requires each
discriminator to be shown non-vacuous, and the fastest way to a low round-count review is to have
already done that self-check before the first submission rather than discover a vacuous
discriminator in round 2.

---

## 4. Review rhythm

E1 will be reviewed at the same rigor as this cycle's L3 runtime PRs: precise-head review in an
isolated worktree, mutation-tested (revert the specific fix, confirm the discriminator goes red;
restore, confirm green), full-suite regression check against `origin/main` (not just the files E1
touches — every PR reviewed this cycle that skipped a full-suite check shipped a collateral
regression it didn't catch itself). Expect multiple rounds; #513 (a comparably-scoped L3
design-lock closing out) took roughly 10 NO-GO/fix rounds over 6 days
(`2026-07-12`→`2026-07-18`) — that is the calibration point, not a target to beat by cutting
verification.

---

## 5. Suggested schedule (week of 2026-08-04 start)

| Week | Focus | Depends on |
|---|---|---|
| **This week** | §2 items (dependabot batch + #476, #534 out of draft); E1 not yet started | Nothing — can run immediately |
| **Next 1–2 weeks** | E1 build, steps 1–5 (A, B, C, D, E) — the layer that doesn't need `verify` finished | This document's build order |
| **Following week** | E1 steps 6–7 (F/`verify`, G self-check), pre-submission discriminator self-check, first submission | Steps 1–5 complete |
| **Remainder of August** | Review rounds (budget multiple, per §4) to GO; owner ratify | First submission |

The only real schedule risk is review round-count, which is a function of how completely the
pre-submission checklist (§3) was actually run, not calendar time — a first submission that has
already fail-first-verified all 14 discriminators converges faster than one that hasn't.

---

## 6. What this plan explicitly does not do

- Does not authorize opening the E1 branch/PR before this document or the underlying design-lock
  status is confirmed current — re-check #530's live status table first if this document has
  aged.
- Does not claim E1 satisfies the full §8.1 exit condition ("a fresh clone reproduces the
  evaluation **result**") — E1 reproduces the **split** deterministically; it emits no metrics.
- Does not touch invariant H, the two-phase release gate, Phase B, or any enablement gate — all
  remain owner-gated and supply-gated exactly as #530 §5.2 describes.
- Does not change the ≤2-active-PR budget or the single-L3-runtime-slot rule — E1 occupies that
  one slot for its entire review lifecycle; no second L3 runtime PR opens while E1 is open.
