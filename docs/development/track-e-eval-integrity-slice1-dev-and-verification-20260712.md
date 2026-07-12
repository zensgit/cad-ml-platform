# Track E slice-1 — evaluation-integrity splitter + artifact — Dev & Verification (2026-07-12)

> **Draft, blocked by L3 (#509), which is blocked by #508.** Parallel dev per the owner's `/goal`
> ("可并行开发, 你来规划开发顺序"). This does NOT merge or activate before L3; it stacks on the L3
> branch and imports the gate's contract so the artifact is gate-conformant by construction.

## 0. What this is / is not

**Is (slice-1, torch-free, fully verified here):** the leakage-safe **split integrity** layer of
Track E (`evaluation-integrity-v2`, PRODUCT_STRATEGY.md §8.1) —

- **content-hash + normalized-family split** so identical drawing content and augmentation/revision
  variants of one drawing never straddle train/holdout (§8.1.1, §8.1.3);
- **conflict quarantine** for identical content with inconsistent labels (§8.1.2);
- a **deterministic `split_digest`** with a `verify` that goes RED when a split is tampered or
  duplicate content is reintroduced — the reproducibility half of the §8.1 exit condition, wired
  **dry-run first** (§8.1.7);
- **gate-conformant artifact assembly**: `build_artifact` emits the `evaluation-integrity-v2`
  artifact the L3 gate (`scripts/eval_integrity_gate.py`) consumes, using the gate's own contract
  constants (single source of truth).

**Is NOT (out of slice-1, needs the model run / follow-up):** real per-class / macro / calibration /
false-duplicate / missed-reuse **metrics** (they require running the model over the holdout — torch +
data, not executable in this environment), and therefore the *full* §8.1 exit condition ("a fresh
clone reproduces the **evaluation result**"). Slice-1 reproduces the **split** deterministically —
the leakage-relevant half — and takes metrics from an eval-results file. Producing real metrics is a
follow-up slice on the model-run lane.

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
- `split_digest(split)` — sha256 over the sorted `(file_path, side)` assignment.
- `build_artifact(rows, metrics)` — assembles the artifact from **imported** contract constants
  (`REQUIRED_VERSION`, `REQUIRED_SPLIT_STRATEGY`, `REQUIRED_METRIC_KEYS`) and asserts it against the
  real `validate_artifact` before returning (drift-proof).
- `verify_reproducible(rows, artifact)` — re-derives the split and compares the digest; RED on any
  drift. CLI: `split` / `build` / `verify`.

## 2. Verification (local, this environment)

| Check | Result |
|---|---|
| Unit tests (`tests/unit/test_track_e_eval_integrity.py`) | **25 passed** |
| — family variant collapse incl. `gear2` / `gear (1)` / `gear - Copy` / NFC-NFD (10 cases) | pass |
| — declared `family` column is authoritative over the filename | pass |
| — unreadable content **and NUL-byte path** → quarantined, not "distinct"/crash (fail-closed) | pass |
| — no family/component straddles train+holdout (incl. bare-digit variants at default fraction) | pass |
| — byte-identical content across families lands on one side | pass |
| — identical content + inconsistent labels → quarantined | pass |
| — split is deterministic (order-independent digest) | pass |
| — `build_artifact` is gate-conformant (passes real `validate_artifact`); missing metric family → rejected | pass |
| **reproducibility discrimination** — `verify` RED on: digest tamper / duplicate reintroduced / row moves family | **pass (3 cases)** |
| CLI smoke (`split`→`build`→`verify`) + built artifact passes the real L3 gate | pass |

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

## 5. Scope boundary (honest, per the portfolio plan)

This is the only cleanly-buildable line right now. The rest of the one-month plan is blocked *not by
effort* but by ownership/gates and stays untouched from here: **cadml #508** needs an independent
reviewer's APPROVE (owner); **metasheet2** D-2 (#4168/#4004) + UI (#4159) are live in other sessions
(collision); **yuantus** #1186 + discussion-auth are blocked by the `adharamans` gh/API 404. Real
Track E **metrics** are the follow-up on the model-run lane.
