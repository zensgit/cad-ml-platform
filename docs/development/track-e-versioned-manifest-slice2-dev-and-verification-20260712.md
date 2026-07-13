# Track E slice-2 — versioned manifest + real/synth/augmented reporting — Dev & Verification (2026-07-12)

> **Draft, stacked on Track E slice-1 (#510); #509 is merged (`8ff94175`).** Retargets to `main` and
> runs full CI after #510 merges. Dry-run tooling only — does NOT unlock retraining (the L3 gate is
> unconditional). Phase-A posture: *Safety foundation complete; retraining remains disabled.*

## 0. What this is / is not

**Is (slice-2, torch-free, verified here):**
- **§8.1.6 versioned manifest** — `build_versioned_manifest` emits one enriched record per row that
  survives slice-1's leakage-safe `compute_split`, carrying **every §8.1.6 field**: a host-independent
  `sample_id`, a dataset-root-relative `locator` + `cache_locator` (NO absolute run path enters the
  manifest), `taxonomy_v2_class`, `family`, `content_hash`, `split`, `category`, `source`, `license`,
  `label_authority` — plus a deterministic `manifest_digest` and the slice-1 `split_digest`.
- **§8.1.5 real / synthetic / augmented reporting** — `categorize` classifies each row from
  filename/family markers (boundary-anchored regex, no I/O, no RNG); `report_by_category` breaks down
  counts per category, per (category × split), per (category × class).
- **tamper/drift detection** — `verify_manifest` re-derives the manifest from the rows and raises on
  any `manifest_digest`/`split_digest` mismatch (content change, added/removed/moved row). Dry-run
  posture (non-blocking), same as slice-1.

**Is NOT:** model **metrics** (§8.1.4) still need the model run — a follow-up. This reuses slice-1's
already-adversarially-reviewed leakage-safe split rather than re-deriving it, so the manifest can
never drift from slice-1's split. (The L3 gate is unconditional — it trusts no artifact; this is inspection/audit tooling only.)

## 1. Design

`scripts/track_e_manifest.py` (stdlib), importing slice-1's primitives (`compute_split`,
`content_hash`, `_family_key`, `split_digest`, `QuarantineRow`, `IntegrityError`):

- `categorize(row)` — an explicit `data_origin`/`provenance`/`category` column is **authoritative**;
  else a boundary-anchored marker positively identifies "augmented" (aug/rot/flip/noise/jitter/scale +
  morphological variants) or "synthetic" (synthetic/synth/generated/gan); **else "unknown"** — an
  unmarked, undeclared sample is NOT inferred to be real (review P1: §8.1.5 is a provenance report, not
  inference). `provenance_complete` is false whenever any row is unknown, so an incomplete-provenance
  dataset can't be treated as a clean evaluation input.
- `build_versioned_manifest(...)` — runs `compute_split` (quarantined rows excluded, fail-closed
  inherited), enriches survivors, surfaces `quarantined` separately for audit.
- `_manifest_digest` — sha256 over the **entire canonicalized manifest envelope** minus the digest
  field itself (schema/provenance/quarantined/rows/split_digest/…), rows list-sorted →
  **order-independent**. (Not per-row only — the whole envelope is covered.)
- `verify_manifest` — TRUSTED inputs come from the CALLER/this module, never the artifact: it PINS
  `schema_version == SCHEMA_VERSION` and takes the holdout policy from `expected_holdout_fraction`
  (default `0.2`), not the artifact's self-declared value. It does NOT trust the digest self-check
  alone; it independently re-derives from the rows under the trusted policy and binds: split_digest,
  the **full per-row projection** (`_BOUND_ROW_FIELDS`), the quarantine `(locator, reason_code)` set,
  the aggregate provenance verdict, AND the **closed key-set** (top-level / per-row / per-quarantine keys must exactly equal what build emits, so a re-digested manifest cannot smuggle an unbound key such as `unlocks_retraining: True`). RAISES `IntegrityError` on any drift or re-digested tamper.
- CLI: `build` / `report` / `verify` (`build`/`verify` take `--root`; absolute rows require it).

## 2. Verification (local)

| Check | Result |
|---|---|
| slice-2 unit tests (`tests/unit/test_track_e_manifest.py`) | **73 passed** |
| combined slice-1 + slice-2 (no interference) | **105 passed** |
| **containment**: file/cache outside the dataset root, `..`-escaping or absolute locators → fail-closed at build AND rejected at verify even when re-digested | pass |
| **default-root cannot be widened**: absolute input rows with NO explicit root → RED; **absolute AND relative** symlink-escape inside the root rejected via `resolve()`; containment **pre-flights BEFORE `compute_split`** so `content_hash` never opens an out-of-root file (spy asserts zero calls on escaping input) — in BOTH the explicit-`--root` path AND the repo-relative `root=None` CLI-default path (symlink resolved against cwd) | pass |
| **stored locator tamper** (`locator`/`cache_locator` redirected in the stored manifest) → RED: a naive redirect trips the envelope digest; a re-digested redirect trips the `(sample_id, locator)` binding | pass |
| **re-digested per-row tamper** (`category` unknown→real / `split` train→holdout / forged `taxonomy` / redirected `locator` / forged `content_hash`, each with a recomputed digest) → RED via full row binding | pass |
| **re-digested quarantine tamper** (drop/re-label a quarantined `(locator, reason_code)`) → RED via quarantine binding | pass |
| **re-digested provenance flip** (`provenance_complete` False→True + zeroed unknown-count) → RED via aggregate provenance binding | pass |
| **re-digested schema tamper** (`schema_version`→`attacker-schema-v999`) → RED: verify PINS `SCHEMA_VERSION`, never trusts the artifact's self-declared schema | pass |
| **re-digested split-policy tamper** (full rebuild at `holdout_fraction=0.9`) → RED: verify uses the TRUSTED policy (caller/default `0.2`), not the artifact's declared fraction; a legit non-default build verifies only when the caller declares the matching `expected_holdout_fraction` | pass |
| **re-digested key smuggling** (an unbound top-level / row / quarantine key — incl. a security-named `unlocks_retraining: True`) → RED: verify binds the CLOSED key-set to exactly what build emits | pass |
| quarantine records digest as (`locator`=root-relative full path, `reason_code`); OS text stays human-only `detail`; same missing file under two clone roots → **same digest** | pass |
| categorize: markers → augmented/synthetic; unmarked/undeclared → **unknown** (never "real"); declared column authoritative | pass |
| a **naive** single-field tamper trips the envelope self-check; a **re-digested** tamper is caught by independent re-derivation of split_digest / **the full per-row projection** / quarantine set / aggregate provenance verdict from the rows → RED | pass |
| manifest is **fresh-clone PORTABLE**: rows carry root-relative `locator`/`cache_locator` (digested); NO absolute run path enters the manifest; **A-build → B-verify of the SAME artifact = PASS** | pass |
| non-empty `source`/`license`/`label_authority` enforced (blank → fail-closed) | pass |
| `manifest_digest` deterministic / order-independent | pass |
| every §8.1.6 field present; quarantined (unreadable) row excluded | pass |
| `report_by_category` sums back to the row count; per-split correct | pass |
| **tamper discrimination** — `verify_manifest` RED on content change / row-removal / row-add | pass |
| CLI `build`→`report`→`verify` | pass |

**Review fixes:** (P1) unmarked/undeclared rows now classify as **unknown**, never silently "real" —
an explicit provenance column is authoritative, and `provenance_complete` surfaces any unknowns;
(P2) `_enrich_rows` now reuses the **single hash snapshot** `compute_split` computed
(`split["content_hashes"]`) instead of re-reading each file, so the manifest's `content_hash` can
never disagree with the split it records; **`report_by_category`** no longer defaults a missing/
illegal category to "real" — it maps to **"unknown"** and surfaces `illegal_category_rows`.
Rebased onto the decoupled slice-1 (#510 no longer imports the L3 gate); slice-2 imports only
slice-1's split primitives.

**Pre-read containment (build):** every file/cache path is resolve()-contained BEFORE `compute_split`
reads any bytes, so an escaping symlink is rejected without `content_hash` ever opening the out-of-root
file (a spy asserts zero `compute_split`/`content_hash` calls on escaping input). This holds in BOTH
the explicit-`--root` path and the repo-relative `root=None` CLI default (there the symlink is
resolved against the current working directory, the effective dataset root).

**verify_manifest** measures the artifact against TRUSTED config, never the artifact's own claims: it
PINS `schema_version == SCHEMA_VERSION` and takes the holdout policy from the caller
(`expected_holdout_fraction`, default `0.2`) — a full rebuild under a hostile schema or split policy,
even with a recomputed digest, is RED. It also does NOT trust the digest self-check alone (a
re-digesting attacker defeats it), so every load-bearing field is independently re-derived from the
rows under the trusted policy: (1) **envelope
self-consistency** — recompute the digest over the stored envelope and compare (catches a naive
single-field tamper); (2) **split drift** — re-derive the split and compare `split_digest`; (3)
**full row binding** — EVERY bound field of every stored row (`_BOUND_ROW_FIELDS`: sample_id /
locator / cache_locator / taxonomy / family / content_hash / split / category / source / license /
label_authority) must equal the row-re-derived value, so a re-digested per-row tamper (category,
split, label, a redirected locator, a forged hash) is RED — this subsumes and extends the earlier
locator-only binding; (4) **quarantine binding** — the stored `(locator, reason_code)` set must
equal the re-derived one, so a quarantined row cannot be dropped or re-labelled; (5) **aggregate
provenance binding** — `provenance_complete` / `unknown_provenance_rows` must equal the row-re-derived
values. **Residual (documented, not defeated):** the free-text `source`/`license`/`label_authority`
are external inputs, not row-derived — verify re-derives FROM the stored values, so a self-consistent
rewrite of those three (consistently across rows + envelope) cannot be distinguished. Acceptable for
a non-blocking dry-run; a signing layer is the Phase-B answer.

**Honest scope:** this is dry-run tooling. It does NOT compute the §8.1.4 metrics, does NOT bind a
candidate model, and does NOT unlock retraining (the L3 gate is unconditional). Safety-infrastructure
only; model improvement stays off until Phase B (real metrics + two-stage release gate).

## 3. CI & model routing

- CI: curated-list step runs `test_track_e_manifest.py` in `ci.yml` + `ci-tiered-tests.yml`.
- **Model routing (per the goal):** slice-2 is medium difficulty (manifest assembly + categorization,
  reusing slice-1's already-reviewed leakage core), so it was **developed on the Sonnet-5 lane** and
  **reviewed + spot-checked + tidied on the Opus-4.8 lane** (I ran the combined suite, the categorizer
  boundary traps, digest determinism, the row-removal tamper case, and the report-sum invariant, and
  removed a dead re-export before push). The security-critical leakage guard stayed in slice-1.

## 4. Scope boundary (portfolio)

#508 and #509 are merged; this PR waits only on #510. Real Track E **metrics**, the two-stage
release gate (manifest/split digest + candidate model hash + evaluator version + thresholds), and
any re-enablement of retraining are **Phase B** — they require a real data/model environment and
owner threshold decisions, and are deliberately NOT claimed here.
