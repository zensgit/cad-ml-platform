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
- `_manifest_digest` — sha256 over per-row `sort_keys` JSON, list-sorted → **order-independent**.
- `verify_manifest` — re-derive + compare both digests; RAISES `IntegrityError` on drift.
- CLI: `build` / `report` / `verify`.

## 2. Verification (local)

| Check | Result |
|---|---|
| slice-2 unit tests (`tests/unit/test_track_e_manifest.py`) | **57 passed** |
| combined slice-1 + slice-2 (no interference) | **89 passed** |
| **containment**: file/cache outside the dataset root, `..`-escaping or absolute locators → fail-closed at build AND rejected at verify even when re-digested | pass |
| **default-root cannot be widened**: absolute input rows with NO explicit root → RED (the old common-parent heuristic would silently widen the root and admit a sibling-dir file); symlink-escape inside the root also rejected via `resolve()` | pass |
| **stored locator tamper** (`locator`/`cache_locator` redirected in the stored manifest) → RED: a naive redirect trips the envelope digest; a re-digested redirect trips the `(sample_id, locator)` binding | pass |
| quarantine records digest as (`locator`=root-relative full path, `reason_code`); OS text stays human-only `detail`; same missing file under two clone roots → **same digest** | pass |
| categorize: markers → augmented/synthetic; unmarked/undeclared → **unknown** (never "real"); declared column authoritative | pass |
| manifest_digest covers the **full envelope** (schema/provenance/quarantined/rows/…); tamper → RED | pass |
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

**verify_manifest** now runs two checks: (1) **envelope self-consistency** — recompute the digest
over the stored envelope (minus the digest) and compare, catching a tamper to ANY field; (2) **split
drift** — re-derive the split from the actual files and compare `split_digest`. Residual boundary: a
*self-consistent* rewrite of provenance-only fields (`source`/`license`/`label_authority`, which
verify re-derives from the stored manifest) is not distinguishable — acceptable for a non-blocking
dry-run.

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
