# Track E slice-2 — versioned manifest + real/synth/augmented reporting — Dev & Verification (2026-07-12)

> **Draft, stacked on Track E slice-1 (#510) → L3 (#509) → #508.** Parallel dev per the owner /goal.
> Does NOT merge/activate before the stack lands. Retarget to `main` as each base merges.

## 0. What this is / is not

**Is (slice-2, torch-free, verified here):**
- **§8.1.6 versioned manifest** — `build_versioned_manifest` emits one enriched record per row that
  survives slice-1's leakage-safe `compute_split`, carrying **every §8.1.6 field**: `file_path`,
  `cache_path`, `taxonomy_v2_class`, `family`, `content_hash`, `split`, `category`, `source`,
  `license`, `label_authority` — plus a deterministic `manifest_digest` and the slice-1 `split_digest`.
- **§8.1.5 real / synthetic / augmented reporting** — `categorize` classifies each row from
  filename/family markers (boundary-anchored regex, no I/O, no RNG); `report_by_category` breaks down
  counts per category, per (category × split), per (category × class).
- **tamper/drift detection** — `verify_manifest` re-derives the manifest from the rows and raises on
  any `manifest_digest`/`split_digest` mismatch (content change, added/removed/moved row). Dry-run
  posture (non-blocking), same as slice-1.

**Is NOT:** model **metrics** (§8.1.4) still need the model run — a follow-up. This reuses slice-1's
already-adversarially-reviewed leakage-safe split rather than re-deriving it, so the manifest can
never drift from the split the L3 gate trusts.

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
| slice-2 unit tests (`tests/unit/test_track_e_manifest.py`) | **27 passed** |
| combined slice-1 + slice-2 (no interference) | **52 passed** |
| categorize boundaries (incl. false-positive traps `rotator`/`generalpurpose`/`gen2_assembly` → real) | pass (spot-checked) |
| `manifest_digest` deterministic / order-independent | pass |
| every §8.1.6 field present; quarantined (unreadable) row excluded | pass |
| `report_by_category` sums back to the row count; per-split correct | pass |
| **tamper discrimination** — `verify_manifest` RED on content change / row-removal / row-add | pass |
| CLI `build`→`report`→`verify` | pass |

**Review fixes (this revision):** (P1) unmarked/undeclared rows now classify as **unknown**, never
silently "real" — an explicit provenance column is authoritative, and `provenance_complete` surfaces
any unknowns; (P2) `_enrich_rows` now reuses the **single hash snapshot** `compute_split` computed
(`split["content_hashes"]`) instead of re-reading each file, so the manifest's `content_hash` can
never disagree with the split it records (closes the two-read snapshot-inconsistency window).

**Documented scope boundary** (intentional, low-risk): `verify_manifest` re-derives
`source`/`license`/`label_authority` from the stored manifest, so it catches content/split/row drift
but not a *self-consistent* relabel of provenance-only fields (they aren't derivable from the rows) —
same class as slice-1's verify authenticating the split, not metrics.

## 3. CI & model routing

- CI: curated-list step runs `test_track_e_manifest.py` in `ci.yml` + `ci-tiered-tests.yml`.
- **Model routing (per the goal):** slice-2 is medium difficulty (manifest assembly + categorization,
  reusing slice-1's already-reviewed leakage core), so it was **developed on the Sonnet-5 lane** and
  **reviewed + spot-checked + tidied on the Opus-4.8 lane** (I ran the combined suite, the categorizer
  boundary traps, digest determinism, the row-removal tamper case, and the report-sum invariant, and
  removed a dead re-export before push). The security-critical leakage guard stayed in slice-1.

## 4. Scope boundary (portfolio)

Only the cadml Track E line is cleanly buildable now. Unchanged/blocked elsewhere: **metasheet2**
(#4168/#4004/#4159, other live sessions), **yuantus** (#1186 / discussion-auth, `adharamans` gh 404),
**cadml #508/#509** (your independent-approval gate). Real Track E **metrics** are the model-run
follow-up.
