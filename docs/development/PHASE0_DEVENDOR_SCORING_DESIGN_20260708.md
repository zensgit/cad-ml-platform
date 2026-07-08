# Phase 0 · de-vendor scoring — DESIGN/VERIFY only (no implementation change)

- **Status**: DESIGN. Branch only, **no PR** (backlog cap). **No code changed** — this is the pin + gate spec the owner asked for before any implementation.
- **Grounded on** `origin/main @ 8337ea6e`, evidence observed live.
- **Scope discipline (owner's instruction):** *"de-vendor scoring 只做设计/验证,不急着改实现:先 pin 当前 vendored copy、重跑 precision baseline、定义 precision-delta gate."* This document does exactly that and nothing more.

---

## 1. The trap: de-vendoring is not free debt removal

`src/core/dedupcad_precision/` **vendors** dedupcad's geometric/semantic JSON similarity for L4 precision checks (`__init__.py`: *"vendors the core … from the `dedupcad` repository"*). It's tempting to delete the copy and import canonical dedupcad. But **the L4 similarity thresholds currently in production were calibrated against these exact vendored bytes.** Repointing to a canonical dedupcad whose scoring differs even slightly **silently shifts precision/grouping with no failing test** — the classic "de-vendor regresses accuracy invisibly" failure.

**Blast surface (positive control fires — `ml.classifier` = 4 files):** `dedupcad_precision` is consumed by `src/core/dedupcad_2d_pipeline.py`, `dedupcad_2d_worker.py`, `src/adapters/factory.py`, and **`src/api/v1/dedup.py` — a live route.** So a scoring shift is a live-behavior shift.

## 2. Pin the current vendored copy (the baseline artifact)

Observed, exact:
- **Version:** `DEDUPCAD2_VERSION = "0.2.5"` (`vendor/config.py:20`) — the dedupcad release this copy corresponds to.
- **Content:** 14 `.py` files, **4,736 LOC**, `vendor/` = `scoring.py`, `entities_match.py`, `v2_normalize.py`, `neighbor_index.py`, `dxf_extract.py`, `json_diff.py`, `config.py`, `parsers/{hatch,text,dimension}.py`, `modules/block_hash.py`, …
- **Exact-bytes hash (the pin):** `sha256 = 135df8462b50c478b62dcb60e7e386a73c2eb109ce8eb50ea52e17ac4abc7032` (concatenated `vendor/**/*.py`).

Any de-vendor must repoint to a canonical dedupcad that reproduces the scoring of `0.2.5`, or pass the delta gate in §4.

## 3. The precision baseline is computable (good news)

The core scoring path (`vendor/scoring.py`, `verifier.py`) imports **only stdlib** (`hashlib`, `dataclasses`, `random`) — **no ezdxf/numpy in the scoring itself**. So the baseline is not blocked on heavy deps. The weak-label calibration data is present: **35 `data/dedup_report_*/matches.csv` corpora** + `scripts/dedup_2d_threshold_scan_manifest.py` (threshold scan over `expected_groups.json + matches.csv`).

**Baseline capture (run against the pinned vendored copy, record the numbers):**
```
python scripts/dedup_2d_threshold_scan_manifest.py \
   --data data/dedup_report_train_local \
   --out reports/DEVENDOR_BASELINE_v0.2.5.json
```
Store precision + grouping metrics per corpus as `DEVENDOR_BASELINE_v0.2.5.json`, keyed to the `sha256` above.

## 4. The precision-delta gate (what any future de-vendor must pass)

1. **Pin dedupcad to an exact tag/commit** that claims to match `0.2.5`; extract a shared pip package rather than a second in-repo copy.
2. On adoption, **re-run the threshold scan** over the same 35 corpora with the canonical scoring.
3. **Gate:** per-corpus `precision_delta` and `grouping_delta` vs `DEVENDOR_BASELINE_v0.2.5.json` must be `≤ ε` (propose `ε = 0.002` absolute precision; tune on the scan spread). Any corpus exceeding ε **fails the PR** — a blocking CI check.
4. **Keep the vendored copy as fallback** until the canonical import passes both the delta gate **and** the existing golden dedup gate. Only then delete `vendor/`.
5. Prefer **`allowed_fallback`**: a config flag selecting vendored vs canonical, so rollback is a flag flip, not a revert.

## 5. Why design-only now (not implementation)
- It touches a **live dedup route** — an accuracy regression is a production regression.
- The canonical dedupcad pin + shared-package extraction is **cross-repo + owner-ratify** (per #499 §2.3, §5) — not an unattended change.
- The right first step is exactly this: pin + baseline + gate spec, so that when the cross-repo work is authorized, the safety net already exists.

## 6. Follow-ups
- **Owner-ratify:** the de-vendor decision itself (which surface becomes canonical; cross-repo coordination with `dedupcad`).
- Capture `DEVENDOR_BASELINE_v0.2.5.json` (a data-gated CI step — the scan needs the corpora mounted; stdlib scoring means no dep blocker).
- Build the delta-gate CI job once a canonical target exists.
- **Model:** grounded/authored in-session on Opus 4.8 (Fable 5 at daily cap). Baseline capture + gate implementation are Sonnet-5-class mechanical work.
