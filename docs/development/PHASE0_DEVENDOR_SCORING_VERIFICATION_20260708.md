# Phase 0 · de-vendor scoring — verification (evidence for the pin + gate design)

Companion to `PHASE0_DEVENDOR_SCORING_DESIGN_20260708.md`. Grounded on `origin/main @ 8337ea6e`. This is a **design/verify** slice — no implementation change — so "verification" here means the evidence the design rests on is real and re-derivable, plus the gate's observed-RED plan.

---

## 1. Pin evidence (observed, re-derivable)

```
dedupcad vendored version : "0.2.5"   (vendor/config.py:20, DEDUPCAD2_VERSION)
vendored files            : 14 .py    (scoring, entities_match, v2_normalize, neighbor_index,
                                        dxf_extract, json_diff, config, parsers/*, modules/block_hash)
vendored LOC              : 4,736
vendored sha256           : 135df8462b50c478b62dcb60e7e386a73c2eb109ce8eb50ea52e17ac4abc7032
```
Re-derive: `find src/core/dedupcad_precision/vendor -name '*.py' -exec cat {} + | shasum -a 256`.

## 2. Blast surface (positive control fires)

Probe control: `ml.classifier` → 4 files (live). Consumers of `dedupcad_precision`:
```
src/core/dedupcad_2d_pipeline.py
src/core/dedupcad_2d_worker.py
src/adapters/factory.py
src/api/v1/dedup.py            <-- LIVE route
```
So de-vendoring is a live-behavior change → design/verify-only is the correct posture.

## 3. Baseline is computable (dep check)

`vendor/scoring.py` and `verifier.py` import **only stdlib** (`hashlib`, `dataclasses`, `random`) — no `ezdxf`/`numpy` in the scoring path. So the precision baseline is **not** blocked on heavy deps; the blocker is only mounting the corpora. Available calibration corpora: **35 `data/dedup_report_*/matches.csv`** + `scripts/dedup_2d_threshold_scan_manifest.py`.

## 4. Observed-RED plan for the delta gate (executed in the de-vendor PR, not here)

The gate must be able to fail. When the delta-gate CI job is built:
1. Capture `DEVENDOR_BASELINE_v0.2.5.json` from the pinned vendored copy.
2. **Observed-RED:** perturb one scoring constant (e.g. a similarity weight) → re-run the scan → the gate reports `precision_delta > ε` and **exits non-zero** on the affected corpus; revert → green. This proves the gate detects a scoring shift rather than rubber-stamping.
3. Control: a no-op re-run (same bytes) → `delta = 0` on all 35 corpora → green.

This is specified now so the future PR can't ship a delta gate that only ever passes (the false-green pattern this program has hit before).

## 5. What is explicitly NOT done
- No `vendor/` bytes changed; no import repointed; no canonical dedupcad pin created (cross-repo + owner-ratify).
- `DEVENDOR_BASELINE_v0.2.5.json` not captured here (data-gated CI step; the exact command is in design §3).
- No CI job added (the delta gate belongs to the de-vendor PR once a canonical target is ratified).

## 6. Honesty note
This slice delivers a **safety design + pin**, not a code change — that's the owner's stated scope. The pin hash and version are real and re-derivable; the baseline and gate are specified with runnable commands + an observed-RED requirement, to be executed in the authorized de-vendor PR. Presenting it as "design complete, implementation gated," not "de-vendor done."
