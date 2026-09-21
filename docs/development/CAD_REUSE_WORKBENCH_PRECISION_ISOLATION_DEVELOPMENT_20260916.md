# ReviewReuse precision + tenant isolation — Development Plan

**Date**: 2026-09-16  
**Branch**: `eng/workbench-precision-isolation-20260915`  
**PR**: https://github.com/zensgit/cad-ml-platform/pull/586  
**Base**: `origin/main@22e3c77c` (post-#565)  
**Authority**: `CAD_REUSE_WORKBENCH_90_DAY_PLAN_20260807.md` §8, `PRODUCT_STRATEGY.md` §3.3  
**Design-lock**: `L3_REVIEW_REUSE_WORKBENCH_DESIGNLOCK_20260808.md` (PROPOSED)  
**Verification**: `CAD_REUSE_WORKBENCH_PRECISION_ISOLATION_VERIFICATION_20260916.md`  
**Board**: `CAD_REUSE_WORKBENCH_TASK_BOARD_20260808.md` O10 / O11 / SYS25

---

## 1. Goal

Close the honesty and isolation gaps that remained after the ReviewReuse MVP
(#547–#565) without opening residual_human work:

- local L4 precision only when both sides have geom-json;
- no visual→geometric copy;
- filesystem tenant dirs that cannot collide (`a/b` vs `a_b`);
- live recall asks for geometric / L4 and keeps ML off;
- JWT middleware e2e (no patched `_reviewer_id`);
- isolated-archive `--file` fail-closed;
- Day 61–90 *review-workflow* labels only.

## 2. Holds (do not weaken)

| Hold | Rule |
|---|---|
| Decision default-off | `REVIEW_REUSE_DECISIONS_ENABLED` unset → POST decision 403 |
| R2 HOLD | no `feedback.py` JSONL; audit export stays `audit_quarantine` |
| Track C / R11 / R12 | residual_human — do not claim complete |
| No Track E / cost_cap / PLM write-back | out of this PR |
| DWG | not auto-converted |
| L3 runtime WIP=1 | this PR is the single open runtime slot |

## 3. Model routing (12-hour unattended)

| Difficulty | Model | Work |
|---|---|---|
| D1 mechanical (docs, board, list JSON, tests) | grok-4.5 | markdown, board flips, operator list fields |
| D2 contract (precision labels, cleanup matching, JWT) | grok-4.6 | store/precision/evidence honesty |
| D3 “closed/safe” claims | grok-4.6 + Sol 5.6 | re-review vs `origin/main`; do not self-approve |
| Isolation / tenancy | grok-4.6 | hashed dirs, mixed-legacy refuse |

## 4. In scope (this PR)

| Area | Deliverable |
|---|---|
| Precision | `src/core/review_reuse/precision.py` — JSON/DXF query geom; candidate `geom_json` or geom-store load; lazy DXF extract; L4 `verification.level >= 4`; low score → `different` |
| File gate | `files.py` — `.dxf/.dwg/.pdf` + rasters |
| Adapter | preserve `geom_json`; live `enable_geometric=True`, `enable_ml=False` |
| Evidence | `_top_confidence` trusts geometric only with verified `precision-l4`; vision-only / missing geom / low precision stay low |
| Store | hashed `sha256(tenant_id)[:24]` + `tenant_meta.json`; unique tmp; skip rewrite if valid; `update_atomically` for pipeline/cancel/decision |
| Store ops | list merge; cleanup `--tenant` by id/meta/hash; refuse mixed legacy dirs |
| JWT | IntegrationAuthMiddleware e2e |
| Isolated archive | `--file` without seed, decisions off |
| Metrics | `false_duplicate` / `missed_reuse` / `usefulness:1-5` |

## 5. Out of scope

- Owner ratify (R11) or enabling decisions (R12)
- Training JSONL / Track E eval_integrity_gate
- Automatic DWG→DXF
- Merging #586 (branch protection BLOCKED; needs human review)
- Inventing Track C contacts / measured pilot

## 6. Sequence

1. Precision pass + file gate + hashed store (landed).
2. Live geometric + JWT e2e + `--file` (landed).
3. Sol 5.6 loop: hashed-vs-legacy list, cleanup matching, DXF extract, mixed refuse, EvidencePack verdict (landed through `c01bc02b`).
4. This document + verification MD + operator list `mixed` flag.
5. CI babysit (Python 3.10 job on #586).
6. Unattended waves: fix CI, Sol re-review, no new L3 runtime track.
7. Sol P2 wave (2026-09-16 ~10:00 UTC, grok-4.6): atomic terminal commit,
   rebuild EvidencePack after mid-flight candidate merge, live-recall
   `asyncio.wait_for` on the no-loop `asyncio.run` path.
8. Sol re-review of `4eb1f77c` (session `01a0a9ad-cd3b-7b41-8341-daae02313e41`):
   P1 hash-name cleanup must not select a legacy dir whose recorded
   tenant_id differs; P2 merge pipeline events into terminal snapshots.
9. Sol re-review of `3857d264` (session in wave7 notes): pre-scored L4
   must export `verification.level >= 4`; successful local L4 must drop
   stale `vision_only_unverified` / `missing_geom_json` so confidence is honest.
10. Sol re-review of `de1f44a6` (wave8): refuse cleanup when meta disagrees
    with task tenant_ids; omit decision-before-evidence from review-time
    median; bump `updated_at` when merging pipeline fields into a terminal task.
11. Sol re-review of `f58e80ee` (wave19, 2026-09-17): filesystem
    `update_atomically` / `put_new_idempotent` need an inter-process flock
    (threading.RLock is per-worker); empty DXF extract (`entities=[]`) must
    fail `_is_geom_json` like empty JSON, not proceed to L4.
12. Sol re-review of `ac2ef01d` (wave20): `_is_geom_json` must require a
    supported geometric primitive (not `{"entities":[{}]}`); cancel/decision
    must translate `OccupiedTenantDirError` to `store_conflict` 409.
13. Second 12-hour unattended window (2026-09-17 13:27 UTC → 2026-09-18
    01:30 UTC): 45m waves, grok-4.6 isolation/precision/CI, grok-4.5 docs,
    Sol 5.6 vs `origin/main`. Scheduler `01a0afa12a55`. Never merge.
14. **Window 3 — 24-hour unattended** (2026-09-18 16:57 UTC → **2026-09-19
    17:00 UTC**). Plan below. Never merge.
15. Sol re-review of `23a35ec9` (wave30): `_restore_after_high_l4` must not
    promote an independent `different` (e.g. `version_gate_filtered`) to
    `similar` just because geometric ≥ threshold.
16. Sol re-review of `b98e23a0` (wave31): restore similar only when
    `low_precision_score` was present before clearing; reasonless adapter
    `different` stays `different`.
17. Sol re-review of `d46c6156` (wave32): **no P1/P2**.
18. Wave33 Sol retry vs `1211c01f` hit Codex usage limit (retry after
    2026-09-23 22:47 UTC). Last completed Sol remains wave32.
19. Wave45 on `f2acff4f`: independent `different` rejects (version-gate)
    no longer raise EvidencePack confidence. `tests (3.10)` / `(3.11)` /
    `e2e-smoke` **pass**. Sol still usage-limited.
20. Wave50 on `2c32cae0`: Sol P2 omitted full-ellipse params must
    canonicalize to `0..2π`; non-finite INSERT rotation is not L4.
21. Wave51 on `4858606b`: `tests (3.10)` / `(3.11)` / `e2e-smoke`
    **pass**. Sol 5.6 vs `origin/main`: **no P1/P2**.
22. Wave53 Sol P2 on `b76c67d8`: mixed valid + malformed supported
    entities (`any(...)`) must not admit junk into L4 scoring.
23. Wave54 Sol P2 on `21c0dfbc`: live `vision_response_to_hits` must
    forward match `geom_json` so local L4 can run without a unit score.
24. Wave54 remaining Sol P2s on `d31a9942`: local L4 scores must be
    finite unit-range; unreadable `tenant_meta.json` must occupy the dir.
25. Wave55 Sol P1 on `d31a9942`/`eebaf117`: fused PrecisionVerifier
    score (text/layers/dims) must not be stored as geometric L4.
26. Wave56 Sol P1s on `fa3eb91d`: keep `entities_geom_hash` off when
    customizing Settings; do not stamp live fused `precision_score` as L4.
27. Wave57 Sol P1 on `a3d97c35`: `CAD_ML_PLATFORM_L4_ENTITIES_GEOM_HASH=1`
    must not re-enable bag-of-features on the ReviewReuse L4 path.
28. Wave58 Sol P2 on `b86733a4`: cleanup must refuse unreadable
    `tenant_meta.json` (not treat it as missing).
29. Wave59 Sol P2 on `395c0f5c`: sub-0.001 primitives that collapse
    after verifier 3-decimal rounding must not be L4 geometry.
30. Wave60 Sol P2 on `9806f46b`: HATCH proxies must not dominate
    geometry-only L4 scores.
31. Wave61 Sol P2 on `256068ce`: per-entity `layer_mismatch_penalty`
    must not reject identical geometry on different CAD layers.
32. Wave62 Sol P1 on `48ec1243`: translated `LWPOLYLINE`/`POLYLINE`
    must not score L4 ~1.0 via canonical (pose-invariant) matching.
33. Wave63 Sol P1 on `4362c886`: exploded polyline segments past the
    64-entity matcher cap must still affect L4.
34. Wave64 Sol P1s on `1a99996f`: unmatched extra entities must not
    score L4 1.0; matcher cap must stay a hard bound.
35. Wave65 Sol P1s on `167831f6`: bulged polylines must not explode to
    chords; INSERT `block_hash` mismatch must not pass L4.
36. Wave66 Sol P1s on `66205333`: DXF extract must keep LWPOLYLINE
    bulge; splines longer than 16 control points must not be L4.
37. Wave68 Sol P1/P2 on `0414daca`/`509f59bb`: INSERT needs block_hash;
    stale extract_sig cache must not drop bulges; cleanup refuses a
    whole tenant group if any sibling is mixed.
38. Wave70 Sol P1s on `89a6667c`/`a6a67123`: mixed LINE+INSERT must not
    drop the INSERT and L4-match; incomplete SPLINE (no knots/weights)
    must fail closed.
39. Wave72 Sol P1/P2 on `9dfc3b87`: unknown geom types (POINT/SOLID/…)
    must fail closed; independent rejects suppress confidence even if
    state is similar.
40. Wave74 Sol P1/P2 on `e3357abd`: wrapped ARC sweeps 0→359.9 vs
    0→0.1 must not L4; fractional INSUNITS must not truncate.
41. Wave75 Sol P2 on `f45c6586`: multi-arc sorted sweep bags must not
    L4-match a near-full arc swapped with a sliver at another center.
42. Wave76 Sol P1 on `04e97e8d`: concentric r=10/r=20 sweep swaps must
    not L4-match; pair ARC sweeps by center and radius tolerances.
43. Wave77 Sol P2 on `cc28e09b`: greedy ARC pairing must not zero
    identical nearby ARCs listed in another order; use bipartite match.
44. Wave79 Sol P1/P2 on `dfb0efe9`: nearby-center sweep swaps must not
    L4; polyline ``closed: "false"`` must not explode a closer.
45. Wave80 Sol P1/P2 on `0604807b`: cap entities/vertices before explode;
    Hungarian ARC cost must include radius so concentric reorder is L4.
46. Wave81 Sol P2s on `02fa35dd`: same-center/radius ARC reorder must
    L4; prefork workers must reopen the store flock fd.
47. Wave82 Sol P2s on `bcc9cc76`: mid-flight decision must not hide a
    later pipeline failure; live-recall timeout must stop the worker loop.
48. **Window 4 — 72-hour unattended** (2026-09-21 15:03 UTC → **2026-09-24
    15:00 UTC**). Plan in §6c. Never merge.

## 6b. Window 3 plan (24h, auto-implement)

**Goal:** keep PR #586 honest and CI-green. No new L3 track. No decisions on.

**Landed:** Sol wave29 P2 — high L4 rescore restores `candidate.state` /
`verification.verdict` when the only stale reason was `low_precision_score`.
Independent `different` rejects no longer inflate pack confidence
(`f2acff4f`). Zero-sweep ELLIPSE rejected (`2c32cae0`). Ellipse/INSERT
P2s on `4858606b`. Mixed-entity L4 gate on `21c0dfbc`. Live geom_json
forward on `d31a9942`. Non-finite local L4 + unreadable meta on
`eebaf117`. Geometry-only L4 on `fa3eb91d`. Positional L4 + ignore
live fused scores on `a3d97c35`. Geom-hash pin on `b86733a4`.
Cleanup refuses unreadable tenant meta on `395c0f5c`. Quantized geom
gate on `9806f46b`. HATCH exclusion on `256068ce`. Layer-penalty pin
on `48ec1243`. Polyline explode on `4362c886`. Matcher-cap raise on
`1a99996f`. Unmatched penalty + hard cap on `167831f6`. Bulge/hash
fail-closed on `66205333`. DXF bulge + spline cap on `0414daca`.
Wave67 fail-closed gates on `509f59bb`. INSERT drop + spline degree on
`a6a67123`. LEADER + polyline vertex scan on `9dfc3b87`. Live geom
strip + width/units on `e3357abd`. Wrapped ARC sweep + integral
INSUNITS on `86e7c783` / `f45c6586`. Center-bound ARC pairing on
`8957213a` / `04e97e8d`. Radius-bound pairing on `cc28e09b`.
Bipartite pairing on `dfb0efe9`. Nearest-center + bool closed on
`0604807b`. Pre-explode cap + radius cost on `02fa35dd`. Angle
tie-break + flock PID on `bcc9cc76`. Pipeline-fail + worker cancel on
`37aac287`. Angle-only-on-equal-center + mixed cleanup grouping on
`d1c3a93e`. Equal-spatial angle scale + DXF size cap on `ea4c612f`.
Two-stage row/column equality on `e5cdc765`. Min-total tight edges on
`ee1d21c6`. JSON size cap + live-recall start-stop on `adcd3b6e`.
**Window 3 closed** at 2026-09-19 17:00 UTC. No further auto-implement.

**Then each 45m wave:**
1. If UTC ≥ 2026-09-19 17:00: stop coding; append handoff; delete scheduler.
2. If `tests (3.10)` red: fix (not Evaluation Report).
3. Else Sol 5.6 vs `origin/main`; implement remaining P1/P2 only.
4. Else record 3.10/Sol in verification + handoff (docs only).
5. `flake8 --max-line-length=100` on changed files; `make test-review-reuse`;
   conventional commit; push; never merge.

**Holds:** decisions default-off; R2 no training JSONL; no Track C claim;
no eval_integrity_gate/cost_cap/PLM; L3 WIP=1; DWG not auto-converted;
Evaluation Report = Track E / ignore.

**Models:** grok-4.6 isolation/precision/CI; grok-4.5 docs; Sol 5.6 honesty.

## 6c. Window 4 plan (72h, auto-implement)

**Owner away:** ~72 hours from 2026-09-21 15:03 UTC.
**Deadline:** **2026-09-24 15:00 UTC**. After that: stop coding, append
handoff, delete the Window-4 scheduler, kill the PR monitor.

**Goal:** keep PR #586 honest and CI-green. No new L3 track. No decisions
on. Continue the Sol honesty loop on ReviewReuse precision + tenant
isolation only.

**Start HEAD:** `dc9267e0` (Window-3 docs closeout on product `adcd3b6e`).
CI on that SHA: `tests (3.10)` / `tests (3.11)` / `lint-type` /
`core-fast-gate` **pass**. Evaluation Report fail = Track E / ignore.

**Landed:** Window-4 docs `a6f3e2b5`. Sol P2 stale running idempotency
resume (180s) on `0edbaf22`. Wave2 bind+CAS on `b611a21f`. Wave3:
INSUNITS 1–24 only; `pipeline_claim_id` lease + heartbeat at commit.
Wave4: classic POLYLINE `is_closed`; raw fractional `$INSUNITS`; FILE
runs hash the idempotency key.

### Model routing (by difficulty)

| Difficulty | Model | Work |
|---|---|---|
| D1 mechanical | grok-4.5 | design/verification/handoff MD, board, list JSON, golden tests |
| D2 contract | grok-4.6 | precision/store/evidence/live-recall honesty; CI 3.10 |
| D3 “closed/safe” | grok-4.6 implement + Codex Sol 5.6 review | do not self-approve; Sol vs `origin/main` |
| Isolation / tenancy | grok-4.6 | hashed dirs, mixed-legacy refuse, flock |

Sol command (read-only):

```bash
codex exec --sandbox read-only review -m gpt-5.6-sol --base origin/main
```

If Codex is usage-limited, grok-4.6 may implement from the last completed
Sol P1/P2; do not invent a “closed” claim without Sol.

### Wave rule (every 20m)

1. If UTC ≥ **2026-09-24 15:00**: stop coding; append handoff; delete this
   scheduler; kill the PR monitor. Do not start a new product commit.
2. If the PR monitor died (10h cap): restart
   `python3 ~/.grok/bundled/skills/long-running-background-tasks/watch_pr.py https://github.com/zensgit/cad-ml-platform/pull/586`
3. If `tests (3.10)` red: fix (grok-4.6). Not Evaluation Report.
4. Else Sol 5.6 vs `origin/main` on current HEAD; implement remaining
   **P1/P2 only** (grok-4.6). No Track C, no decisions enable, no merge.
5. Else idle: do **not** docs-spam. One handoff line is enough when Sol
   is clean and 3.10 is green.
6. After a product change: `flake8 --max-line-length=100` on changed
   files; `make test-review-reuse`; conventional commit; push; never merge.
7. After a product wave, grok-4.5 (or same agent if 4.5 unavailable)
   records the SHA + tests in verification MD + a short handoff stanza.
   Design MD §6c “Landed” line only when the gate actually changed.

**Holds (unchanged):** decisions default-off; R2 no training JSONL; no
Track C claim; no eval_integrity_gate/cost_cap/PLM; L3 WIP=1; DWG not
auto-converted; Evaluation Report = Track E / ignore; **never merge**.

**PR monitor:** 10-hour max; scheduler restarts it. Ignore
`kind: mergeable` for merge action (owner must merge). Ignore Evaluation
Report failures. Ignore GitHub EOF/transport. Human review comments:
record in handoff, do not guess owner intent; do not resolve threads.

## 7. Risk

| Risk | Mitigation |
|---|---|
| Fake L4 from visual | never copy visual→geometric; unverified numeric geom does not raise confidence |
| L4 method with level 0 | `_ensure_l4_level` on pre-scored and local L4 paths |
| Stale vision_only after real L4 | `_clear_provisional_unverified` so `_top_confidence` can use the score |
| Cross-tenant delete | mixed legacy dirs refused (`refused_mixed`, exit 1) |
| Temp-file race on task/idem/meta JSON | `_atomic_write_text` unique `mkstemp` in the same dir, then replace |
| Multi-worker cancel/decision/idempotency clobber | store-root flock; reopen fd when PID changes after prefork |
| Empty DXF extract labeled precision-l4 | `_extract_dxf_geom` / `_parse_query_geom` require `_is_geom_json` |
| Lowercase `"line"` scored as unknown | `_canonical_geom` uppercases types before PrecisionVerifier |
| `precision_score: true` becomes L4 1.0 | `optional_unit_score` rejects bool before Pydantic coerce |
| Malformed `{"entities":[{}]}` labeled L4 | `_is_geom_entity` requires LINE/CIRCLE/… fields |
| Negative/NaN/inf primitives labeled L4 | finite coords; CIRCLE/ARC radius > 0; polyline/spline ≥ 2 points |
| Cancel/decision on occupied hashed dir | `_update_atomically` maps `OccupiedTenantDirError` → `store_conflict` |
| DXF parse cost on offline insufficient | extract only when a candidate can be scored |
| Pipeline overwrites cancel/decision | `store.update_atomically` checks terminal status and writes under one lock |
| Decided task vs empty EvidencePack | rebuild pack after merging mid-flight candidates into the decided snapshot |
| Terminal merge drops pipeline events | `_merge_append_only_events` keeps cancel/decision and fills recall/precision/pack |
| Live recall hangs the AnyIO worker | `_run_coro` wait_for + cancel worker-loop tasks on timeout |
| Mid-flight decision hides pipeline_failed | always raise; copy `task.error` onto terminal snapshot |
| ARC angle tie-break outweighs a 0.001 center step | add sweep/start cost only when `dist+dr == 0` |
| Cleanup `--tenant A` deletes hashed A while mixed B holds A | group dirs by every recorded identity before rmtree |
| Cleanup deletes another tenant's legacy dir | hash-name fallback only when meta/tasks have no recorded identity |
| Sidecar tenant A, tasks tenant B | treat as mixed/corrupt; refuse `--apply` |
| Stale `different` after high L4 rescore | `_restore_after_high_l4` sets similar + matching verdict |
| Version-gate L4 1.0 inflates EvidencePack confidence | `_top_confidence` skips independent `different` rejects |
| Zero-sweep ELLIPSE labeled precision-l4 | equal supplied `start_param`/`end_param` rejected like ARC |
| Omitted ellipse params vs DXF 0..2π look different | `_canonical_geom` fills full-span params before L4 |
| NaN/inf INSERT rotation labeled precision-l4 | `_is_geom_entity` requires finite rotation when present |
| Mixed valid LINE + zero-radius CIRCLE fake L4 match | `_is_geom_json` rejects malformed supported types; `_canonical_geom` drops them |
| Local L4 NaN/inf/out-of-range trusted as similar | `_try_l4_score` requires `_is_finite_unit_score` before persist |
| Corrupt tenant_meta.json overwritten by new tenant | `_existing_dir_tenants` treats unreadable sidecar as `_UNREADABLE` |
| High L4 overrides version-gate `different` | restore similar only when no independent rejection remains |
| High L4 promotes reasonless adapter `different` | restore only if `low_precision_score` existed before clear |
| Env `CAD_ML_PLATFORM_L4_ENTITIES_GEOM_HASH=1` recertifies shifted clones | `_try_l4_score` replace() forces `entities_geom_hash=False` |
| Sub-0.001 LINEs/radii collapse to identical zero-length L4 | `_is_geom_entity` + post-normalize `_is_geom_json` use 3-decimal quant |
| Layer names reject identical geometry via entity penalty | strip `layer`; `layer_mismatch_penalty=0` in L4 Settings |
| Translated polylines score L4 1.0 after canonical pose | `_explode_polyline` to absolute LINEs before PrecisionVerifier |
| Subset geometry scores L4 1.0 (min-length truncate) | `_penalize_unmatched` scales by min/max entity counts |
| Unbounded max_match_entities exhausts workers | cap entities/vertices before explode; over-cap is not L4 |
| DXF extract drops bulge so curve matches chord | `dxf_extract` keeps `bulges`; `_polyline_has_bulge` refuses L4 |
| Long SPLINE tails ignored (first 16 controls) | `_is_geom_entity` rejects splines with >16 control points |
| INSERT name+pose without hash certifies unknown block | `_is_geom_entity` requires non-empty `block_hash` |
| Stale extract_sig cache omits polyline bulges | `_EXTRACT_CACHE_VERSION=2`; unversioned hits ignored |
| Cleanup deletes hashed sibling while mixed dir is refused | refuse every dir in the grouped tenant |
| LINE+INSERT drops INSERT and L4-matches the LINE | `_UNSCORED_GEOM_TYPES` fails the whole payload |
| SPLINE knots/weights omitted, vendor matches 16 points | SPLINE is unscored geom; not precision-l4 |
| POINT/SOLID/MESH dropped so a shared LINE L4-matches | unknown geom types fail `_is_geom_json` |
| similar+version_gate still raises EvidencePack confidence | `_top_confidence` skips independent reasons regardless of state |
| ARC 0→359.9 vs 0→0.1 scores L4 ~0.9 via modulo ends | `_arc_records` uses CCW sweep, not endpoint pairing |
| Multi-arc sorted sweep bags swap near-full vs sliver | `_arc_sweeps_conflict` pairs by center/radius tols |
| Greedy ARC pairing zeros identical reorder | Hungarian nearest-center then sweep check |
| Nearby ARC sweep swap scores L4 ~0.9 | `_arc_sweeps_conflict` assigns by center+radius, not sweep |
| Concentric near-radii reorder zeros L4 | Hungarian cost is center dist + radius delta |
| Same-center/radius ARC reorder zeros L4 | tiny sweep/start tie-break in assignment cost |
| ``closed: "false"`` explodes a closer | `_polyline_closed` requires a real bool |
| Fractional INSUNITS 4.9 truncates to 4 | `_drawing_units` requires an integral value |
| 24h unattended overreach | Window 3 stopped 2026-09-19 17:00 UTC |
| 72h unattended overreach | Window 4 scheduler stops **2026-09-24 15:00 UTC**; never merge; never enable decisions |
