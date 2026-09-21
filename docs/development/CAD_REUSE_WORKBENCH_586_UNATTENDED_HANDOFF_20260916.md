# #586 12-hour unattended handoff

**Deadline (window 2)**: 2026-09-18 01:30 UTC. After that, stop coding, append this file, delete scheduler `01a0afa12a55`.
**Previous deadline**: 2026-09-16 17:00 UTC (closed).

## Goal

Keep PR https://github.com/zensgit/cad-ml-platform/pull/586 honest and CI-green.
Do not merge. Do not enable decisions. Do not claim Track C.

## Holds

- `REVIEW_REUSE_DECISIONS_ENABLED` stays default-off
- R2 HOLD: no `feedback.py` JSONL
- No eval_integrity_gate / cost_cap / PLM write-back
- L3 runtime WIP=1 (this PR only)
- DWG is not auto-converted

## Model routing

- grok-4.6: CI failures, store/precision/isolation, Sol-class bugs
- grok-4.5: docs, board, list JSON, mechanical tests
- Codex gpt-5.6-sol: review vs origin/main after a code wave (`codex exec --sandbox read-only review -m gpt-5.6-sol --base origin/main`)

## Open now

1. **P0** CI `tests (3.10)` and `tests (3.11)` were **green** on `dc2c8fec`.
   Local enumerator RED from gitignored files is still not CI.
2. Wave 2026-09-16 ~16:45 UTC: `tests (3.10)` / `tests (3.11)` **pass**
   on `6b29ab81`. Sol 5.6 P2: empty inline geom_json must not skip the
   hash-keyed geom store. Local `make test-review-reuse` **136 passed**.
3. After push: wait for 3.10; if red, fix; else Sol P1/P2 only.
4. Evaluation Report (Track E) stays out of scope.

## Do not

- Merge #586
- Open a second L3 runtime PR
- Invent Track C evidence
- Enable decisions in production config

## Deadline reached — 2026-09-16 17:09 UTC

Unattended loop stopped. Scheduler deleted. PR monitor not restarted.

| Item | State |
|---|---|
| Branch | `eng/workbench-precision-isolation-20260915` |
| HEAD | `704fd1c6` — empty inline `geom_json` falls back to hash-keyed geom store |
| Last fully green 3.10/3.11 | `6b29ab81` |
| `lint-type` on `704fd1c6` | pass |
| `tests (3.10)` / `tests (3.11)` on `704fd1c6` | pending at stop (not red) |
| Local `make test-review-reuse` | **136 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | **not done** (`blocked`; needs human review) |
| Decisions | still default-off |

Owner next: wait for 3.10 on `704fd1c6`; human review of #586; do not enable `REVIEW_REUSE_DECISIONS_ENABLED`.

## Continuation — 2026-09-17 01:08 UTC (owner `continue`)

Scheduler stays deleted. PR monitor restarted. No new product code.

| Item | State |
|---|---|
| HEAD | `7002bfa2` |
| `tests (3.10)` / `tests (3.11)` / `lint-type` | **pass** |
| Local `make test-review-reuse` | **136 passed** |
| Sol 5.6 vs `origin/main` | **no P1/P2** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Continuation — 2026-09-17 (owner `continue`, wave19)

Sol 5.6 on `f58e80ee` vs `origin/main`: remaining P1 multi-worker FS
lock and P2 empty DXF extract. Implemented; never merge; decisions stay off.

| Item | State |
|---|---|
| P1 | store-root flock + unique tmp on `FilesystemReviewReuseStore` |
| P2 | `_is_geom_json` on DXF extract / query geom / `_try_l4_score` |
| Local `make test-review-reuse` | **140 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Continuation — 2026-09-17 wave20

Sol 5.6 on `ac2ef01d`: P2 malformed entity L4; P2 cancel/decision occupied-dir
must be `store_conflict`. Implemented locally. Never merge.

| Item | State |
|---|---|
| P2 entity shape | `_is_geom_entity` required fields for LINE/CIRCLE/ARC/poly/ellipse/spline |
| P2 occupied updates | `_update_atomically` translates `OccupiedTenantDirError` |
| Local `make test-review-reuse` | **142 passed** |
| Merge | still not done |

## Continuation — 2026-09-17 wave21

Sol 5.6 on `f049a14c`: P2 negative-radius CIRCLE (and degenerate primitives)
still received L4. Finite positive radius / nondegenerate fields required.

| Item | State |
|---|---|
| Local `make test-review-reuse` | **143 passed** |
| Merge | still not done |

## Continuation — 2026-09-17 wave22

`tests (3.10)` / `tests (3.11)` **pass** on `52c820b5`. Sol P2: zero-length
polyline still L4. Require two distinct points. Never merge.

| Item | State |
|---|---|
| Local `make test-review-reuse` | **144 passed** |
| Merge | still not done |

## Continuation — 2026-09-17 wave23

`tests (3.10)` / `tests (3.11)` **pass** on `4a9a390a`. Sol P2: INSERT geom,
ellipse ratio, pre-scored L4 range. Never merge.

| Item | State |
|---|---|
| Local `make test-review-reuse` | **148 passed** |
| Merge | still not done |

## Continuation — 2026-09-17 wave24

`tests (3.10)` / `tests (3.11)` **pass** on `d2d0beb5`. Sol P2: clear 2.0
geometric scores, reject zero-scale INSERT and zero-sweep ARC. Never merge.

| Item | State |
|---|---|
| Local `make test-review-reuse` | **150 passed** |
| Merge | still not done |

## Continuation — 2026-09-17 wave25

PR monitor restarted after 10h. `tests (3.10)` / `tests (3.11)` **pass** on
`a999cc63`. Sol P2: boolean JSON numbers still L4. Never merge.

| Item | State |
|---|---|
| Local `make test-review-reuse` | **151 passed** |
| Merge | still not done |

## Window 2 start — 2026-09-17 13:27 UTC

Owner away ~12h. Scheduler `01a0afa12a55` every 45m until **2026-09-18 01:30 UTC**.
PR monitor `01a0af0a` already running. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD at start | `1ae5be3b` (docs; parent `c9a7ce4b` Sol wave26 clean) |
| tests (3.10) / tests (3.11) / e2e-smoke on `1ae5be3b` | **pass** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Continuation — 2026-09-17 wave26

`tests (3.10)` / `tests (3.11)` / `lint-type` **pass** on `c9a7ce4b`. Sol 5.6
vs `origin/main`: **no P1/P2**. Evaluation Report still Track E. Never merge.

| Item | State |
|---|---|
| HEAD | `c9a7ce4b` |
| Local `make test-review-reuse` | **151 passed** |
| Sol wave26 | no P1/P2 |
| Merge | still not done |

## Window 2 wave27 — 2026-09-17 ~13:50 UTC

Sol P2: lowercase `"line"` scored as unknown; `precision_score: true`
became L4 1.0. Canonicalize types before scoring; sanitize adapter scores.

| Item | State |
|---|---|
| Local `make test-review-reuse` | **153 passed** |
| Merge | still not done |

## Window 2 wave28 — 2026-09-17 ~15:47 UTC

Scheduled wave died on grok proxy. Picked up here. `tests (3.10)`/`(3.11)`
pass on `d6d9a1c9`. Sol P2: stale `low_precision_score` after high L4 rescore.

| Item | State |
|---|---|
| Local `make test-review-reuse` | **154 passed** |
| Merge | still not done |

## Deadline reached — 2026-09-18 04:10 UTC

Window-2 unattended loop stopped (`now` ≥ **2026-09-18 01:30 UTC**).
No product coding this fire. Scheduler `01a0afa12a55` deleted. PR monitor
for #586 was not running; not restarted. Never merge.

| Item | State |
|---|---|
| Branch | `eng/workbench-precision-isolation-20260915` |
| HEAD | `aed74d04` — drop stale `low_precision_score` after a high L4 rescore |
| Last fully green 3.10/3.11 | `aed74d04` (`tests (3.10)` / `tests (3.11)` / `lint-type` / `e2e-smoke` **pass**) |
| Local `make test-review-reuse` | **154 passed** (recorded after `aed74d04`; not re-run at stop) |
| Sol 5.6 wave29 vs `origin/main` on `aed74d04` | remaining **P2**: restore `state` / `verification.verdict` when clearing a stale low-score rejection (`precision.py` `_apply_l4_score`) |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | **not done** (`unstable`; needs human review) |
| Decisions | still default-off |

Owner next: human review of #586; optional remaining Sol P2 (restore
verdict on high L4 rescore); do not enable `REVIEW_REUSE_DECISIONS_ENABLED`.

## Window 3 start — 2026-09-18 16:57 UTC (24h)

Owner away ~24h. Deadline **2026-09-19 17:00 UTC**. Plan in development MD §6b.
Scheduler `01a0b5758345` every 45m. First item: Sol wave29 P2 restore
state/verdict on high L4 rescore. Never merge; decisions stay off.

| Item | State |
|---|---|
| Local `make test-review-reuse` | **154 passed** |
| Merge | still not done |

## Window 3 wave30 — 2026-09-18 ~17:50 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `23a35ec9`.
Sol 5.6 vs `origin/main`: P2 high L4 must not promote an independent
`different` (`version_gate_filtered`) to `similar`. Restore similar only
when no independent rejection remains. Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 | `_restore_after_high_l4` keeps independent `different` |
| Local `make test-review-reuse` | **156 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave31 — 2026-09-18 ~18:40 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `b98e23a0`.
Sol 5.6 vs `origin/main`: P2 restore similar only when
`low_precision_score` existed before it was cleared. Reasonless adapter
`different` stays `different`. Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 | `_restore_after_high_l4(..., had_low_precision=)` |
| Local `make test-review-reuse` | **158 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave32 — 2026-09-18 ~19:20 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `d46c6156`.
Sol 5.6 vs `origin/main`: **no P1/P2**. Docs-only record. Never merge;
decisions stay off.

| Item | State |
|---|---|
| HEAD | `d46c6156` |
| Last fully green 3.10/3.11 | `d46c6156` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave32 | no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave33 — 2026-09-18 ~20:00 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `1211c01f`.
Sol 5.6 vs `origin/main` **blocked** (Codex usage limit; retry after
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `1211c01f` |
| Last fully green 3.10/3.11 | `1211c01f` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave33 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave34 — 2026-09-18 ~20:45 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `58757c69`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `58757c69` |
| Last fully green 3.10/3.11 | `58757c69` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave34 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave35 — 2026-09-18 ~21:30 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `800bb4fd`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `800bb4fd` |
| Last fully green 3.10/3.11 | `800bb4fd` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave35 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave36 — 2026-09-18 ~22:15 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `c231f79c`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `c231f79c` |
| Last fully green 3.10/3.11 | `c231f79c` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave36 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave37 — 2026-09-18 ~23:00 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `a04f3b91`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `a04f3b91` |
| Last fully green 3.10/3.11 | `a04f3b91` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave37 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave38 — 2026-09-18 ~23:45 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `9a686f13`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `9a686f13` |
| Last fully green 3.10/3.11 | `9a686f13` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave38 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave39 — 2026-09-19 ~00:30 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `cbd987f3`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `cbd987f3` |
| Last fully green 3.10/3.11 | `cbd987f3` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave39 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave40 — 2026-09-19 ~01:15 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `aae99b6e`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `aae99b6e` |
| Last fully green 3.10/3.11 | `aae99b6e` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave40 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave41 — 2026-09-19 ~02:00 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `077c188a`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `077c188a` |
| Last fully green 3.10/3.11 | `077c188a` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave41 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave42 — 2026-09-19 ~02:45 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `472042c7`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `472042c7` |
| Last fully green 3.10/3.11 | `472042c7` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave42 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave43 — 2026-09-19 ~03:30 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `715401d0`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `715401d0` |
| Last fully green 3.10/3.11 | `715401d0` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave43 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave44 — 2026-09-19 ~04:15 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `e5da5144`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. No product change. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `e5da5144` |
| Last fully green 3.10/3.11 | `e5da5144` |
| Local `make test-review-reuse` | **158 passed** (not re-run; no product change) |
| Sol wave44 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 accelerate — 2026-09-19 ~04:32 UTC

Owner asked to go faster. Scheduler interval 45m → 20m; skip docs-only
when Sol is blocked. Product: version-gate `different` must not raise
EvidencePack confidence to 1.0.

| Item | State |
|---|---|
| Local `make test-review-reuse` | **158 passed** |
| Merge | still not done |

## Window 3 wave49 — 2026-09-19 ~06:19 UTC

Owner `continue`. `tests (3.10)` green on `8fe2c393`. Sol P2: zero-sweep
ELLIPSE (`start_param == end_param`) still L4. Reject like ARC.

| Item | State |
|---|---|
| Local `make test-review-reuse` | **159 passed** |
| Merge | still not done |

## Deadline reached — 2026-09-19 17:01 UTC

Window-3 unattended loop stopped (`now` ≥ **2026-09-19 17:00 UTC**).
No product coding this fire. Scheduler `01a0b5758345` deleted. PR
monitor for #586 was still running; not killed. Never merge; decisions
stay default-off.

| Item | State |
|---|---|
| Branch | `eng/workbench-precision-isolation-20260915` |
| HEAD | `adcd3b6e` — cap JSON geom decode; stop live-recall on start timeout |
| Last fully green 3.10/3.11 | `ea4c612f` |
| `lint-type` / `e2e-smoke` on `adcd3b6e` | **pass** |
| `tests (3.10)` / `tests (3.11)` on `adcd3b6e` | in_progress at stop (not red) |
| Local `make test-review-reuse` | **229 passed** (recorded after `adcd3b6e`) |
| Last Sol 5.6 | wave89 P2s implemented on `adcd3b6e`; no further Sol this fire |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | **not done** (`blocked`; needs human review) |
| Decisions | still default-off |

Owner next: wait for 3.10 on `adcd3b6e`; human review of #586; do not
enable `REVIEW_REUSE_DECISIONS_ENABLED`.

## Window 4 start — 2026-09-21 15:03 UTC (72h)

Owner away ~72h. Deadline **2026-09-24 15:00 UTC**. Plan in development
MD §6c. Verification MD is the evidence log.

**Never merge. Never enable `REVIEW_REUSE_DECISIONS_ENABLED`.**

| Item | State |
|---|---|
| Branch | `eng/workbench-precision-isolation-20260915` |
| Start HEAD | `dc9267e0` (docs closeout; product `adcd3b6e`) |
| `tests (3.10)` / `tests (3.11)` / `lint-type` / `core-fast-gate` | **pass** on `dc9267e0` |
| Local `make test-review-reuse` | **229 passed** (last Window-3 record) |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |
| Decisions | still default-off |

**Models:** grok-4.6 isolation/precision/CI; grok-4.5 docs; Codex
`gpt-5.6-sol` vs `origin/main`.

**Wave rule:** if UTC ≥ 2026-09-24 15:00 stop + delete scheduler + kill
PR monitor; else if `tests (3.10)` red fix it; else Sol P1/P2 only;
else idle (no docs-spam). Restart the 10h PR monitor if it died.

## Window 4 wave2 — 2026-09-21 ~15:50 UTC

Sol 5.6 vs `origin/main` on `0edbaf22`: **P2** stale resume scored a
retry's bytes under the old task identity; **P2** two workers could
both rerun and overwrite `evidence_ready`. Bind retry name/hash;
CAS-claim before `_run_pipeline`; first terminal result wins.
Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 resume bind | `_require_idempotent_input` 409 `idempotency_conflict` |
| P2 stale claim | `_claim_stale_running` CAS on `updated_at` |
| Local `make test-review-reuse` | **233 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 4 wave1 — 2026-09-21 ~15:22 UTC

Sol 5.6 vs `origin/main` on `a6f3e2b5`: **P2** filesystem idempotency
replay returned a crashed `running` snapshot forever. Resume when
`updated_at` is older than 180s; leave in-flight running tasks alone.
Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 stale running | `_return_idempotent` resumes after `STALE_RUNNING_SECONDS` |
| Local `make test-review-reuse` | **231 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave89 — 2026-09-19 ~16:50 UTC

Sol 5.6 vs `origin/main` on `ee1d21c6`: **P2** JSON query geom decoded
the full payload before the DXF size cap; **P2** live-recall nested
loop raised on worker start timeout before `finally`, leaking
`run_forever`. Cap JSON bytes first; always stop/join the worker.
Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 JSON size cap | `_parse_query_geom` uses `_dxf_extract_limit_bytes` |
| P2 live-recall start | `_run_coro` stops loop on startup timeout |
| Local `make test-review-reuse` | **229 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave88 — 2026-09-19 ~16:22 UTC

Sol 5.6 vs `origin/main` on `e5cdc765`: **P2** row/column spatial
equality dropped the other min-total assignment (0.1+0.1 vs 0.0+0.2),
so matching sweeps on the alternate were a fake conflict. Rematch by
angle on tight dual edges (reduced cost ≤ 1e-9). Never merge;
decisions stay off.

| Item | State |
|---|---|
| P2 min-total ARC | `_arc_hungarian` duals; angle on tight spatial edges |
| Local `make test-review-reuse` | **227 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave87 — 2026-09-19 ~16:05 UTC

`tests (3.10)` / `tests (3.11)` **pass** on `ea4c612f`. Sol 5.6 vs
`origin/main`: **P2** scaled spatial+angle cost lets a ~5e-7
3-decimal assignment gap lose to a 359° sweep term, so swapped
nearby ARCs score L4 ~0.61. Two-stage: spatial Hungarian, then
angle only on equal-spatial edges. Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 two-stage ARC | `_arc_sweeps_conflict` rematch by angle only on equal spatial |
| Local `make test-review-reuse` | **226 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave85 — 2026-09-19 ~15:25 UTC

`lint-type` pass / `e2e-smoke` pass on `d1c3a93e`; `tests (3.10)` was
still in progress at Sol start. Sol 5.6 vs `origin/main`: **P2**
uniform-offset reordered ARCs lost angle tie-break when `base != 0`;
**P2** local DXF extract wrote/parsed bytes over `DEDUPCAD2_MAX_FILE_MB`.
Lexicographic spatial-then-angle cost; skip extract before write.
Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 ARC equal-spatial | `_arc_sweeps_conflict` scales spatial then angle |
| P2 DXF size cap | `_extract_dxf_geom` checks `_dxf_extract_limit_bytes` |
| Local `make test-review-reuse` | **225 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave84 — 2026-09-19 ~15:10 UTC

`tests (3.10)` / `tests (3.11)` **pass** on `37aac287`. Sol 5.6 vs
`origin/main`: **P1** ARC angle tie-break can outweigh a 0.001
center/radius step and pair swapped nearby sweeps (L4 ~0.995);
**P1** `cleanup --tenant A --apply` grouped mixed dirs only under
`_tenant_label`, so A's hashed dir was deleted while mixed B (holding
A tasks) was refused. Angle cost only when `dist+dr == 0`; group
cleanup by every recorded identity and refuse the linked set. Never
merge; decisions stay off.

| Item | State |
|---|---|
| P1 ARC tie-break | `_arc_sweeps_conflict` adds angle only on equal center+radius |
| P1 mixed cleanup | `_dir_tenant_identities` groups mixed siblings before rmtree |
| Local `make test-review-reuse` | **223 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave82 — 2026-09-19 ~14:26 UTC

Sol 5.6 vs `origin/main` on `bcc9cc76`: **P2** mid-flight decision made
`POST /tasks` return 200 with `status=decided` and no error after a
pipeline boom; **P2** nested-loop live-recall timeout leaked the worker
thread. Always raise `pipeline_failed` and persist `error`; cancel
tasks on the worker loop then stop it. Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 pipeline fail | `_commit_pipeline_result` copies `error`; always raise |
| P2 live recall | `_run_coro` uses a dedicated loop and cancels on timeout |
| Local `make test-review-reuse` | **221 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave81 — 2026-09-19 ~14:08 UTC

Sol 5.6 vs `origin/main` on `02fa35dd`: **P2** same-center/radius
`ARC(0→90)` + `ARC(180→0)` reorder zeros L4; **P2** prefork workers
inherit the parent's flock fd so `flock` is not exclusive. Angle
tie-break; reopen lock fd when PID changes. Never merge; decisions
stay off.

| Item | State |
|---|---|
| P2 ARC tie-break | Hungarian cost adds 1e-6 sweep/start delta |
| P2 flock fork | `_StoreFileLock._ensure_fd` reopens when PID changes |
| Local `make test-review-reuse` | **220 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave80 — 2026-09-19 ~13:44 UTC

Sol 5.6 vs `origin/main` on `0604807b`: **P1** the 128-entity cap ran
after polyline explode, so a huge vertex list could exhaust workers;
**P2** concentric r=10/r=10.3 reorder could Hungarian-pair by list
order and zero a valid L4. Cap before explode; add radius to assignment
cost. Never merge; decisions stay off.

| Item | State |
|---|---|
| P1 cap | `_is_geom_json` / polyline vertex cap before explode |
| P2 ARC cost | Hungarian `dist + abs(radius)` |
| Local `make test-review-reuse` | **218 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave79 — 2026-09-19 ~13:22 UTC

Sol 5.6 vs `origin/main` on `dfb0efe9`: **P1** nearby r=10 arcs at
x=0/x=0.3 can swap 359.9°/0.1° sweeps and still score ~0.9 L4;
**P2** ``closed: "false"`` explodes a closer. Pair by nearest center
then compare sweeps; require boolean `closed`. Never merge; decisions
stay off.

| Item | State |
|---|---|
| P1 ARC assign | `_arc_sweeps_conflict` Hungarian on center, then sweep |
| P2 polyline closed | `_polyline_closed` rejects non-bool |
| Local `make test-review-reuse` | **217 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave77 — 2026-09-19 ~12:42 UTC

Sol 5.6 vs `origin/main` on `cc28e09b`: **P2** greedy ARC pairing
consumes the only valid neighbor and zeros identical nearby ARCs
listed in another order. Bipartite matching. Never merge; decisions
stay off.

| Item | State |
|---|---|
| P2 ARC order | `_arc_sweeps_conflict` Kuhn matching on center/radius/sweep |
| Local `make test-review-reuse` | **215 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave76 — 2026-09-19 ~12:22 UTC

Sol 5.6 vs `origin/main` on `04e97e8d`: **P1** concentric r=10/r=20
arcs can swap 359.9° and 0.1° sweeps and still score ~0.9 L4 because
pairing ignored radius. Pair by center and radius tolerances. Never
merge; decisions stay off.

| Item | State |
|---|---|
| P1 ARC radius | `_arc_records` / `_arc_sweeps_conflict` use `tol_circle_radius` |
| Local `make test-review-reuse` | **214 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave75 — 2026-09-19 ~12:02 UTC

Sol 5.6 vs `origin/main` on `f45c6586`: **P2** sorted ARC sweep bags
let a 359.9° arc and a 0.1° arc swap centers and still score ~0.9 L4.
Pair sweeps by quantized center. Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 ARC identity | `_arc_records` / `_arc_sweeps_conflict` pair by center |
| Local `make test-review-reuse` | **212 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave74 — 2026-09-19 ~11:43 UTC

Sol 5.6 vs `origin/main` on `e3357abd`: **P1** vendor ARC matching
pairs endpoints modulo 360, so 0→359.9 vs 0→0.1 scored ~0.9 L4;
**P2** `int(4.9)` treated fractional INSUNITS as millimeters. Compare
CCW sweep; require integral units. Never merge; decisions stay off.

| Item | State |
|---|---|
| P1 ARC sweep | `_arc_signatures` / `_ccw_sweep_deg` |
| P2 INSUNITS | `_drawing_units` rejects non-integers |
| Local `make test-review-reuse` | **210 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave73 — 2026-09-19 ~11:40 UTC

Sol vs `origin/main` on `01ec8379`: **P1** live `geom_json` was
forwarded into tenant GET/EvidencePack (cross-tenant geometry if
search_2d is unscoped); **P2** nonzero polyline widths exploded to
zero-width LINEs; **P2** DXF `$INSUNITS` was dropped so inch vs mm
with the same numbers scored L4 1.0. Strip inline geom after L4;
fail closed on width and unknown/mismatched units; extract cache v3.
HEAD `e3357abd`. Never merge; decisions stay off.

| Item | State |
|---|---|
| P1 live geom_json | stripped after L4; EvidencePack/GET/audit have no key |
| P2 polyline width | explode refuses nonzero/non-finite width |
| P2 $INSUNITS | missing vs set, 0, or mismatch → no L4 |
| Local `make test-review-reuse` | **208 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave72 — 2026-09-19 ~11:21 UTC

Sol 5.6 vs `origin/main` on `9dfc3b87`: **P1** POINT/SOLID/3DFACE/XLINE
/MESH (and other leftover geom types) were dropped so a shared LINE
scored L4 1.0; **P2** `_top_confidence` skipped independent rejects
only when `state == different`, so similar+`version_gate_filtered`+L4
0.99 stayed high. Fail closed on unknown geom; skip independent
reasons regardless of state. Never merge; decisions stay off.

| Item | State |
|---|---|
| P1 leftover geom | annotation allowlist; unknown types fail closed |
| P2 confidence | independent reasons skip `_top_confidence` |
| Local `make test-review-reuse` | **198 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave70 — 2026-09-19 ~11:02 UTC

Sol 5.6 vs `origin/main` on `89a6667c`: **P1** a valid LINE plus an
`INSERT` was admitted, then `_geometry_only_geom` dropped the INSERT,
so different blocks with the same incidental LINE scored L4 1.0;
**P1** SPLINE identity omits knots/weights and the matcher uses 16
control points. Fail closed on INSERT/SPLINE presence. Never merge;
decisions stay off.

| Item | State |
|---|---|
| P1 mixed INSERT | `_UNSCORED_GEOM_TYPES` includes INSERT |
| P1 spline | `_UNSCORED_GEOM_TYPES` includes SPLINE |
| Local `make test-review-reuse` | **194 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave68 — 2026-09-19 ~10:42 UTC

Sol 5.6 vs `origin/main` on `0414daca`: **P1** INSERT with only a
block name+pose is admitted as L4, so different `DOOR` definitions
score 1.0; **P1** pre-bulge `extract_sig` cache hits skip the new
bulge field; **P2** cleanup deletes hashed siblings while refusing a
mixed dir in the same tenant group. Require `block_hash`; version the
extract cache; refuse the whole group. Never merge; decisions stay off.

| Item | State |
|---|---|
| P1 INSERT hash | `_is_geom_entity` requires `block_hash` |
| P1 extract cache | `_EXTRACT_CACHE_VERSION=2` |
| P2 cleanup group | mixed sibling refuses all grouped dirs |
| Local `make test-review-reuse` | **191 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave67 — 2026-09-19 ~10:35 UTC

Sol 5.6 vs `origin/main` on `0414daca`: **P1** bulge 0.0004 quantized
to 0 and exploded to a chord; **P1** 3 vs 16 spline controls with the
same head scored 1.0; **P1** swapped INSERT hashes kept set equality;
**P1** opposite half-ellipses 0..π vs π..2π scored 1.0 on span only.
Never merge; decisions stay off.

| Item | State |
|---|---|
| P1 bulge | `_bulge_is_unsafe` without 3-decimal rounding |
| P1 spline counts | `_spline_control_counts` must match |
| P1 INSERT identity | per (block, insert) hash compare |
| P1 ellipse | partial ellipses are not L4 geom |
| Local `make test-review-reuse` | **188 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave66 — 2026-09-19 ~10:22 UTC

Sol 5.6 vs `origin/main` on `66205333`: **P1** `dxf_extract` stored
LWPOLYLINE vertices as `[x, y]`, so `_polyline_has_bulge` never saw
DXF curves and a chord still scored L4 1.0; **P1** SPLINE matching
uses only 16 control points, so identical heads with different tails
scored 1.0. Keep `bulges` in extract; refuse splines with >16 controls.
Never merge; decisions stay off.

| Item | State |
|---|---|
| P1 DXF bulge | `_polyline_xy_and_bulges` on extract |
| P1 spline cap | `_L4_MAX_SPLINE_CTRL=16` in `_is_geom_entity` |
| Local `make test-review-reuse` | **184 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave65 — 2026-09-19 ~10:12 UTC

Sol 5.6 vs `origin/main` on `167831f6`: **P1** exploding LWPOLYLINE
drops bulge, so a semicircle and its chord scored L4 1.0; **P1** same
INSERT pose with different `block_hash` fused to 0.5556 and passed
the L4 threshold. Fail closed on bulge; veto hash mismatch as 0.0.
Never merge; decisions stay off.

| Item | State |
|---|---|
| P1 bulge | `_polyline_has_bulge` empties geometry-only entities |
| P1 block_hash | `_insert_block_hash_conflict` returns 0.0 |
| Local `make test-review-reuse` | **182 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave64 — 2026-09-19 ~10:03 UTC

Sol 5.6 vs `origin/main` on `1a99996f`: **P1** `entities_similarity`
truncates to `min(len(A), len(B))`, so a one-LINE query vs that LINE
plus extra geometry scored L4 1.0; **P1** raising `max_match_entities`
to the exploded count is unbounded. Penalize unmatched counts; keep a
hard 128-entity cap and refuse over-cap L4. Never merge; decisions off.

| Item | State |
|---|---|
| P1 unmatched | `_penalize_unmatched` min/max count ratio |
| P1 matcher bound | `_L4_MAX_MATCH_ENTITIES=128`; over-cap returns None |
| Local `make test-review-reuse` | **180 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave63 — 2026-09-19 ~09:55 UTC

Sol 5.6 vs `origin/main` on `4362c886`: **P1** exploding polylines into
LINEs hits `max_match_entities=64`, so 101-point polylines that match
on the first 64 segments and diverge after scored L4 1.0. Raise the
matcher cap to the exploded entity count. Never merge; decisions stay
off.

| Item | State |
|---|---|
| P1 matcher cap | `_try_l4_score` sets `max_match_entities` to full count |
| Local `make test-review-reuse` | **178 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave62 — 2026-09-19 ~09:43 UTC

Sol 5.6 vs `origin/main` on `48ec1243`: **P1** PrecisionVerifier
canonicalizes each `LWPOLYLINE`/`POLYLINE` (translate/scale/rotate)
so identical shapes at different coordinates score ~1.0 and are
certified `precision-l4`. Explode polylines to positional LINEs.
Never merge; decisions stay off.

| Item | State |
|---|---|
| P1 polyline pose | `_explode_polyline` before PrecisionVerifier |
| Local `make test-review-reuse` | **177 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave61 — 2026-09-19 ~09:24 UTC

Sol 5.6 vs `origin/main` on `256068ce`: **P2** `w_layers=0` does not
clear `layer_mismatch_penalty`, so identical LINEs on different CAD
layers can score 0 and be labeled `different`. Strip entity `layer`
and force `layer_mismatch_penalty=0`. Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 layer penalty | strip `layer`; Settings `layer_mismatch_penalty=0` |
| Local `make test-review-reuse` | **175 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave60 — 2026-09-19 ~09:15 UTC

Sol 5.6 vs `origin/main` on `9806f46b`: **P2** `_geometry_only_geom`
kept HATCH rows (not in `_GEOM_ENTITY_TYPES`). Matching HATCH plus
orthogonal LINEs scored ~0.667 and passed L4. Keep only
`_is_geom_entity` primitives. Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 HATCH | `_geometry_only_geom` keeps validated geom entities only |
| Local `make test-review-reuse` | **174 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave59 — 2026-09-19 ~09:02 UTC

Sol 5.6 vs `origin/main` on `b86733a4`: **P2** `_is_geom_entity`
validated raw coords, then PrecisionVerifier rounded to 3 decimals.
Orthogonal sub-0.001 LINEs both collapsed to zero-length and scored
L4 1.0. Quantize the admission gate and re-check after `normalize_v2`.
Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 quantized geom | `_quantized` / post-normalize `_is_geom_json` |
| Local `make test-review-reuse` | **173 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave58 — 2026-09-19 ~09:00 UTC

Sol 5.6 vs `origin/main` on `b86733a4`: **P2** cleanup treated
unreadable `tenant_meta.json` as missing and could `rmtree` a dir
whose identity sidecar could not be validated. Fail closed like the
FS store (`_UNREADABLE`). Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 cleanup meta | `_tenant_id_from_meta` returns `_UNREADABLE`; mixed refuse |
| Local `make test-review-reuse` | **172 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave57 — 2026-09-19 ~08:51 UTC

Sol 5.6 vs `origin/main` on `a3d97c35`: **P1** cloning
`PrecisionVerifier().settings` still keeps `entities_geom_hash=True`
when `CAD_ML_PLATFORM_L4_ENTITIES_GEOM_HASH=1`, so layout-shifted
clones are certified as `precision-l4`. Force `entities_geom_hash=False`
in the ReviewReuse L4 config. Never merge; decisions stay off.

| Item | State |
|---|---|
| P1 env geom-hash | `_try_l4_score` replace() sets `entities_geom_hash=False` |
| Local `make test-review-reuse` | **171 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave56 — 2026-09-19 ~08:40 UTC

Sol 5.6 vs `origin/main` on `fa3eb91d`: **P1** `replace(Settings())`
re-enabled `entities_geom_hash` so layout-shifted clones scored ~1.0;
**P1** live `precision_score` is fused and was stamped as geometric L4.
Keep platform geom-hash off; do not copy fused remote scores; local
geometry-only rescore wins when geom is present. Never merge.

| Item | State |
|---|---|
| P1 geom-hash | `_try_l4_score` clones `PrecisionVerifier().settings` |
| P1 live fused | `vision_response_to_hits` leaves geometric None |
| Local `make test-review-reuse` | **170 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave54 remaining P2 — 2026-09-19 ~08:25 UTC

Sol 5.6 vs `origin/main` on `d31a9942`: P2 local `PrecisionVerifier`
NaN/inf/out-of-range scores were trusted as L4; P2 unreadable
`tenant_meta.json` with no tasks was treated as empty and overwritten.
Reject non-finite local scores; occupy the dir as `_UNREADABLE`.
Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 local L4 range | `_try_l4_score` uses `_is_finite_unit_score` |
| P2 unreadable meta | `_existing_dir_tenants` fail-closed |
| Local `make test-review-reuse` | **167 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave55 — 2026-09-19 ~08:30 UTC

Sol 5.6 vs `origin/main` on `d31a9942`: **P1** fused
`PrecisionVerifier.score_pair()` (entities+text+layers+dims) was stored
as `scores["geometric"]`. Orthogonal LINEs plus shared TEXT scored
~0.58, above `LOW_PRECISION_THRESHOLD`, and kept `similar` with
`precision-l4`. Score geometry only (drop annotation entities; zero
text/layers/dims weights). Never merge; decisions stay off.

| Item | State |
|---|---|
| P1 geometry-only L4 | `_geometry_only_geom` + Settings w_text/layers/dims=0 |
| Local `make test-review-reuse` | **168 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave54 — 2026-09-19 ~08:20 UTC

Sol 5.6 vs `origin/main` on `21c0dfbc`: P2 live
`vision_response_to_hits` dropped match `geom_json`, so L4-verifiable
hits without a unit `precision_score` (and without a 64-hex file hash)
were labeled `missing_geom_json` / `vision_only_unverified`. Forward
inline geometry. Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 live geom_json | `vision_response_to_hits` copies match/provenance geom |
| Local `make test-review-reuse` | **164 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave53 — 2026-09-19 ~08:05 UTC

`tests (3.10)` green on docs HEAD `b76c67d8` (product `4858606b`).
Sol 5.6 vs `origin/main`: P2 mixed valid LINE + zero-radius CIRCLE
still admitted by `_is_geom_json` `any(...)` and scored as similar.
Reject malformed supported entities before L4; drop them in
`_canonical_geom`. Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 mixed entities | `_is_geom_json` fails closed; `_canonical_geom` drops junk |
| Local `make test-review-reuse` | **163 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave51 — 2026-09-19 ~07:16 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `4858606b`.
Sol 5.6 vs `origin/main`: **no P1/P2**. Never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `4858606b` |
| Last fully green 3.10/3.11 | `4858606b` |
| Local `make test-review-reuse` | **161 passed** (not re-run; no product change) |
| Sol wave51 | no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave50 — 2026-09-19 ~06:42 UTC

`2c32cae0` is a new product SHA (zero-sweep ELLIPSE). Sol 5.6 vs
`origin/main`: P2 omitted full-ellipse params must canonicalize to
`0..2π`; P2 non-finite INSERT rotation must not be L4. Implemented.
Never merge; decisions stay off.

| Item | State |
|---|---|
| P2 ellipse | `_canonical_geom` fills omitted `start_param`/`end_param` |
| P2 INSERT | `_is_geom_entity` requires finite rotation when present |
| Local `make test-review-reuse` | **161 passed** |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |

## Window 3 wave45 — 2026-09-19 ~05:04 UTC

`tests (3.10)` / `tests (3.11)` / `e2e-smoke` **pass** on `f2acff4f`.
Sol 5.6 vs `origin/main` still **blocked** (Codex usage limit until
2026-09-23 22:47 UTC). Last completed Sol is wave32 on `d46c6156`:
**no P1/P2**. Product SHA recorded; never merge; decisions stay off.

| Item | State |
|---|---|
| HEAD | `f2acff4f` |
| Last fully green 3.10/3.11 | `f2acff4f` |
| Local `make test-review-reuse` | **158 passed** |
| Sol wave45 | usage-limited; last clean = wave32 no P1/P2 |
| Evaluation Report | fail (Track E; out of scope) |
| Merge | still not done |
