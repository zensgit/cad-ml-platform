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
