# #586 12-hour unattended handoff

**Deadline**: 2026-09-16 17:00 UTC. After that, stop coding, append this file, delete the scheduler.

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
