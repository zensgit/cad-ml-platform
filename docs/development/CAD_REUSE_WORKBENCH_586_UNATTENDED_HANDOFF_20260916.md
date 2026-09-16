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
