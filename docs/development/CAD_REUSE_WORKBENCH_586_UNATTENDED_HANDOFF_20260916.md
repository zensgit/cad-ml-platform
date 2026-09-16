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
2. Wave 2026-09-16 ~11:15 UTC (grok-4.6): Sol P2 on `c89fdcb2` — do not
   persist `str(exc)` on `task.error` / failed events; GET and audit
   export stay generic (`review-reuse pipeline failed`). Local
   `make test-review-reuse` **125 passed**.
3. After push: wait for 3.10 CI on the new SHA; if red, fix; else Sol
   re-review vs `origin/main`; implement P1/P2 honesty findings only.
4. Evaluation Report (Track E) stays out of scope.

## Do not

- Merge #586
- Open a second L3 runtime PR
- Invent Track C evidence
- Enable decisions in production config
