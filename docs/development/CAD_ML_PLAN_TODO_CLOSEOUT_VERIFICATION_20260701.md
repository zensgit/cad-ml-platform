# CAD ML plan/TODO closeout verification (2026-07-01)

## Scope

This is a docs-only closeout for `docs/DETAILED_DEVELOPMENT_PLAN.md`.
The plan already stated that phases 1A-6 were complete, but its P1/P2/P3
follow-up list still read like open work. This record re-checks the current
main branch and makes the remaining TODO pool accurate.

## Current baseline

- Repository: `zensgit/cad-ml-platform`
- Baseline: `origin/main@561da8eb`
- Prepared from an isolated worktree based on `origin/main`.

## Closeout evidence

| Follow-up | Evidence on main | Verification surface |
|---|---|---|
| P1 auto-recovery persistence / `next_recovery_eta` | `src/api/v1/health.py` returns `next_recovery_eta` and `manual_recovery_in_progress` from the Faiss recovery state; `src/utils/analysis_metrics.py` exposes `faiss_next_recovery_eta_seconds`; `.github/workflows/stress-tests.yml` asserts both health keys. | `tests/unit/test_faiss_eta_schedules_on_failed_recovery.py`, `tests/unit/test_faiss_eta_reset_on_recovery.py`, `tests/unit/test_degraded_mode_health.py`, `tests/unit/test_faiss_health_response.py`. |
| P2 preview percentiles / distribution breakdown | `VectorMigrationPreviewResponse` includes `avg_delta` and `median_delta`; migration status/summary/pending/trends models expose version distributions, distribution completeness, pending counts, scan limits, and readiness fields; `vectors_stats.py` exposes material/complexity/format/coarse/decision-source distribution and backend health. | `tests/unit/test_migration_preview_trends.py`, `tests/unit/test_vector_migration_models.py`, `tests/unit/test_vectors_migration_read_router.py`, `tests/unit/test_vector_stats.py`, `tests/unit/test_vector_distribution_endpoint.py`. |
| P3 ops issue templates / CI hardening | `.github/ISSUE_TEMPLATE/degraded-incident.md`, `cache-rollback-request.md`, and `observability-incident.md` exist; Makefile includes `check-gh-actions-ready`, `check-gh-actions-ready-soft`, `validate-check-gh-actions-ready`, `validate-watch-commit-workflows`, and `validate-ci-watchers`. | `tests/unit/test_watch_commit_workflows_make_target.py`, `tests/unit/test_check_gh_actions_ready.py`, CI workflow inventory/readiness guard tests referenced from `Makefile`. |

## Historical coverage-plan note

`claudedocs/TEST_COVERAGE_PLAN.md` is an older February planning artifact. Its
remaining unchecked week-4 rows are not treated as current release blockers:
the repo now has tiered test strategy documentation, CI watcher hardening, and
hundreds of test modules. If a future coverage target is needed, it should be
opened as a fresh measured coverage gate with a current coverage report rather
than reviving that stale checklist.

## Verification performed for this closeout

- Inspected the plan and implementation files above on `origin/main`.
- Confirmed the issue templates exist under `.github/ISSUE_TEMPLATE`.
- Confirmed the current tree contains 871 `tests/test_*.py` style test files.
- Used the repo's Python 3.11 test environment (`.venv311`) because the
  system Python 3.9 cannot evaluate the codebase's PEP 604 annotations.
- Ran
  `python -m pytest -q tests/unit/test_check_gh_actions_ready.py tests/unit/test_watch_commit_workflows_make_target.py`
  -> 21 passed.
- Ran `python -m pytest -q tests/unit/test_vector_migration_models.py` -> 10
  passed.
- Ran
  `python -m pytest -q tests/unit/test_degraded_mode_health.py tests/unit/test_faiss_health_response.py tests/unit/test_faiss_eta_schedules_on_failed_recovery.py tests/unit/test_faiss_eta_reset_on_recovery.py`
  -> 5 passed.
- Ran `git diff --check` after the docs update.

## Outcome

No unimplemented P1/P2/P3 work remains in `docs/DETAILED_DEVELOPMENT_PLAN.md`.
The CAD ML plan/TODO pool now points to the implemented evidence instead of
presenting already-shipped follow-ups as open work.
