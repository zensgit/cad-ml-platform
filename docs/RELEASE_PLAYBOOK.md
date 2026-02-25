#!/usr/bin/env markdown
# Release Playbook (Rules + Models)

This playbook defines a dual-release flow for rules and ML models with rollback.

## Pre-Release CI Gate (Required)

Run the CI watcher gate on the exact commit to be released before any rule/model rollout.

1. Resolve the target commit SHA (for example, `HEAD` or a pushed SHA on `main`).
2. Run the safe auto watcher:
   - `make watch-commit-workflows-safe-auto CI_WATCH_SHA=HEAD CI_WATCH_SUCCESS_CONCLUSIONS="success,skipped,neutral"`
3. Confirm artifact files are generated under `reports/ci/`:
   - `gh_readiness_watch_<sha>.json`
   - `watch_commit_<sha>_summary.json`
4. Release gate acceptance criteria:
   - `exit_code = 0`
   - `reason = all_workflows_success`
   - `counts.failed = 0`
   - `counts.missing_required = 0`
5. If gate fails, stop release and inspect failed workflows via `gh run view <run_id>`.

## Rule Release

1. Update JSON under `data/knowledge/`
2. Reload rules:
   - API: `POST /api/v1/maintenance/knowledge/reload`
   - CLI: `python3 scripts/reload_knowledge.py`
3. Verify:
   - `GET /api/v1/maintenance/knowledge/status`

## Model Release

1. Upload model file to the target host
2. Reload model:
   - API: `POST /api/v1/model/reload`
3. Verify:
   - `GET /api/v1/model/version`

## Rollback

- Rules: replace JSON with previous version and reload.
- Model: re-run reload with the previous model path/version.

## Recommended Order

1. Pre-release CI gate (`watch-commit-workflows-safe-auto`)
2. Rules first (low-risk)
3. Model next (requires admin token)
