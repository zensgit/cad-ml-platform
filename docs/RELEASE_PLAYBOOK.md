#!/usr/bin/env markdown
# Release Playbook (Rules + Models)

This playbook defines a dual-release flow for rules and ML models with rollback.

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

1. Rules first (low-risk)
2. Model next (requires admin token)
