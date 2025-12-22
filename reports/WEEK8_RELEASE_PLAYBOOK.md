#!/usr/bin/env markdown
# Week 8 Report: Release Playbook

## Summary
- Documented dual-release workflow for rules and ML models.

## Changes
- Added `docs/RELEASE_PLAYBOOK.md`
- Linked in `README.md`

## Tests
- Not run (manual API verification recommended).

## Verification
- Rules: `/api/v1/maintenance/knowledge/reload` + `/knowledge/status`
- Model: `/api/v1/model/reload` + `/api/v1/model/version`
