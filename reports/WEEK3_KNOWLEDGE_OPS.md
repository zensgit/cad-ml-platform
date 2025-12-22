#!/usr/bin/env markdown
# Week 3 Report: Knowledge Rules Ops

## Summary
- Added knowledge rule hot-reload endpoints and CLI tooling.

## Changes
- API endpoints in `src/api/v1/maintenance.py`:
  - `POST /api/v1/maintenance/knowledge/reload`
  - `GET /api/v1/maintenance/knowledge/status`
- CLI tool: `scripts/reload_knowledge.py`
- Doc: `docs/KNOWLEDGE_RULES_OPERATIONS.md` + README link
- `scripts/reload_knowledge.py` now adds repo root to `sys.path`.

## Tests
- `python3 scripts/reload_knowledge.py`

## Verification
- CLI output shows version change and reload success.
- API: call reload/status endpoints with maintenance API key
