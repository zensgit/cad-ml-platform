#!/usr/bin/env markdown
# Week 10 Report: Replay Validation

## Summary
- Added replay script and documentation for real-data validation.

## Changes
- Script: `scripts/replay_analysis.py`
- Doc: `docs/REPLAY_VALIDATION.md`
- Linked in `README.md`

## Tests
- `python3 -m uvicorn src.main:app` (local health check on port 8010)
- `python3 scripts/replay_analysis.py --input-list /tmp/replay_list.txt --api-url http://127.0.0.1:8010/api/v1`

## Verification
- Full run writes:
  - `reports/replay/replay_results.jsonl`
  - `reports/replay/summary.json` (total=2, success=2, error=0)
