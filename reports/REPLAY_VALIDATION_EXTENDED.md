#!/usr/bin/env markdown
# Replay Validation (Extended)

## Summary
- Replayed full standards DXF set plus sample STEP against local API.

## Tests
- Local service: `python3 -m uvicorn src.main:app --host 127.0.0.1 --port 8010`
- Replay: `python3 scripts/replay_analysis.py --input-list /tmp/replay_list.txt --api-url http://127.0.0.1:8010/api/v1 --output-dir reports/replay_extended`

## Results
- `reports/replay_extended/summary.json`: total=26, success=26, error=0
- `reports/replay_extended/replay_results.jsonl`
