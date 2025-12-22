#!/usr/bin/env markdown
# Replay Validation

Replay validation runs a list of CAD files through the analysis API and stores
the results for offline review.

## Prepare Input List

Create a text file with one path per line:

```
data/standards_dxf/Bolt_M6x20.dxf
data/standards_dxf/Washer_M6.dxf
```

## Run

```
python3 scripts/replay_analysis.py --input-list path/to/list.txt
```

Output:
- `reports/replay/replay_results.jsonl`
- `reports/replay/summary.json`

## Dry Run

```
python3 scripts/replay_analysis.py --input-list path/to/list.txt --dry-run
```
