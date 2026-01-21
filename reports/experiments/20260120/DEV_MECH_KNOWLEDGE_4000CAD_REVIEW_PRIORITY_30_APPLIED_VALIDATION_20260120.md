# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_PRIORITY_30_APPLIED_VALIDATION_20260120

## Summary
Validated updated auto-review counts, conflict size, and refreshed Top-30 pack
contents after applying manual decisions.

## Commands
```
./.venv-graph/bin/python scripts/auto_review_dxf_sheet.py \
  --input reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_20260120.csv \
  --output reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_AUTO_20260120.csv \
  --conflicts reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_CONFLICTS_20260120.csv \
  --confidence-threshold 0.05

python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path

path = Path('reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_AUTO_20260120.csv')
with path.open('r', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))

status = Counter((row.get('review_status') or '').strip() for row in rows)
verdict = Counter((row.get('auto_review_verdict') or '').strip() for row in rows)
manual_confirmed = sum(1 for row in rows if (row.get('auto_review_verdict') or '').strip() == 'manual_confirmed')
print('review_status', dict(status))
print('auto_review_verdict', dict(verdict))
print('manual_confirmed', manual_confirmed)
PY

python3 - <<'PY'
from pathlib import Path
import csv

pack_dir = Path('reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_30_20260120')
html_path = pack_dir / 'index.html'
csv_path = pack_dir / 'review_priority_pack.csv'
preview_dir = pack_dir / 'previews'

rows = []
with csv_path.open('r', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))

preview_count = len(list(preview_dir.glob('*.png')))
print('html_exists', html_path.exists())
print('csv_rows', len(rows))
print('preview_count', preview_count)
PY
```

## Result
- review_status: confirmed=104, pending=96
- auto_review_verdict: confirmed=64, manual_confirmed=40, needs_followup=96
- manual_confirmed: 40
- conflicts: 96 rows
- pack: html_exists=True, csv_rows=30, preview_count=30
