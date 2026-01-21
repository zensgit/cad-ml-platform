# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_PRIORITY_PACK_30B_VALIDATION_20260120

## Summary
Validated the second Top-30 review pack contents (HTML + CSV + previews).

## Command
```
python3 - <<'PY'
from pathlib import Path
import csv

pack_dir = Path('reports/experiments/20260120/MECH_4000_DWG_REVIEW_PRIORITY_PACK_30B_20260120')
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
- html_exists: True
- csv_rows: 30
- preview_count: 30
