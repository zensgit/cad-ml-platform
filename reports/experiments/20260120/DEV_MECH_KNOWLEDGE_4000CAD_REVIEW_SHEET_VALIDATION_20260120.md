# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_SHEET_VALIDATION_20260120

## Summary
Validated the DXF review sheet output sizes and extraction coverage metrics.

## Command
```
python3 - <<'PY'
import csv
from pathlib import Path

path = Path('reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_20260120.csv')
rows = []
with path.open('r', encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)

count = len(rows)
text_non_empty = sum(1 for r in rows if (r.get('text_sample') or '').strip())
normalized_non_empty = sum(1 for r in rows if (r.get('normalized_text_sample') or '').strip())
part_name_non_empty = sum(1 for r in rows if (r.get('title_block_part_name') or '').strip())
drawing_non_empty = sum(1 for r in rows if (r.get('title_block_drawing_number') or '').strip())

print('rows', count)
print('text_sample_non_empty', text_non_empty)
print('normalized_text_non_empty', normalized_non_empty)
print('title_block_part_name', part_name_non_empty)
print('title_block_drawing_number', drawing_non_empty)
PY
```

## Result
- rows: 200
- text_sample_non_empty: 122
- normalized_text_non_empty: 122
- title_block_part_name: 4
- title_block_drawing_number: 0
