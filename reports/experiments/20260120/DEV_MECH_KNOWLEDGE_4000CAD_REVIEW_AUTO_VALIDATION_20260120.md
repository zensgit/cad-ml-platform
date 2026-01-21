# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_AUTO_VALIDATION_20260120

## Summary
Validated the auto-review outputs and confirmed counts for confirmed vs
follow-up samples.

## Command
```
python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path

path = Path('reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_AUTO_20260120.csv')
rows = []
with path.open('r', encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)

status = Counter(row.get('review_status') for row in rows)
print('review_status', dict(status))
PY
```

## Result
- review_status: confirmed=66, pending=134
