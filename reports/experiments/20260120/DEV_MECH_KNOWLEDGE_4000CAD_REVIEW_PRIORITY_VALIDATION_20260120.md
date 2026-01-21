# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_PRIORITY_VALIDATION_20260120

## Summary
Validated the priority review list size and summary counts.

## Command
```
python3 - <<'PY'
import csv
from pathlib import Path

priority_path = Path('reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_20260120.csv')
summary_path = Path('reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_SUMMARY_20260120.csv')

with priority_path.open('r', encoding='utf-8') as handle:
    priority_rows = list(csv.DictReader(handle))

with summary_path.open('r', encoding='utf-8') as handle:
    summary_rows = list(csv.DictReader(handle))

print('priority_rows', len(priority_rows))
print('summary_rows', len(summary_rows))
PY
```

## Result
- priority_rows: 10
- summary_rows: 9
