# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_PRIORITY_13E_VALIDATION_20260120

## Summary
Validated the final 13-sample priority list size and label breakdown.

## Command
```
python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path

path = Path('reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_13E_20260120.csv')
with path.open('r', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))

labels = Counter((row.get('suggested_label_cn') or '').strip() for row in rows)
print('rows', len(rows))
print('unique_labels', len(labels))
print('top_labels', labels.most_common(5))
PY
```

## Result
- rows: 13
- unique_labels: 10
- top_labels: 模板=4, 三视图=1, 三视图练习=1, 基准代号=1, 底座视图=1
