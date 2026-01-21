# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_PRIORITY_30C_VALIDATION_20260120

## Summary
Validated the Top-30C priority list size and label breakdown.

## Command
```
python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path

path = Path('reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_30C_20260120.csv')
with path.open('r', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))

labels = Counter((row.get('suggested_label_cn') or '').strip() for row in rows)
print('rows', len(rows))
print('unique_labels', len(labels))
print('top_labels', labels.most_common(5))
PY
```

## Result
- rows: 30
- unique_labels: 24
- top_labels: 盖=3, 圆柱滚子轴承=2, 装配图=2, 轴类=2, 零件轴改=2
