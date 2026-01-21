# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_PRIORITY_30D_VALIDATION_20260120

## Summary
Validated the Top-30D priority list size and label breakdown.

## Command
```
python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path

path = Path('reports/experiments/20260120/MECH_4000_DWG_REVIEW_SHEET_PRIORITY_30D_20260120.csv')
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
- unique_labels: 25
- top_labels: 泵盖图=2, 激光固定座=2, 紧固件=2, 轴的标注=2, 阶梯轴=2
