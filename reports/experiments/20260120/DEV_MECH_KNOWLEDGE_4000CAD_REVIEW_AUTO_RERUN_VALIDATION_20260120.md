# DEV_MECH_KNOWLEDGE_4000CAD_REVIEW_AUTO_RERUN_VALIDATION_20260120

## Summary
Validated the rerun auto-review counts and verified manual-confirmed entries
remain locked.

## Command
```
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
```

## Result
- review_status: confirmed=74, pending=126
- auto_review_verdict: confirmed=64, manual_confirmed=10, needs_followup=126
- manual_confirmed: 10
