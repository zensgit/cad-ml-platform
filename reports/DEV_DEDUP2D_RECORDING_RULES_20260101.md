# DEV_DEDUP2D_RECORDING_RULES_20260101

## Design Summary
- Added Dedup2D recording rules for job success/error rates, throughput per minute, and storage error/latency metrics.
- Rules align with emitted metrics (`dedup2d_jobs_total`, `dedup2d_file_*`) and preserve backend labels where relevant.

## Files Updated
- `docs/prometheus/recording_rules.yml`

## Verification
```bash
python3 - <<'PY'
import yaml
from pathlib import Path
path = Path('docs/prometheus/recording_rules.yml')
with path.open('r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
assert 'groups' in data
print('groups', len(data['groups']))
PY
```
Result: YAML parsed successfully.
