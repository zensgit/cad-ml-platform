# Benchmark Competitive Surpass Trend CI Validation

## Scope
- Wire `benchmark_competitive_surpass_trend` into `.github/workflows/evaluation-report.yml`
- Expose workflow-dispatch inputs for current/previous summary handoff
- Build/upload the trend artifact in CI
- Pass trend JSON through:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Surface trend status in job summary

## Design
- Keep `benchmark_competitive_surpass_trend` as a standalone benchmark component, parallel to `benchmark_competitive_surpass_index`
- Use `benchmark_competitive_surpass_index` output as the default `--current-summary` handoff when available
- Preserve explicit override hooks through workflow-dispatch inputs and env vars
- Limit this layer to CI/build/summary wiring only; PR comment and signal lights stay in the next stacked PR

## Key Changes
- Added workflow-dispatch inputs:
  - `benchmark_competitive_surpass_trend_enable`
  - `benchmark_competitive_surpass_trend_current_summary_json`
  - `benchmark_competitive_surpass_trend_previous_summary_json`
  - passthrough JSON inputs for bundle / companion / release / runbook
- Added env vars:
  - `BENCHMARK_COMPETITIVE_SURPASS_TREND_*`
  - downstream passthrough env vars for bundle / companion / release / runbook
- Added workflow step:
  - `Build benchmark competitive surpass trend (optional)`
- Added upload step:
  - `Upload benchmark competitive surpass trend`
- Extended downstream build steps to accept `--benchmark-competitive-surpass-trend`
- Extended job summary with:
  - standalone trend lines
  - bundle trend lines
  - companion trend lines
  - release decision trend lines
  - release runbook trend lines

## Validation
```bash
python3 - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path('.github/workflows/evaluation-report.yml').read_text())
print('yaml_ok')
PY

python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

git diff --check
```

## Result
- YAML parse: passed
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `3 passed`
- `git diff --check`: passed

## Notes
- This layer intentionally does not yet modify PR comment / signal lights
- PR comment wiring is expected in the next stacked PR on top of this branch
