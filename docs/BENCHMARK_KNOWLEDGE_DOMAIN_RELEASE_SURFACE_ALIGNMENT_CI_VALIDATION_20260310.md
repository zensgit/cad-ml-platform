# Benchmark Knowledge Domain Release Surface Alignment CI Validation

## Scope
- workflow dispatch inputs for `benchmark_knowledge_domain_release_surface_alignment`
- standalone CI build step and artifact upload
- downstream wiring into:
  - `benchmark_competitive_surpass_index`
  - `benchmark_artifact_bundle`
  - `benchmark_companion_summary`
- job summary exposure for standalone, bundle, and companion surfaces

## Files
- `.github/workflows/evaluation-report.yml`
- `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Changes
- added dispatch inputs:
  - `benchmark_knowledge_domain_release_surface_alignment_enable`
  - `benchmark_knowledge_domain_release_surface_alignment_release_decision_json`
  - `benchmark_knowledge_domain_release_surface_alignment_release_runbook_json`
  - `benchmark_competitive_surpass_index_knowledge_domain_release_surface_alignment_json`
  - `benchmark_artifact_bundle_knowledge_domain_release_surface_alignment_json`
  - `benchmark_companion_summary_knowledge_domain_release_surface_alignment_json`
- added env defaults and output paths for standalone exporter
- added step:
  - `Build benchmark knowledge domain release surface alignment (optional)`
- added artifact upload step
- wired new JSON into competitive index, artifact bundle, and companion summary
- extracted new downstream outputs:
  - `knowledge_domain_release_surface_alignment_status`
  - `knowledge_domain_release_surface_alignment_summary`
  - `knowledge_domain_release_surface_alignment_mismatches`
- added summary lines for:
  - standalone benchmark
  - artifact bundle surface
  - companion summary surface

## Validation
```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100
pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
python3 - <<'PY'
import yaml
from pathlib import Path
path = Path('.github/workflows/evaluation-report.yml')
yaml.safe_load(path.read_text(encoding='utf-8'))
print('YAML_OK')
PY
git diff --check
```

## Result
- `pytest`: `3 passed, 1 warning`
- workflow YAML parse: `YAML_OK`
- `py_compile`: passed
- `flake8`: passed
- `git diff --check`: passed

## Notes
- This CI layer intentionally wires only the surfaces already touched by the main-layer exporter branch:
  - standalone exporter
  - competitive surpass index
  - artifact bundle
  - companion summary
- `release decision` and `release runbook` CI wiring for this component remain out of scope for this stacked PR.
