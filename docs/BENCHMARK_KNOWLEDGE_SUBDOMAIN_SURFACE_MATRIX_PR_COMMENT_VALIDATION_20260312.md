# Benchmark Knowledge Subdomain Surface Matrix PR Comment Validation

## Goal

Expose `benchmark_knowledge_subdomain_surface_matrix` in PR comment output and
signal lights so benchmark reviews surface subdomain-level knowledge gaps
without opening artifacts manually.

## Key Changes

- Added PR comment variables for:
  - `status`
  - `total / ready / partial / blocked`
  - `priority_subdomains`
  - `public_api_gap_subdomains`
  - `reference_gap_subdomains`
  - `recommendations`
  - `artifact`
- Added status line row:
  - `Benchmark Knowledge Subdomain Surface Matrix`
- Added signal light row:
  - `Benchmark Knowledge Subdomain Surface Matrix`
- Extended workflow contract coverage in:
  - `tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`

## Validation

Commands:

```bash
python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py

flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py \
  --max-line-length=100

pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py
```

Results:

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `13 passed, 1 warning`

## Outcome

The benchmark PR comment now has a dedicated, visible line for subdomain-level
knowledge surface readiness and can signal whether the current codebase is ready
or blocked in specific standards/tolerance/GD&T subareas.
