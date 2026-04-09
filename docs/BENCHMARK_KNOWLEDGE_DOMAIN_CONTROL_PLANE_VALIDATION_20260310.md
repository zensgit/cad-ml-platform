# Benchmark Knowledge Domain Control Plane Validation

## Scope

- add a benchmark-level `knowledge_domain_control_plane` component
- unify knowledge benchmark control signals from:
  - `knowledge_domain_capability_matrix`
  - `knowledge_domain_capability_drift`
  - `knowledge_realdata_correlation`
  - `knowledge_outcome_correlation`
  - `knowledge_domain_action_plan`
- propagate control-plane status into:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook

## Design

The new control-plane layer turns fragmented standards / tolerance / GD&T
signals into a single benchmark control surface.

Per domain, it summarizes:

- foundation and application readiness
- real-data and outcome alignment
- drift state
- missing metrics and weak surfaces
- next action and action priority
- release blockers

Current domains:

- `tolerance`
- `standards`
- `gdt`

Top-level outputs include:

- `status`
- `ready_domain_count`
- `partial_domain_count`
- `blocked_domain_count`
- `missing_domain_count`
- `release_blockers`
- `priority_domains`
- `focus_areas`
- `focus_areas_detail`
- `domains`
- `total_action_count`
- `high_priority_action_count`
- `recommendations`

## Files

- `src/core/benchmark/knowledge_domain_control_plane.py`
- `src/core/benchmark/__init__.py`
- `scripts/export_benchmark_knowledge_domain_control_plane.py`
- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_knowledge_domain_control_plane.py`

## Verification

Commands run in isolated worktree:
`/private/tmp/cad-ml-platform-stdtolgdt-control-plane-20260310`

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_control_plane.py \
  scripts/export_benchmark_knowledge_domain_control_plane.py \
  tests/unit/test_benchmark_knowledge_domain_control_plane.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py

flake8 \
  src/core/benchmark/knowledge_domain_control_plane.py \
  scripts/export_benchmark_knowledge_domain_control_plane.py \
  tests/unit/test_benchmark_knowledge_domain_control_plane.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_knowledge_domain_control_plane.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `3 passed`

## Expected Result

- standalone exporter writes valid JSON and Markdown control-plane outputs
- bundle / companion / release decision / release runbook expose:
  - `knowledge_domain_control_plane_status`
  - `knowledge_domain_control_plane`
  - `knowledge_domain_control_plane_domains`
  - `knowledge_domain_control_plane_release_blockers`
  - `knowledge_domain_control_plane_recommendations`
- release surfaces can now distinguish isolated signal regressions from
  cross-domain control-plane blockers

## Limitations

- this layer only covers exporter and downstream surfaces
- CI wiring and PR comment exposure stay in stacked follow-up branches
- domain coverage still depends on upstream standards / tolerance / GD&T
  benchmark inputs being present
