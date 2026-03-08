# Benchmark Knowledge Drift Release Validation

Date: 2026-03-08

## Scope

- Extend `scripts/export_benchmark_release_decision.py` with optional
  `benchmark_knowledge_drift` input support.
- Extend `scripts/export_benchmark_release_runbook.py` with optional
  `benchmark_knowledge_drift` input support.
- Expose knowledge drift artifact rows and stable payload fields in both
  exporters.
- Render knowledge drift status, summary, counts, and focus-area lists in
  Markdown outputs.
- Keep the existing contract compatible by only adding fields and optional
  inputs.
- Keep knowledge drift as review guidance only. It must not become a hard
  blocker or a required missing artifact.
- Do not modify workflow, bundle, or companion exporters.

## Implemented Behavior

- Added CLI flag `--benchmark-knowledge-drift` to both release exporters.
- Added optional `benchmark_knowledge_drift` input to:
  - `build_release_decision(...)`
  - `build_release_runbook(...)`
- Added artifact row `benchmark_knowledge_drift` to both payloads.
- Added stable knowledge drift payload fields:
  - `knowledge_drift_status`
  - `knowledge_drift_summary`
  - `knowledge_drift`
- Normalized `knowledge_drift` includes:
  - `status`
  - `summary`
  - `regressions`
  - `improvements`
  - `new_focus_areas`
  - `resolved_focus_areas`
  - `counts`
  - `recommendations`
  - `has_drift`
- Release decision now surfaces `component_statuses.knowledge_drift` but keeps
  drift out of hard blocker evaluation.
- Knowledge drift only contributes review signals for cautionary states such as
  `baseline_missing`, `regressed`, `mixed`, or stable drift with new focus
  areas.
- Improved-only drift remains informational and does not force
  `review_required`.
- Runbook excludes `benchmark_knowledge_drift` from required missing artifacts,
  so missing drift input cannot block `next_action` or
  `ready_to_freeze_baseline` by itself.

## Files

- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_runbook.py`
- `docs/BENCHMARK_KNOWLEDGE_DRIFT_RELEASE_VALIDATION_20260308.md`

## Validation Commands

```bash
python3 -m py_compile \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

pytest tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py -v
```

## Validation Results

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `8 passed` in `1.42s`
- Warnings: one existing `python_multipart` `PendingDeprecationWarning` from
  `starlette`; no release decision/runbook failures

## Verified Scenarios

- Decision payload exposes knowledge drift status, summary, counts, and artifact
  presence.
- Regressed or mixed knowledge drift stays review-only and does not override a
  stronger upstream blocker.
- Improved-only knowledge drift preserves a `ready` release decision and a
  freezeable runbook.
- Runbook Markdown renders a dedicated `## Knowledge Drift` section.
- Missing knowledge drift input does not become a required artifact gap.
