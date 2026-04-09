# Benchmark Surpass Bundle Progress 2026-03-08

## Goal

This batch pushes the benchmark-surpass track from standalone artifacts toward a
CI-visible, reviewer-visible operating surface.

The target is not another isolated model metric. The target is a benchmark
delivery plane that lets operators and reviewers answer these questions quickly:

- What is the current benchmark status?
- Which benchmark sub-systems are healthy or lagging?
- Which artifacts are present and reusable?
- What should be reviewed or fixed first?

## Design

The benchmark delivery plane is organized in layers:

1. `benchmark scorecard`
   - compact cross-capability status
2. `feedback flywheel benchmark`
   - feedback / finetune / metric-training loop health
3. `benchmark operational summary`
   - operator-facing summary across scorecard, flywheel, assistant, review
     queue, and OCR review
4. `benchmark artifact bundle`
   - one reusable manifest for handoff, packaging, and downstream CI/reporting
5. `CI + PR visibility`
   - GitHub job summary and PR comment exposure

This progression is important because it turns benchmark data from raw JSON into
an operating interface.

## Delivered In Main

Already integrated into `main`:

- standalone benchmark scorecard export and CI wiring
- standalone feedback flywheel benchmark export and CI wiring
- standalone benchmark operational summary export
- benchmark operational summary CI summary integration
- benchmark operational summary PR comment surfacing
- standalone benchmark artifact bundle export

Key documents already present:

- `docs/FEEDBACK_FLYWHEEL_BENCHMARK_DELIVERY_20260308.md`
- `docs/BENCHMARK_OPERATIONAL_BUNDLE_DELIVERY_20260308.md`

## Current In-Flight Work

Current stacked PRs:

- `#187` `feat: wire benchmark artifact bundle into ci`
- `#188` `feat: add benchmark artifact bundle pr comment`

### `#187` scope

- workflow-dispatch inputs for bundle sources
- environment variables for bundle enablement and outputs
- optional build step in `evaluation-report.yml`
- artifact upload for bundle JSON/Markdown
- job summary lines for:
  - overall status
  - available artifact count
  - feedback status
  - assistant status
  - review queue status
  - OCR status
  - blockers
  - recommendations

### `#188` scope

- PR comment exposure for bundle status
- signal-light row for bundle health
- compact reviewer-facing bundle status string with artifact availability and
  blockers/recommendations

## Why This Matters For Surpass Strategy

Compared with a generic document-extraction benchmark flow, this bundle track
improves three things:

1. `operator usability`
   - benchmark state is visible without opening multiple JSON artifacts
2. `review velocity`
   - PR comments carry benchmark bundle context directly
3. `governance`
   - one manifest can be reused by CI, handoff, audit, or later dashboards

This is closer to a production benchmark operating model than a one-off model
score dump.

## Validation Snapshot

### `#187`

- `python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100`
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- YAML parse of `.github/workflows/evaluation-report.yml`

Result:

- `3 passed`
- YAML parse succeeded

### `#188`

- `python3 -m py_compile tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- `flake8 tests/unit/test_evaluation_report_workflow_graph2d_extensions.py --max-line-length=100`
- `pytest -q tests/unit/test_evaluation_report_workflow_graph2d_extensions.py`
- YAML parse of `.github/workflows/evaluation-report.yml`

Result:

- `3 passed`
- YAML parse succeeded

## Remaining Gaps

This batch still leaves a few follow-up opportunities:

- optional standalone benchmark bundle artifact CI companion export
- benchmark bundle status roll-up into broader benchmark delivery docs
- possible future dashboard wiring if bundle becomes the benchmark handoff
  contract

These are follow-up improvements, not blockers for the current delivery stack.

## Conclusion

The benchmark-surpass line has moved from:

- isolated artifact generation

to:

- scorecard + flywheel + operational summary + bundle + CI visibility + PR
  visibility

That is a meaningful productization step, not just another metric script.
