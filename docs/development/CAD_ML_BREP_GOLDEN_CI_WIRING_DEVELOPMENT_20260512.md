# CAD ML B-Rep Golden CI Wiring Development

Date: 2026-05-12

## Goal

Move the Phase 4 B-Rep golden manifest from a local contract into an optional
CI/release control path. The default workflow remains non-blocking for ordinary PRs,
but release runs can now require a real manifest and publish validation/evaluation
evidence.

## Changes

- Added `scripts/ci/build_brep_golden_manifest_optional.sh`.
  - Reads workflow input aliases from `GITHUB_EVENT_INPUTS_JSON`.
  - Uses `BREP_GOLDEN_MANIFEST_ENABLE=false` as the default skip mode.
  - Validates `BREP_GOLDEN_MANIFEST_JSON` with
    `scripts/validate_brep_golden_manifest.py`.
  - Emits GitHub outputs for:
    - `validation_json`
    - `validation_status`
    - `ready_for_release`
    - `case_count`
    - `release_eligible_count`
    - `min_release_samples`
  - Honors `BREP_GOLDEN_MANIFEST_FAIL_ON_NOT_RELEASE_READY=true` so insufficient
    or invalid manifests can stop a release run.
  - Optionally runs `scripts/eval_brep_step_dir.py --manifest` when
    `BREP_GOLDEN_EVAL_ENABLE=true`.
  - Emits strict evaluator outputs for:
    - `eval_summary_json`
    - `eval_results_csv`
    - `eval_graph_qa_json`
- Updated `.github/workflows/evaluation-report.yml`.
  - Added `BREP_GOLDEN_*` environment variables.
  - Added `Build B-Rep golden manifest gate (optional)` before the forward
    scorecard step.
  - Added artifact uploads for manifest validation and strict B-Rep evaluation.
- Updated `scripts/ci/build_forward_scorecard_optional.sh`.
  - Reads `steps.brep_golden_manifest.outputs.eval_summary_json`.
  - Feeds the manifest-driven strict B-Rep summary into the forward scorecard as
    `--brep-summary`.
- Added `tests/unit/test_brep_golden_manifest_ci_wrapper.py`.
  - Covers disabled skip behavior.
  - Covers non-release example manifest reporting.
  - Covers failure when `FAIL_ON_NOT_RELEASE_READY=true`.
  - Covers a release-ready temp manifest with a reduced sample floor.
- Extended `tests/unit/test_forward_scorecard_release_gate.py`.
  - Covers forward scorecard consumption of the B-Rep golden step output.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Operating Model

Default PR behavior:

```text
BREP_GOLDEN_MANIFEST_ENABLE=false
BREP_GOLDEN_EVAL_ENABLE=false
```

Result: the step writes `enabled=false` and exits successfully.

Validation-only CI behavior:

```text
BREP_GOLDEN_MANIFEST_ENABLE=true
BREP_GOLDEN_MANIFEST_FAIL_ON_NOT_RELEASE_READY=false
```

Result: CI publishes the manifest validation report, including insufficient-sample
status, without blocking ordinary evidence collection.

Release gate behavior:

```text
BREP_GOLDEN_MANIFEST_ENABLE=true
BREP_GOLDEN_MANIFEST_FAIL_ON_NOT_RELEASE_READY=true
```

Result: invalid or insufficient manifests fail the run.

Strict evaluation behavior:

```text
BREP_GOLDEN_EVAL_ENABLE=true
BREP_GOLDEN_EVAL_STRICT=true
BREP_GOLDEN_EVAL_ALLOW_DEMO_GEOMETRY=false
```

Result: CI evaluates the manifest samples, publishes `summary.json`, `results.csv`,
and `graph_qa.json`, then lets the forward scorecard consume `summary.json`.

## Remaining Work

- Populate a private or tracked manifest with 50-100 licensed real STEP/IGES files.
- Run `BREP_GOLDEN_EVAL_ENABLE=true` only on a runner with `pythonocc-core`
  available.
- Decide whether the manifest validation should become required for all PRs after the
  real golden set exists.
