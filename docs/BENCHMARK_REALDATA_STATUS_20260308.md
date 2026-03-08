# Benchmark Real-Data Status

## Scope

This report summarizes the current real-data benchmark readiness using the
public example sets already downloaded locally:

- `HPSketch` `.h5` example data
- `foxtrot` STEP example data

## Inputs

- HPSketch sample:
  `/private/tmp/cad-ai-example-data-20260307/HPSketch/data/0000/00000007_1.h5`
- foxtrot STEP directory:
  `/private/tmp/cad-ai-example-data-20260307/foxtrot/examples`

## Results

### 1. History Sequence `.h5`

The current history-sequence path runs successfully on the base Python
environment.

- `status`: `ok`
- `label`: `轴类`
- `confidence`: `0.5069`
- `top2_confidence`: `0.4931`
- `source`: `history_sequence_prototype`
- `sequence_length`: `5`
- `vec_shape`: `(5, 21)`

This confirms that the benchmark stack can consume real `.h5` example inputs
without requiring the OCC / STEP toolchain.

### 2. STEP Single-File Validation

The base Python environment still reports:

- `status`: `skipped_no_occ`

This is expected on the default interpreter. The single-file validation script
is still useful as an environment readiness check.

### 3. STEP Directory Validation Under Micromamba

Using the Apple Silicon micromamba environment with `pythonocc-core`, the STEP
batch evaluation succeeds.

- `sample_size`: `3`
- `status_counts.ok`: `3`
- `shape_loaded_count`: `3`
- `valid_3d_count`: `3`
- `hint_coverage_count`: `2`
- `assembly_count`: `1`
- `avg_faces_ok`: `7.3333`
- `avg_nodes_ok`: `7.3333`
- `avg_edges_ok`: `26.6667`
- `graph_schema_version_counts.v2`: `3`

## Interpretation

- Real `.h5` validation is already benchmark-ready on the base environment.
- Real STEP validation is benchmark-ready on the micromamba/OCC path.
- The remaining gap is not feature extraction availability; it is consistent
  operator packaging so the same 3D path can be invoked in CI or staging
  without local environment drift.

## Generated Artifacts

- `reports/experiments/20260308/online_example_ai_inputs_validation.json`
- `reports/experiments/20260308/brep_step_dir_eval_foxtrot/summary.json`

These local reports were intentionally not added to git. This document records
their benchmark implications instead.
