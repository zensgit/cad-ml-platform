# B-Rep STEP Directory Evaluation Validation 2026-03-07

## Summary

Added a directory-level STEP/B-Rep evaluation script:

- `scripts/eval_brep_step_dir.py`

The script scans a STEP directory, loads each shape through `GeometryEngine`,
extracts B-Rep features plus graph metadata, and writes:

- `results.csv`
- `summary.json`

It is intended to give a fast, repeatable smoke and diagnostics path for real
STEP corpora on machines that already have `pythonocc-core` available.

## Why this was added

The existing validation path proved that a single online example STEP file could
be loaded successfully on macOS ARM64 after micromamba setup. The next missing
piece was directory-level evaluation so the same environment could be used to:

1. batch-check multiple STEP samples
2. inspect graph schema output consistency
3. aggregate feature-hint and surface-type distributions

## Files

- `scripts/eval_brep_step_dir.py`
- `tests/unit/test_eval_brep_step_dir.py`

## Validation

Static and unit validation:

```bash
python3 -m py_compile scripts/eval_brep_step_dir.py \
  tests/unit/test_eval_brep_step_dir.py

flake8 scripts/eval_brep_step_dir.py \
  tests/unit/test_eval_brep_step_dir.py \
  --max-line-length=100

pytest -q tests/unit/test_eval_brep_step_dir.py
```

Observed result:

- `5 passed`

## Real STEP directory run

Executed with the local Apple Silicon micromamba environment:

```bash
~/.local/bin/micromamba run -r ~/.micromamba -n cad-ml-brep-m4 \
  python scripts/eval_brep_step_dir.py \
  --step-dir /private/tmp/cad-ai-example-data-20260307/foxtrot/examples \
  --output-dir reports/experiments/20260307/brep_step_dir_eval_foxtrot
```

Generated outputs:

- `reports/experiments/20260307/brep_step_dir_eval_foxtrot/results.csv`
- `reports/experiments/20260307/brep_step_dir_eval_foxtrot/summary.json`

Observed summary:

- `sample_size=3`
- `status_counts.ok=3`
- `shape_loaded_count=3`
- `valid_3d_count=3`
- `hint_coverage_count=2`
- `assembly_count=1`
- `avg_faces_ok=7.3333`
- `avg_nodes_ok=7.3333`
- `avg_edges_ok=26.6667`
- `graph_schema_version_counts.v2=3`
- `primary_surface_type_counts.plane=3`
- `top_hint_label_counts.block=2`

Observed per-file highlights:

- `cube_hole.step`
  - `faces=7`
  - `node_count=7`
  - `edge_count=28`
  - `top_hint_label=block`
- `cuboid.step`
  - `faces=6`
  - `node_count=6`
  - `edge_count=24`
  - `top_hint_label=block`
- `abstract_pca.step`
  - `solids=2`
  - `is_assembly=true`
  - `graph_schema_version=v2`
  - no stable feature hint emitted

## Interpretation

- The macOS ARM64 micromamba setup is sufficient not only for single-file smoke
  validation, but also for repeatable directory-level STEP evaluation.
- The current `extract_brep_graph()` path produces stable `v2` graph outputs on
  all three public sample files.
- The current feature-hint heuristic is intentionally conservative:
  plane-dominant solids are tagged as `block`, while more ambiguous assemblies
  may emit no hint.

## Current limitations

- The evaluation depends on `pythonocc-core`; outside the micromamba B-Rep
  environment the script will fail fast with a clear error.
- `prepare_brep_features_for_report()` can emit no hint for complex samples,
  which is expected today but should be improved later if stronger 3D semantic
  hints are needed.
