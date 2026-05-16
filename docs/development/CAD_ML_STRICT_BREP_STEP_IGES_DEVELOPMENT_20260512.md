# CAD ML Strict STEP/IGES B-Rep Development

Date: 2026-05-12

## Goal

Move Phase 4 B-Rep evidence from smoke-level parsing toward strict benchmark
evidence. The slice focuses on evaluator behavior, not on building the full
50-100 file golden manifest.

## Changes

- Updated `src/core/geometry/engine.py`.
  - Added `GeometryEngine.load_iges(...)`.
  - Uses OpenCascade `IGESControl_Reader` when the installed pythonocc build exposes it.
  - Keeps STEP loading behavior unchanged.
- Updated `scripts/eval_brep_step_dir.py`.
  - Added default IGES discovery patterns: `*.iges`, `*.igs`, `*.IGES`, `*.IGS`.
  - Added `--strict`.
  - Added `--allow-demo-geometry`.
  - Added row-level fields:
    - `file_format`
    - `evaluation_mode`
    - `failure_reason`
    - `parse_success`
    - `graph_valid`
    - `synthetic_geometry`
    - `demo_geometry_allowed`
    - `extraction_latency_ms`
  - Added strict failure reasons:
    - `step_parse_failed`
    - `iges_parse_failed`
    - `iges_loader_missing`
    - `unsupported_file_format`
    - `synthetic_geometry_not_allowed`
    - `brep_features_invalid`
    - `brep_faces_missing`
    - `brep_graph_invalid`
    - `exception`
  - Added summary fields:
    - `parse_success_count`
    - `graph_valid_count`
    - `failure_reason_counts`
    - `face_count_total`
    - `edge_count_total`
    - `solid_count_total`
    - `surface_type_histogram`
    - `avg_extraction_latency_ms`
    - `max_extraction_latency_ms`
    - `strict_mode`
    - `demo_geometry_allowed`
  - Added `graph_qa.json` beside `results.csv` and `summary.json`.
- Updated `tests/unit/test_eval_brep_step_dir.py`.
  - Covered strict invalid B-Rep rows.
  - Covered synthetic/demo geometry rejection and explicit allow mode.
  - Covered IGES loader routing.
  - Covered IGES unsupported-loader behavior.
  - Covered graph QA report output.
- Updated `docs/development/CAD_ML_DETAILED_DEVELOPMENT_TODO_20260512.md`.

## Usage

Strict real-data run:

```bash
python scripts/eval_brep_step_dir.py \
  --step-dir /path/to/step_iges_set \
  --strict \
  --output-dir reports/experiments/20260512/brep_step_dir_eval
```

Demo-only run that allows explicitly marked synthetic/demo geometry:

```bash
python scripts/eval_brep_step_dir.py \
  --step-dir /path/to/demo_set \
  --strict \
  --allow-demo-geometry \
  --output-dir reports/experiments/20260512/brep_step_dir_demo_eval
```

## Outputs

- `results.csv`: one row per STEP/IGES file.
- `summary.json`: aggregate release-scorecard compatible metrics.
- `graph_qa.json`: graph extraction QA status and invalid graph row list.

## Remaining Phase 4 Work

- Build the 50-100 file STEP/IGES golden manifest.
- Add manifest-level metadata: source, license, part family, expected topology range,
  and expected failure behavior.
- Run strict evaluation on the real manifest in an OCC-enabled CI/runtime and feed the
  generated `summary.json` into the forward scorecard.
