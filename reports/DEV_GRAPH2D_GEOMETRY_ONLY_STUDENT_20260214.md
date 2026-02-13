# DEV_GRAPH2D_GEOMETRY_ONLY_STUDENT_20260214

## Goal

Add a geometry-only training/evaluation option for Graph2D so the student model can be trained/evaluated without DXF text/annotation entities (TEXT/MTEXT/DIMENSION/ATTRIB), preventing accidental label leakage from text into the graph features.

This is intended to support later distillation experiments where the teacher can use text (titleblock/process) while the student learns from geometry-only graphs.

## Changes

File: `src/ml/train/dataset_2d.py`

- `DXFManifestDataset.__getitem__` now supports an opt-in mode:
  - Env: `DXF_STRIP_TEXT_ENTITIES=true`
  - Behavior: read DXF bytes -> strip text entities (including blocks) -> parse -> build graph
- `DXFManifestDataset._graph_cache_key(...)` now includes `DXF_STRIP_TEXT_ENTITIES` so cached graphs cannot be reused across strip/non-strip modes.

### Unit Tests

File: `tests/unit/test_dxf_manifest_dataset_strip_text_entities.py`

- Verifies that enabling `DXF_STRIP_TEXT_ENTITIES=true` removes TEXT nodes from the built graph.
- Verifies cache key differs when toggling the strip flag.

## Validation

### Static Checks

- `.venv/bin/python -m py_compile src/ml/train/dataset_2d.py` (passed)

### Unit Tests

- `.venv/bin/pytest tests/unit/test_dxf_manifest_dataset_strip_text_entities.py -q` (passed)

### Pipeline Smoke (Training Drawings)

Command (geometry-only enabled):

```bash
/usr/bin/time -p env DXF_STRIP_TEXT_ENTITIES=true .venv/bin/python scripts/run_graph2d_pipeline_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --epochs 1 \
  --max-samples 40 \
  --diagnose-max-files 20 \
  --model edge_sage \
  --loss cross_entropy \
  --class-weighting inverse \
  --sampler balanced \
  --graph-cache both \
  --empty-edge-fallback knn \
  --empty-edge-knn-k 8
```

Observed:

- Work dir: `/tmp/graph2d_pipeline_local_20260214_012008`
- Completed successfully (artifacts in `/tmp`)
- Timing: `real 24.94s`

## Notes / Caveats

- Geometry-only mode is slower than `ezdxf.readfile(...)` because it strips entities via a bytes round-trip; use `--graph-cache both` to amortize repeated parsing.
- DXF graphs built from INSERT-only titleblocks do not include block TEXT nodes (the graph builder reads modelspace only). This mode primarily removes modelspace text/annotation entities.

