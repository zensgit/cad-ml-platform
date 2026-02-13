# DEV_DXF_GEOMETRY_ONLY_GRAPH2D_BASELINE_20260213

## Summary

Added a **geometry-only DXF evaluation mode** to quantify Graph2D performance when **no filename or
embedded titleblock text** is available.

This is useful because many datasets include strong textual supervision (filename, titleblock
part-name text) which can mask the true geometry capability of Graph2D.

## Changes

- `src/utils/dxf_io.py`
  - Added `strip_dxf_text_entities_from_bytes()` to remove annotation entities (`TEXT`, `MTEXT`,
    `DIMENSION`, `ATTRIB`, `ATTDEF`) from **modelspace and block definitions**.
  - Added `write_dxf_document_to_bytes()` and `strip_dxf_entities_from_bytes()` helpers.

- `scripts/batch_analyze_dxf_local.py`
  - Added `--strip-text` to upload sanitized DXF bytes without text/annotation entities.
  - Added `--geometry-only` which implies `--mask-filename` + `--strip-text` and disables Hybrid
    text branches via env vars (`TITLEBLOCK_ENABLED=false`, `PROCESS_FEATURES_ENABLED=false`,
    `FILENAME_CLASSIFIER_ENABLED=false`).
  - Added summary flags: `strip_text`, `geometry_only`.

## Validation

Executed:

```bash
.venv/bin/python -m pytest tests/unit/test_dxf_io.py -v
.venv/bin/python -m py_compile scripts/batch_analyze_dxf_local.py src/utils/dxf_io.py
make validate-core-fast

.venv/bin/python scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --max-files 30 \
  --geometry-only \
  --output-dir /tmp/dxf_batch_eval_geometry_only_20260213
```

Key results (`/tmp/dxf_batch_eval_geometry_only_20260213/summary.json`):

- `weak_labels.covered_rate = 1.0` (`30/30`)
- Titleblock effectively removed:
  - `titleblock.texts_present_rate = 0.0`
  - `weak_labels.accuracy.titleblock_label`: evaluated `0`, missing `30`
- Graph2D baseline (geometry-only):
  - `weak_labels.accuracy.graph2d_label.accuracy = 0.2` (`6/30`)
  - `graph2d.confidence.mean_all ~= 0.059` (low-confidence many-class softmax)
- Hybrid becomes mostly fallback (as expected) because text branches are disabled and
  graph2d is frequently filtered by minimum confidence:
  - `hybrid.source_counts = {fallback: 23, graph2d: 7}`

## Notes / Caveats

- Weak labels are derived from **original filenames** via `FilenameClassifier` and are not ground
  truth; treat them as regression indicators.
- This mode is intentionally harsh: it removes titleblock text from `INSERT` blocks too, so it is
  a closer proxy for geometry-only inference.
