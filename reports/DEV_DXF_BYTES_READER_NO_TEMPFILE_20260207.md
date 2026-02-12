# DEV_DXF_BYTES_READER_NO_TEMPFILE_20260207

## Goal

Remove temp-file DXF parsing when callers already have DXF bytes (FastAPI uploads/TestClient), to reduce I/O overhead and avoid noisy cache/path issues in constrained environments.

## Changes

### 1) New DXF bytes reader utility

- Added `src/utils/dxf_io.py`:
  - `guess_dxf_encoding(data: bytes) -> str`
    - Uses DXF header keys `$ACADVER` / `$DWGCODEPAGE` (best-effort).
    - Matches ezdxf guidance: R2007+ (`AC1021+`) uses UTF-8; older uses codepage mapping.
  - `read_dxf_document_from_bytes(data: bytes)`
  - `read_dxf_entities_from_bytes(data: bytes)`

### 2) Refactors to stop writing temp DXF files

- `src/ml/hybrid_classifier.py`: DXF parsing for TitleBlock/Process now uses `read_dxf_entities_from_bytes(file_bytes)` (no `NamedTemporaryFile`).
- `src/ml/vision_2d.py`: `Graph2DClassifier.predict_from_bytes()` now uses `read_dxf_document_from_bytes(data)` (no temp file).
- `src/adapters/factory.py`: `DxfAdapter.parse()` now uses `read_dxf_document_from_bytes(data)` (no temp file).

### 3) Local batch script hygiene

- `scripts/batch_analyze_dxf_local.py`:
  - Sets `XDG_CACHE_HOME=/tmp/xdg-cache` by default so ezdxf cache writes don’t target `$HOME` (helpful for sandboxed runs).

## Verification

### Tests

- `.venv/bin/pytest -q tests/unit/test_dxf_io.py`
  - `3 passed`
- `.venv/bin/pytest -q tests/unit/test_filename_classifier.py tests/unit/test_hybrid_classifier.py tests/unit/test_titleblock_extractor.py`
  - `44 passed`
- `.venv/bin/pytest -q tests/integration/test_analyze_dxf_hybrid_override.py`
  - `3 passed`

### Local DXF Batch (Training corpus, 110 files)

- Ran:
  - `GRAPH2D_ENABLED=false GRAPH2D_FUSION_ENABLED=false FUSION_ANALYZER_ENABLED=false HYBRID_CLASSIFIER_ENABLED=true HYBRID_CLASSIFIER_AUTO_OVERRIDE=true .venv/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" --output-dir reports/experiments/20260207/batch_analyze_dxf_local_hybrid_v4 --seed 2207 --min-confidence 0.8`
- Results:
  - `total=110`, `success=110`, `error=0`
  - `low_confidence_count=0`
  - Summary: `reports/experiments/20260207/batch_analyze_dxf_local_hybrid_v4/summary.json`

## Notes / Limits

- This change focuses on DXF parsing from bytes. Torch-backed Graph2D inference remains optional (not installed in this `.venv`), but the byte-parsing path is now ready for environments with torch.

