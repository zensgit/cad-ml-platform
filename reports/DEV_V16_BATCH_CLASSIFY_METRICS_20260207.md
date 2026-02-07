# DEV_V16_BATCH_CLASSIFY_METRICS_20260207

## Summary

Added Prometheus counters for the V16 batch classify API so we can track request outcomes (success/partial/failed) and per-file results (success/failed) alongside existing latency and batch-size histograms.

## Why

`POST /api/v1/analyze/batch-classify` processes multiple files concurrently; without explicit counters it is hard to distinguish between:

- a fully successful request vs. partial success (some files rejected/failed), and
- frequent per-file failures masked by a `200` response.

## Changes

- `src/api/v1/analyze.py`
  - Records:
    - `v16_batch_classify_requests_total{status="success|partial|failed"}`
    - `v16_batch_classify_files_total{result="success|failed"}`
  - Also observes existing histograms:
    - `v16_classifier_batch_seconds`
    - `v16_classifier_batch_size`
  - Metrics are best-effort (guarded by `try/except`) so environments with metrics disabled do not fail requests.

- `src/utils/analysis_metrics.py`
  - Added:
    - `v16_batch_classify_requests_total`
    - `v16_batch_classify_files_total`

## Validation

Executed:

```bash
python3 -m pytest -q tests/unit/test_v16_classifier_endpoints.py::TestBatchClassifyEndpoint
python3 -c "import src.utils.analysis_metrics as m; print(bool(m.v16_batch_classify_requests_total), bool(m.v16_batch_classify_files_total))"
```

Result: PASS

