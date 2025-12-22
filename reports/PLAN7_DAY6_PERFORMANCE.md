# Day 6 - Performance and Capacity

Date: 2025-12-21

## Scope

- Add a simple render latency benchmark.
- Collect baseline timing for DWG render service.

## Changes

- Added benchmark script:
  - `cad-ml-platform/scripts/benchmark_cad_render.py`
- Documented in:
  - `cad-ml-platform/docs/CAD_RENDER_SERVER.md`

## Verification

- Command:
  - `python scripts/benchmark_cad_render.py --file <dwg> --requests 3 --concurrency 1 --token test-token`
- Result:
  - status_counts: `{200: 3}`
  - min: `1.847s`
  - max: `2.386s`
  - mean: `2.035s`
  - p50: `1.872s`
  - p95: `1.872s`
