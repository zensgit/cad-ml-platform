# DEV Diagnose Graph2D Order Stabilization (2026-02-20)

## Background
- CI previously showed a flaky failure in:
  - `tests/unit/test_diagnose_graph2d_manifest_truth.py::test_diagnose_graph2d_supports_manifest_truth_mode`
- Failure symptom:
  - `predictions.csv` row order intermittently became `["b.dxf", "a.dxf"]` instead of `["a.dxf", "b.dxf"]`.

## Root Cause
- `scripts/diagnose_graph2d_on_dxf_dir.py` used unconditional shuffle before slicing:
  - Always called `random.shuffle(files)`, even when `max_files >= len(files)`.
- File collection order (`rglob`) can differ by environment, so this introduced nondeterministic CSV ordering.

## Implemented Fix
- File: `scripts/diagnose_graph2d_on_dxf_dir.py`
1. Sort files before any sampling.
2. Only sample when `max_files < len(files)`.
3. Re-sort sampled files before inference/output.

## Test Hardening
- File: `tests/unit/test_diagnose_graph2d_manifest_truth.py`
1. Stabilized traversal in the existing manifest-truth test by monkeypatching `_collect_dxfs` to a fixed order.
2. Added a new regression test:
   - `test_diagnose_graph2d_manifest_truth_sorted_after_sampling`
   - Verifies sampled outputs remain sorted by `relative_path`.

## Local Verification
- Command:
  - `pytest -q tests/unit/test_diagnose_graph2d_manifest_truth.py -q`
- Result: passed.

## CI Verification (commit: `8d92d83`)
- `CI Tiered Tests`: success  
  `https://github.com/zensgit/cad-ml-platform/actions/runs/22209271632`
- `CI`: success  
  `https://github.com/zensgit/cad-ml-platform/actions/runs/22209271642`
- `CI Enhanced`: success  
  `https://github.com/zensgit/cad-ml-platform/actions/runs/22209271637`
- `Code Quality`: success  
  `https://github.com/zensgit/cad-ml-platform/actions/runs/22209271645`
- `Multi-Architecture Docker Build`: success  
  `https://github.com/zensgit/cad-ml-platform/actions/runs/22209271646`

## Outcome
- The manifest-truth diagnose path now has deterministic output ordering.
- Prior flaky assertion condition has been removed by both runtime fix and stronger regression coverage.
