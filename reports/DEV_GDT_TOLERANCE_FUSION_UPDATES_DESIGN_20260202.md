# DEV_GDT_TOLERANCE_FUSION_UPDATES_DESIGN_20260202

## Summary
Targeted reliability improvements across GD&T parsing, tolerance-fit calculations, and benchmark stability.

## Scope
1. GD&T feature control frame parsing stability.
2. Fusion analyzer override configuration refresh.
3. Tolerance fit calculation clarity for hole-basis support (H/JS + selected non-H).
4. Benchmark test stability adjustments.

## Design Decisions
### 1) GD&T FeatureControlFrame defaults
- **Problem**: `FeatureControlFrame.notes` defaulted to `None`, risking `NoneType` errors when appending notes.
- **Decision**: Use `field(default_factory=list)` for safe list initialization.
- **Impact**: Eliminates runtime errors without changing public API.

### 2) FusionAnalyzer environment refresh
- **Problem**: Graph2D override settings were only loaded at singleton initialization; env updates during runtime/tests were ignored.
- **Decision**: Add `refresh_from_env()` and allow `get_fusion_analyzer(reset=False, refresh_env=False)` to re-read env vars.
- **Impact**: Enables deterministic tests and on-demand config refresh without global reload.

### 3) Tolerance fits support boundaries
- **Problem**: `get_fit_deviations` implied broader hole/shaft support, but only H-basis fits were effectively computed.
- **Decision**:
  - Explicitly support H/JS hole symbols and a subset of non-H holes where fundamental deviations are available.
  - Load `HOLE_FUNDAMENTAL_DEVIATIONS` from ISO 286 override file and apply them as EI for holes.
  - Add representative non-H fit codes (`D10/h9`, `N9/h9`, `P9/h9`) to `COMMON_FITS` for shaft-basis/keyway scenarios.
  - Allow optional overrides via `data/knowledge/iso286_hole_deviations.json` (env: `HOLE_DEVIATIONS_PATH`).
- **Impact**: Prevents silent failures and extends fit computations to common shaft-basis cases while remaining explicit about unsupported symbols.
  - Current override file covers symbols: E, F, G, H, J, JS, K, M, N, P, R.
  - C/D are derived from shaft fundamental deviations when ISO table entries are missing.

### 4) Benchmark stability
- **Problem**: Concurrency and client transport usage were flaky and incompatible with httpx 0.28.
- **Decision**:
  - Use `httpx.AsyncClient` + `ASGITransport` with `pytest.mark.anyio` for async benchmark requests.
  - Add a warm-up call before latency assertion to avoid cold-start skew.
  - Replace GC object-count check with `tracemalloc` growth guard.
- **Impact**: Benchmarks are now compatible with httpx 0.28 and provide stable, reproducible measurements.

## Files Updated
- `src/core/knowledge/gdt/application.py`
- `src/core/knowledge/fusion_analyzer.py`
- `src/core/knowledge/tolerance/fits.py`
- `tests/benchmarks/test_performance.py`
- `tests/test_gdt_application.py`
- `tests/test_tolerance_fits.py`
- `data/knowledge/iso286_hole_deviations.json`
- `requirements.txt`
- `scripts/build_iso286_hole_deviations.py`
- `scripts/validate_iso286_hole_deviations.py`
