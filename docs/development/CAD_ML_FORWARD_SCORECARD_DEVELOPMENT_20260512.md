# CAD ML Forward Scorecard Development

Date: 2026-05-12

## Goal

Add a single forward-looking scorecard that answers whether the platform is
release-ready, benchmark-ready with gaps, shadow-only, or blocked.

The scorecard is deliberately stricter than individual demos. It prevents
fallback-only model branches or missing benchmark artifacts from being described
as production readiness.

## Implemented Files

### `src/core/benchmark/forward_scorecard.py`

Added reusable scorecard helpers:

- `build_forward_scorecard(...)`
- `render_forward_scorecard_markdown(...)`

Canonical status values:

- `release_ready`
- `benchmark_ready_with_gap`
- `shadow_only`
- `blocked`

The scorecard aggregates:

- model readiness registry state;
- Hybrid DXF coarse accuracy, exact accuracy, macro F1, and low-confidence rate;
- Graph2D blind accuracy and low-confidence rate;
- History Sequence coarse/exact accuracy, macro F1, and low-confidence rate;
- B-Rep parse success, valid 3D count, graph validity, and failure reasons;
- Qdrant/vector readiness, indexed ratio, unindexed count, scan truncation, and hints;
- active-learning review queue critical/high backlog and evidence coverage;
- knowledge readiness coverage and reference count.

Overall release rules:

- `release_ready` requires every component to be release-ready.
- Fallback model branches prevent `release_ready`.
- Missing Hybrid DXF benchmark evidence blocks the overall scorecard.
- B-Rep/3D remains a separate component so strong 2D results cannot hide weak
  STEP/IGES evidence.
- Release claims must cite the scorecard artifact.

### `scripts/export_forward_scorecard.py`

Added CLI exporter.

Default outputs:

```text
reports/benchmark/forward_scorecard/latest.json
reports/benchmark/forward_scorecard/latest.md
```

Optional inputs:

- `--model-readiness-summary`
- `--hybrid-summary`
- `--graph2d-summary`
- `--history-summary`
- `--brep-summary`
- `--qdrant-summary`
- `--review-queue-summary`
- `--knowledge-summary`

If no model-readiness summary is provided, the script uses the live
`build_model_readiness_snapshot()` registry.

### `src/core/benchmark/__init__.py`

Exported the forward scorecard helpers for reuse by scripts and tests.

### `tests/unit/test_forward_scorecard.py`

Added coverage for:

- fully ready scorecard producing `release_ready`;
- fallback model branch preventing `release_ready`;
- missing Hybrid DXF evidence blocking release claims;
- Markdown component table rendering;
- CLI exporter writing JSON and Markdown outputs.

### Generated Reports

Generated the current local checkout scorecard:

```text
reports/benchmark/forward_scorecard/latest.json
reports/benchmark/forward_scorecard/latest.md
```

The current local default output is `blocked` because no benchmark artifact
inputs were supplied for Hybrid DXF, Graph2D blind, History Sequence, B-Rep,
Qdrant, or knowledge coverage. That is expected and useful: the scorecard is
now refusing to infer readiness from absent evidence.

## Follow-Up Work

- Wire CI/release jobs to pass real benchmark artifact paths into
  `scripts/export_forward_scorecard.py`.
- Add release-label checks that fail when the scorecard is `blocked` or
  `shadow_only`.
- Add a trend exporter once multiple forward scorecard snapshots exist.
