# DEV_GRAPH2D_CONTEXT_DRIFT_KEY_COUNTS_SUMMARY_20260216

## Goal

Expose context-drift key frequency in CI summary so warn-channel output is measurable and actionable.

## Implementation

### 1) New renderer script

Added `scripts/ci/render_graph2d_context_drift_key_counts.py`:

- accepts multiple `--report-json` inputs,
- extracts per-report:
  - channel/status/context mode/context match,
  - warning/failure counts,
  - `baseline_metadata.context_diff` keys,
- outputs markdown:
  - per-report table,
  - aggregated `Drift key -> Count` table.

### 2) CI workflow integration

Updated `.github/workflows/ci.yml` (tests job, Python 3.11):

- render step:
  - `Render Graph2D context drift key counts (3.11 only)`
- append to step summary:
  - `Append Graph2D context drift key counts (3.11 only)`
- upload artifact:
  - `graph2d-context-drift-key-counts-ci-<python>.md`

### 3) Tests

Added `tests/unit/test_graph2d_context_drift_key_counts.py`:

- verifies aggregated key counting across reports,
- verifies empty-report handling,
- verifies no-drift handling.

## Validation

```bash
pytest tests/unit/test_graph2d_context_drift_key_counts.py \
       tests/unit/test_graph2d_seed_gate_regression_check.py \
       tests/unit/test_graph2d_seed_gate_regression_summary.py -q
```

Result: `22 passed`.

Local render smoke:

```bash
.venv/bin/python scripts/ci/render_graph2d_context_drift_key_counts.py \
  --report-json /tmp/graph2d-seed-gate-regression-ci-local.json \
  --report-json /tmp/graph2d-context-drift-warn-ci-local.json \
  --title "Graph2D Context Drift Key Counts (Local)" \
  --output-md /tmp/graph2d-context-drift-key-counts-local.md
```

Observed markdown includes:

- per-report status rows,
- aggregated drift key count table (`max_samples -> 1` in smoke sample).
