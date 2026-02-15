# DEV_GRAPH2D_SEED_GATE_TREND_LOW_CONF_GUARD_20260215

## Goal

Execute `1+2+3` in one iteration:

1. add `strict_top_pred_ratio` trend visualization in CI summary/artifacts,
2. add a low-confidence ratio gate,
3. run both seed gates and write a new baseline snapshot.

## Implementation

### A) CI trend visualization

- Added `scripts/ci/render_graph2d_seed_gate_trend.py`.
- Per-seed markdown trend table now includes:
  - `strict_accuracy`
  - `strict_top_pred_ratio`
  - `strict_low_conf_ratio`
  - ASCII trend bars (`[####....]`) for top-pred and low-conf ratios.
- Updated `.github/workflows/ci.yml`:
  - generate trend markdown for standard gate and append to `GITHUB_STEP_SUMMARY`,
  - upload trend markdown artifact,
  - same for strict gate optional channel.

### B) Low-confidence ratio gate

Updated `scripts/sweep_graph2d_profile_seeds.py`:

- New per-run metric extraction from `diagnose/predictions.csv`:
  - `strict_low_conf_threshold`
  - `strict_low_conf_count`
  - `strict_low_conf_total`
  - `strict_low_conf_ratio`
- New CLI options:
  - `--strict-low-confidence-threshold`
  - `--max-strict-low-conf-ratio`
- Gate now fails when any seed exceeds `max_strict_low_conf_ratio`.
- New summary metrics:
  - `strict_low_conf_threshold`
  - `strict_low_conf_ratio_mean`
  - `strict_low_conf_ratio_max`

Updated configs:

- `config/graph2d_seed_gate.yaml`
  - `strict_low_confidence_threshold: 0.20`
  - `max_strict_low_conf_ratio: 0.20`
- `config/graph2d_seed_gate_strict.yaml`
  - `strict_low_confidence_threshold: 0.20`
  - `max_strict_low_conf_ratio: 0.20`

Updated CI summary renderer:

- `scripts/ci/summarize_graph2d_seed_gate.py`
  - added row: `Low-conf ratio < <threshold> (mean/max)`.

## Tests

### Unit tests

```bash
pytest tests/unit/test_sweep_graph2d_profile_seeds.py \
       tests/unit/test_graph2d_seed_gate_summary.py \
       tests/unit/test_graph2d_seed_gate_trend.py -q
```

Result: `14 passed`.

## Runtime verification

### Standard channel

```bash
make validate-graph2d-seed-gate
```

Summary (`/tmp/graph2d-seed-gate/seed_sweep_summary.json`):

- strict accuracy mean/min/max: `0.362500 / 0.291667 / 0.433333`
- top-pred ratio mean/max: `0.600000 / 0.708333`
- low-conf ratio (<0.2) mean/max: `0.050000 / 0.050000`
- distinct labels min/max: `5 / 5`
- gate: passed

### Strict channel

```bash
make validate-graph2d-seed-gate-strict
```

Summary (`/tmp/graph2d-seed-gate-strict/seed_sweep_summary.json`):

- strict accuracy mean/min/max: `0.945833 / 0.941667 / 0.950000`
- top-pred ratio mean/max: `0.258333 / 0.275000`
- low-conf ratio (<0.2) mean/max: `0.050000 / 0.050000`
- distinct labels min/max: `5 / 5`
- gate: passed

### Trend rendering check

```bash
.venv/bin/python scripts/ci/render_graph2d_seed_gate_trend.py \
  --summary-json /tmp/graph2d-seed-gate/seed_sweep_summary.json \
  --title "Graph2D Seed Gate Local" \
  --output-md /tmp/graph2d-seed-gate/seed_gate_trend.md
```

```bash
.venv/bin/python scripts/ci/render_graph2d_seed_gate_trend.py \
  --summary-json /tmp/graph2d-seed-gate-strict/seed_sweep_summary.json \
  --title "Graph2D Strict Seed Gate Local" \
  --output-md /tmp/graph2d-seed-gate-strict/seed_gate_trend.md
```

Outputs confirmed with per-seed trend rows.

## Baseline snapshot

- Added snapshot: `reports/experiments/20260215/graph2d_seed_gate_baseline_snapshot_20260215.json`

