# DEV_GRAPH2D_SEED_GATE_LABEL_DIVERSITY_GUARD_20260215

## Goal

Continue hardening Graph2D seed-gate so it cannot pass when runs collapse into low label diversity.

## Implementation

### 1) Label-diversity gate in seed sweep

Updated `scripts/sweep_graph2d_profile_seeds.py`:

- Added per-run metric extraction:
  - `manifest_distinct_labels` (from `pipeline_summary.json -> manifest.distinct_labels`)
- Added gate threshold:
  - `--min-manifest-distinct-labels` (default disabled)
- Gate now fails when any run has distinct labels below threshold.
- Summary now includes:
  - `manifest_distinct_labels_min`
  - `manifest_distinct_labels_max`
  - gate metadata field `min_manifest_distinct_labels`.

### 2) Config thresholds updated

- `config/graph2d_seed_gate.yaml`
  - `min_manifest_distinct_labels: 3`
- `config/graph2d_seed_gate_strict.yaml`
  - `min_manifest_distinct_labels: 3`

### 3) CI summary enhancement

Updated `scripts/ci/summarize_graph2d_seed_gate.py`:

- Added table row:
  - `Manifest distinct labels (min/max)`.

## Tests

Updated unit tests:

- `tests/unit/test_sweep_graph2d_profile_seeds.py`
  - added coverage for distinct-label threshold failure.
- `tests/unit/test_graph2d_seed_gate_summary.py`
  - validates summary includes `Manifest distinct labels (min/max)`.

Validation:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_sweep_graph2d_profile_seeds.py \
  tests/unit/test_graph2d_seed_gate_summary.py \
  tests/unit/test_run_graph2d_pipeline_local_profile.py \
  tests/unit/test_run_graph2d_pipeline_local_manifest_wiring.py \
  tests/unit/test_run_graph2d_pipeline_local_distill_wiring.py \
  tests/unit/test_run_graph2d_pipeline_local_diagnose_strict_wiring.py -q
```

Result: `19 passed`.

## Runtime Verification

### A) Standard channel

```bash
make validate-graph2d-seed-gate
```

Artifacts:

- `/tmp/graph2d-seed-gate/seed_sweep_summary.json`

Result:

- strict mean/min/max: `0.362500 / 0.291667 / 0.433333`
- manifest distinct labels min/max: `5 / 5`
- gate: `passed=true`

### B) Strict channel

```bash
make validate-graph2d-seed-gate-strict
```

Artifacts:

- `/tmp/graph2d-seed-gate-strict/seed_sweep_summary.json`

Result:

- strict mean/min/max: `0.945833 / 0.941667 / 0.950000`
- manifest distinct labels min/max: `5 / 5`
- gate: `passed=true`

### C) Summary rendering

```bash
.venv/bin/python scripts/ci/summarize_graph2d_seed_gate.py \
  --summary-json /tmp/graph2d-seed-gate/seed_sweep_summary.json \
  --title "Graph2D Seed Gate Local Check"
```

Confirmed summary includes `Manifest distinct labels (min/max)`.

## Notes

- `ezdxf` font cache warnings were observed during local runs.
- These warnings did not affect gate outcome.
