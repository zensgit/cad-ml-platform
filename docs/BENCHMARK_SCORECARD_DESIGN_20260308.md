# Benchmark Scorecard Design 2026-03-08

## Goal

Provide a single benchmark artifact that summarizes the current production line
across recognition quality, evidence coverage, and rollout governance.

The repository already had multiple evaluation tools, but each tool reported
its own local summary. This design adds one consolidator so release decisions
can point to one scorecard instead of many scattered docs.

## Inputs

The scorecard consumes existing JSON summaries and does not introduce a new
evaluation pipeline:

- `eval_hybrid_dxf_manifest.py` summary
- `train_2d_graph.py --metrics-out` summary
- `diagnose_graph2d_on_dxf_dir.py` summary
- `eval_history_sequence_classifier.py` summary
- `eval_brep_step_dir.py` summary
- vector migration pending/plan summary

Every input is optional except the output destination. Missing inputs are
reported explicitly as `missing`.

## Output

Two output formats are generated:

- JSON for downstream automation
- Markdown for docs, release notes, and benchmark review

Top-level fields:

- `overall_status`
- `components.hybrid`
- `components.graph2d`
- `components.history_sequence`
- `components.brep`
- `components.migration_governance`
- `recommendations`

## Scoring Policy

The policy is intentionally conservative.

### Hybrid

- `strong_primary` when exact hybrid accuracy is already strong
- flags `has_output_gap=true` when `hybrid_label` materially outperforms
  `final_part_type`

### Graph2D

- `weak_signal_only` when diagnosis accuracy is low or low-confidence saturation
  is very high
- this matches the current product position of Graph2D as a weak branch, not
  the main classifier

### History Sequence

- `smoke_only` when there are too few labeled `.h5` samples
- `evidence_ready` only when sample size, coverage, and coarse accuracy are all
  meaningful

### B-Rep

- `graph_ready` only when real 3D validation exists and `graph_schema_version`
  shows `v2`

### Migration Governance

- `operationally_ready` only when the migration plan is both ready and
  coverage-complete

## Why This Design

This scorecard is not trying to hide weak branches. It is designed to surface
them:

- which branch is strong enough to trust
- which branch is still only smoke-tested
- which governance controls are blocking rollout
- what recommendation follows from the current scorecard

That makes the benchmark line usable for product and release decisions, not
just research tracking.
