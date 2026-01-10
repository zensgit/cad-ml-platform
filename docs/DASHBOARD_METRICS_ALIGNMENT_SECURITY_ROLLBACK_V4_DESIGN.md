# Dashboard Metrics Alignment: Security, Rollback, and v4 Metrics

## Overview
Align Grafana dashboard panels with metrics that are actually exported and
available in the backend, and add missing observability metrics to support
model security, rollback, and v4 feature extraction panels.

## Updates
- Added model rollback gauges/counters (`model_rollback_level`,
  `model_snapshots_available`, `model_rollback_total`).
- Added opcode scan counters (`model_opcode_scans_total`,
  `model_opcode_blocked_total`) and v4 feature histograms (`v4_surface_count`,
  `v4_shape_entropy`).
- Wired model reloads and v4 extraction to record the new metrics.
- Updated dashboard panels to use existing metric names and drift histograms.
- Extended dashboard validation script to include the main dashboard.

## Files
- `src/utils/analysis_metrics.py`
- `src/ml/classifier.py`
- `src/api/v1/health.py`
- `src/core/feature_extractor.py`
- `config/grafana/dashboard_main.json`
- `scripts/validate_dashboard_metrics.py`
- `tests/unit/test_model_rollback_health.py`
- `tests/unit/test_model_rollback_level3.py`
- `tests/unit/test_model_opcode_modes.py`
- `tests/unit/test_v4_feature_performance.py`
