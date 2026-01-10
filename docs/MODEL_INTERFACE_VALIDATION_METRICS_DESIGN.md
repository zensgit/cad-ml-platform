# Model Interface Validation Metrics Design

## Overview
Record interface validation failures during model reloads and surface the
failure reason through a dedicated Prometheus counter.

## Updates
- Added `model_interface_validation_fail_total{reason}` metric.
- Integrated `validate_model_interface` into model reload to enforce validation
  and increment the failure counter on invalid models.
- Extended reload failure coverage to assert the metric increments when a model
  lacks required methods.

## Files
- `src/utils/analysis_metrics.py`
- `src/ml/classifier.py`
- `tests/unit/test_model_security_validation.py`
