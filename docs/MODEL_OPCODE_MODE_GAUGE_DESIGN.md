# Model Opcode Mode Gauge Design

## Overview
Expose the configured opcode validation mode as a Prometheus gauge for
observability when model reloads run opcode scans.

## Updates
- Added `model_opcode_mode` gauge (0=audit, 1=blocklist/blacklist, 2=whitelist).
- Set the gauge during model reload based on `MODEL_OPCODE_MODE`.
- Added coverage to verify the gauge updates when reload runs under whitelist mode.

## Files
- `src/utils/analysis_metrics.py`
- `src/ml/classifier.py`
- `tests/unit/test_model_opcode_modes.py`
