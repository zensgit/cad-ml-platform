# DEV_GRAPH2D_ALLOW_EXCLUDE_CONFIG_UNIFICATION_20260208

## Goal
Make Graph2D allow/exclude filtering consistent across:
- API (`/api/v1/analyze`) payload fields: `graph2d_prediction.allowed/excluded`
- Soft-override gating (which depends on those flags)
- HybridClassifier config (`config/hybrid_classifier.yaml`) and env overrides

Previously:
- `src/api/v1/analyze.py` used legacy env vars: `GRAPH2D_ALLOW_LABELS` / `GRAPH2D_EXCLUDE_LABELS`
- `src/ml/hybrid_config.py` used newer env vars: `GRAPH2D_FUSION_ALLOW_LABELS` / `GRAPH2D_FUSION_EXCLUDE_LABELS`

This mismatch meant Hybrid and API could disagree about what labels are allowed/excluded.

## Changes
- `src/ml/hybrid_config.py`
  - Backward-compatible env precedence:
    1. Prefer `GRAPH2D_FUSION_*` variables when present
    2. Fall back to legacy `GRAPH2D_*` variables
- `src/api/v1/analyze.py`
  - When `GRAPH2D_ALLOW_LABELS` / `GRAPH2D_EXCLUDE_LABELS` are not provided,
    default to `src/ml/hybrid_config.get_config().graph2d.{allow_labels,exclude_labels}`
    so `config/hybrid_classifier.yaml` drives behavior consistently.

## Verification
### Unit
```bash
mkdir -p /tmp/pycache /tmp/xdg-cache
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/tmp/pycache \
XDG_CACHE_HOME=/tmp/xdg-cache \
pytest -q tests/unit/test_hybrid_config_loader.py
```

### Integration
```bash
mkdir -p /tmp/pycache /tmp/xdg-cache
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/tmp/pycache \
XDG_CACHE_HOME=/tmp/xdg-cache \
pytest -q tests/integration/test_analyze_dxf_fusion.py
```

Result: `6 passed`

