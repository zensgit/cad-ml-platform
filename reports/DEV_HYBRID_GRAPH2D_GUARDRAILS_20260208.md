# DEV_HYBRID_GRAPH2D_GUARDRAILS_20260208

## Goal
Prevent Graph2D outputs (especially coarse taxonomy labels like `传动件/壳体类/轴类/连接件/其他`)
from overriding rule/text signals (filename/titleblock/process), while keeping Graph2D available
as a safe fallback/confirmatory signal.

This was motivated by earlier observations where Graph2D could:
- produce coarse labels with low confidence and still appear in downstream decisions
- block `TITLEBLOCK_OVERRIDE_ENABLED` adoption even when titleblock confidence was high
- ignore configured `graph2d.exclude_labels` / `graph2d.allow_labels`

## Changes
- `src/ml/hybrid_classifier.py`
  - Parse Graph2D allow/exclude lists from config/env and apply filtering.
  - Enforce `graph2d_min_conf` threshold: below-threshold predictions are ignored.
  - Guardrail: if filename/titleblock/process produced any label and Graph2D label does not
    match them, ignore Graph2D for fusion/adoption (prevents override).
  - Titleblock override is no longer blocked by Graph2D being present.
  - Decision output now returns accurate `source` when only one predictor is used
    (`filename_only/titleblock_only/graph2d_only/process_only`).
- `config/hybrid_classifier.yaml`
  - Expanded default `graph2d.exclude_labels` to include coarse taxonomy labels:
    `other,传动件,壳体类,轴类,连接件,其他`.
- Tests
  - Updated `tests/unit/test_filename_classifier.py` to reflect the new guardrail behavior.
  - Added `tests/unit/test_hybrid_classifier_graph2d_guardrails.py` covering:
    - exclude_labels filter
    - allow_labels filter
    - titleblock override wins over graph2d

## Verification
### Unit tests (targeted)
Command:
```bash
mkdir -p /tmp/pycache /tmp/xdg-cache
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/tmp/pycache \
XDG_CACHE_HOME=/tmp/xdg-cache \
pytest -q \
  tests/unit/test_filename_classifier.py \
  tests/unit/test_hybrid_classifier_graph2d_guardrails.py \
  tests/unit/test_golden_dxf_hybrid_manifest.py
```

Result: `9 passed`

### Local batch API smoke (no manual drawing review)
Dataset:
- `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`

Command:
```bash
mkdir -p /tmp/pycache /tmp/xdg-cache
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/tmp/pycache \
XDG_CACHE_HOME=/tmp/xdg-cache \
TITLEBLOCK_ENABLED=true \
TITLEBLOCK_OVERRIDE_ENABLED=true \
TITLEBLOCK_MIN_CONF=0.75 \
python3 scripts/batch_analyze_dxf_local.py \
  --dxf-dir "/Users/huazhou/Downloads/训练图纸/训练图纸_dxf" \
  --output-dir "reports/experiments/20260208/batch_analysis_training_dxf_hybrid_guardrails" \
  --max-files 30 \
  --seed 22
```

Artifacts:
- `reports/experiments/20260208/batch_analysis_training_dxf_hybrid_guardrails/summary.json`
- `reports/experiments/20260208/batch_analysis_training_dxf_hybrid_guardrails/batch_results_sanitized.csv`
- `reports/experiments/20260208/batch_analysis_training_dxf_hybrid_guardrails/label_distribution.csv`

Key results (from `summary.json`):
- `total=30 success=30 error=0`
- confidence buckets: `gte_0_8=30`
- titleblock: `label_present_rate=1.0`, `status_counts={"matched": 29, "partial_match": 1}`

