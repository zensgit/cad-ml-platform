# DEV_DXF_CLASS_BALANCE_STRATEGY_VALIDATION_20260125

## Validation Summary
- Verified class-balancer loss selection and weighting logic through unit tests.
- Confirmed evaluation script writes macro/weighted F1 fields.

## Tests
```
.venv-graph/bin/python -m pytest tests/unit/test_class_balancer.py -v
```
- 3 passed

## Checks
- `scripts/train_2d_graph.py` uses `ClassBalancer` for loss selection.
- `scripts/eval_2d_graph.py` outputs precision/recall/F1 per label and macro/weighted F1 overall.
