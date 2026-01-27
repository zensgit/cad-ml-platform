# DEV_DXF_NODE_IMPORTANCE_SAMPLING_VALIDATION_20260125

## Validation Summary
- Ran unit test to verify importance sampling respects max node count and preserves text entities.

## Test Run
```
.venv-graph/bin/python -m pytest tests/unit/test_importance_sampling.py -v
```

## Results
- 1 passed (with ezdxf deprecation warnings)

## Notes
- Warnings come from `ezdxf` query parser deprecations; no functional impact observed.
