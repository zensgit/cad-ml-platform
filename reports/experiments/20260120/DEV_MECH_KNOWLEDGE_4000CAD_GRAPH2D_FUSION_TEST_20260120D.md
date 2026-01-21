# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_FUSION_TEST_20260120D

## Summary
Validated the DXF fusion integration path with Graph2D enabled after retraining
on the finalized labels.

## Command
```
GRAPH2D_ENABLED=true GRAPH2D_MODEL_PATH=models/graph2d_merged_latest.pth \
  ./.venv-graph/bin/python -m pytest tests/integration/test_analyze_dxf_fusion.py -v
```

## Result
- `tests/integration/test_analyze_dxf_fusion.py::test_analyze_dxf_triggers_l2_fusion` PASSED
- Warnings: 7 deprecation warnings from ezdxf query parser
