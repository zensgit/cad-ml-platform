# Test Statistics (Auto-generated)

Generated: 2025-11-17 21:30

## Vision Module

**Total**: 29 pytest nodes, 29 test functions

| File | Pytest Nodes | Test Functions | Notes |
|------|-------------|----------------|-------|
| `test_image_loading.py` | 9 | 9 |  |
| `test_vision_endpoint.py` | 8 | 8 |  |
| `test_vision_golden_mvp.py` | 8 | 8 |  |
| `test_vision_ocr_integration.py` | 4 | 4 |  |

## OCR Module

**Total**: 94 pytest nodes, 75 test functions

| File | Pytest Nodes | Test Functions | Notes |
|------|-------------|----------------|-------|
| `test_dimension_matching.py` | 30 | 11 | parametrized (19 expanded) |
| `test_fallback.py` | 18 | 18 |  |
| `test_cache_key.py` | 12 | 12 |  |
| `test_idempotency.py` | 11 | 11 |  |
| `test_dimension_parser_precision.py` | 4 | 4 |  |
| `test_dimension_parser_regex.py` | 4 | 4 |  |
| `test_calibrator_v2.py` | 3 | 3 |  |
| `test_bbox_mapper.py` | 2 | 2 |  |
| `test_calibration.py` | 2 | 2 |  |
| `test_distributed_control.py` | 2 | 2 |  |
| `test_dynamic_threshold.py` | 2 | 2 |  |
| `test_golden_eval_report.py` | 1 | 1 |  |
| `test_image_enhancer.py` | 1 | 1 |  |
| `test_missing_fields_fallback.py` | 1 | 1 |  |
| `test_ocr_endpoint.py` | 1 | 1 |  |

## Summary

**口径说明**:
- **Pytest Nodes**: pytest --collect-only 的执行单元数（包含参数化展开）
- **Test Functions**: def test_* / async def test_* 的函数定义数
- 当 Nodes > Functions 时，表示存在 @pytest.mark.parametrize 展开
