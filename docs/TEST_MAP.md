# Test Map - 测试文件覆盖说明

**生成日期**: 2025-11-17
**测试总数**: 123 pytest nodes (Vision: 29, OCR: 94)
**统计口径**: Pytest 节点数（包含参数化展开）

> **自动生成**: `python3 scripts/list_tests.py --markdown`

---

## Vision 模块测试 (29 nodes / 29 functions)

| 文件 | Pytest Nodes | Functions | 覆盖模块 | 关键场景 |
|------|-------------|-----------|---------|---------|
| `test_image_loading.py` | 9 | 9 | Image URL Loading | URL 下载、错误处理、重定向 |
| `test_vision_endpoint.py` | 8 | 8 | API Endpoint + Stub Provider | 端点调用、base64 编解码、错误响应 |
| `test_vision_golden_mvp.py` | 8 | 8 | Golden Evaluation | 关键词匹配、样本评估、标注结构 |
| `test_vision_ocr_integration.py` | 4 | 4 | Vision+OCR 联合 | 集成成功、降级、跳过 OCR |

### 详细说明

**test_image_loading.py** (9 tests)
- URL 方案验证（HTTPS 强制）
- HTTP 错误处理（404, 403, timeout）
- 大文件拒绝
- 空图像检测
- 重定向跟随

**test_vision_endpoint.py** (8 tests)
- Base64 正常路径
- 缺失图像错误
- 无效 Base64 格式
- 健康检查端点
- Stub Provider 直接调用

**test_vision_golden_mvp.py** (8 tests)
- 关键词完美匹配
- 部分匹配计算
- 大小写不敏感
- 空关键词处理
- Golden 标注结构验证

**test_vision_ocr_integration.py** (4 tests)
- Vision+OCR 成功集成
- OCR 失败时 Vision 降级
- 跳过 OCR 模式
- 无 OCR Manager 时行为

---

## OCR 模块测试 (94 nodes / 75 functions)

| 文件 | Pytest Nodes | Functions | 覆盖模块 | 关键场景 |
|------|-------------|-----------|---------|---------|
| `test_dimension_matching.py` | 30 | 11 | 尺寸匹配 | 公差、单位转换、螺纹、召回率 **(parametrized)** |
| `test_fallback.py` | 18 | 18 | 三级降级策略 | JSON→Markdown→Regex、中文、性能 |
| `test_cache_key.py` | 12 | 12 | 缓存键生成 | 确定性、版本化、失效策略 |
| `test_idempotency.py` | 11 | 11 | 幂等性支持 | 键构建、缓存命中/未命中、存储 |
| `test_dimension_parser_precision.py` | 4 | 4 | 精密尺寸解析 | 双向公差、螺距、混合序列 |
| `test_dimension_parser_regex.py` | 4 | 4 | 正则解析器 | 直径/半径/螺纹、几何符号 |
| `test_calibrator_v2.py` | 3 | 3 | 多证据校准器 | 全证据、缺失证据、自适应权重 |
| `test_bbox_mapper.py` | 2 | 2 | BBox 映射 | 原始文本→边界框匹配 |
| `test_calibration.py` | 2 | 2 | 置信度校准 V1 | 权重平衡、缺失输入处理 |
| `test_distributed_control.py` | 2 | 2 | 分布式控制 | 限流、熔断 |
| `test_dynamic_threshold.py` | 2 | 2 | 动态阈值 | EMA 自适应、边界限制 |
| `test_golden_eval_report.py` | 1 | 1 | 评估报告生成 | 报告格式验证 |
| `test_image_enhancer.py` | 1 | 1 | 图像增强 | 无 PIL 时回退 |
| `test_missing_fields_fallback.py` | 1 | 1 | 缺失字段降级 | Schema 不完整时处理 |
| `test_ocr_endpoint.py` | 1 | 1 | API 冒烟测试 | 端点基本功能 |

> **Note**: `test_dimension_matching.py` 使用 @pytest.mark.parametrize，11 个函数展开为 30 个测试节点

### 详细说明

**test_cache_key.py** (12 nodes)
- 确定性：相同输入产生相同键
- 敏感性：图像/provider/prompt 版本变化时键变化
- 裁剪配置顺序无关性
- 缓存查找模拟
- 版本变更失效矩阵

**test_dimension_matching.py** (30 nodes / 11 functions)
- 数值匹配（公差范围内/外）- 参数化展开
- 单位标准化（mm/cm/m/inch/毫米/厘米）- 参数化展开
- 螺纹匹配（直径+螺距）- 参数化展开
- 召回率计算（完美/部分/零）
- 阈值一致性

**test_fallback.py** (18 nodes)
- JSON 直接解析（无降级）
- Markdown 围栏提取（大小写不敏感）
- 纯文本正则提取（公差/螺距）
- 无效 JSON 恢复
- 中文文本处理
- 空输出处理
- Schema 深度验证
- 性能测试（<100ms）

**test_idempotency.py** (11 nodes)
- 键格式：`idempotency:{endpoint}:{key}`
- 缓存命中返回已存储响应
- 缓存未命中返回 None
- 空键跳过存储
- 自定义 TTL
- 完整流程：存储→检索

---

## 测试依赖图

```
Vision Tests
├── test_image_loading (独立)
├── test_vision_endpoint → VisionManager, StubProvider
├── test_vision_golden_mvp → StubProvider
└── test_vision_ocr_integration → VisionManager, OcrManager

OCR Tests
├── Core Logic (无外部依赖)
│   ├── test_dimension_matching
│   ├── test_dimension_parser_*
│   ├── test_calibration
│   └── test_cache_key
├── Strategy (内部抽象)
│   ├── test_fallback
│   ├── test_distributed_control
│   └── test_dynamic_threshold
└── Integration (需要 Mock)
    ├── test_idempotency (Redis mock)
    └── test_ocr_endpoint (FastAPI)
```

---

## 运行指南

### 按模块运行
```bash
pytest tests/vision/ -v        # 所有 Vision 测试
pytest tests/ocr/ -v           # 所有 OCR 测试
```

### 按功能运行
```bash
# 核心逻辑
pytest tests/ocr/test_dimension_matching.py -v
pytest tests/ocr/test_fallback.py -v

# 集成测试
pytest tests/vision/test_vision_ocr_integration.py -v
pytest tests/ocr/test_idempotency.py -v

# 性能相关
pytest tests/ocr/test_fallback.py::TestFallbackStrategy::test_fallback_performance -v
```

### 快速健康检查
```bash
pytest tests/ -q | tail -5     # 快速汇总
make health-check               # 系统状态
```

### 联合评估（Vision+OCR）
```bash
# 快速联合评估（打印报告与分数）
make eval-combined

# 保存到历史（reports/eval_history/*_combined.json）
make eval-combined-save

# 仅查看基线配置（不运行评估）
python3 scripts/evaluate_vision_ocr_combined.py --report-only
```

### CI 联合评估门禁（支持环境变量阈值）
```bash
# 本地/CI 默认阈值（combined=0.80 / vision=0.65 / ocr=0.90）
make ci-combined-check

# 覆盖阈值（示例）
MIN_COMBINED=0.82 MIN_VISION=0.68 MIN_OCR=0.90 make ci-combined-check

# GitHub Actions / Jenkins 中可通过环境变量注入
#   MIN_COMBINED: "0.82"
#   MIN_VISION:   "0.68"
#   MIN_OCR:      "0.90"
```

---

## 覆盖率分析

### 已覆盖
- [x] 数据模型（DimensionInfo, SymbolInfo, OcrResult）
- [x] 解析逻辑（正则、降级、单位转换）
- [x] 缓存策略（键生成、失效）
- [x] 分布式控制（限流、熔断）
- [x] 置信度校准（单/多证据）
- [x] API 端点（冒烟测试）
- [x] 幂等性（请求去重）
- [x] Golden 评估（召回率、Brier）

### 待扩展
- [ ] 真实图片测试（synthetic → real samples）
- [ ] Provider 集成测试（Paddle/DeepSeek 真实调用）
- [ ] 端到端流程测试（图片上传 → 结构化输出）
- [ ] 边缘案例（极端公差、复杂 GD&T 符号）
- [ ] 性能压力测试（并发、大文件）

---

## 维护指南

### 添加新测试时
1. 确定覆盖模块和场景
2. 更新本文档对应表格
3. 遵循命名约定：`test_{module}_{scenario}.py`
4. 添加必要的文档字符串

### 测试文件命名约定
- `test_{feature}.py` - 单一功能测试
- `test_{feature}_{aspect}.py` - 功能某方面（如 `test_dimension_parser_regex.py`）
- `test_{module}_integration.py` - 集成测试

### 何时更新此文档
- 添加新测试文件时
- 测试覆盖范围显著变化时
- 重构测试结构时
