# V16分类器API增强开发总结

**日期**: 2026-02-07
**开发者**: Claude
**测试状态**: 7539 passed, 84 skipped

---

## 提交记录

| Commit | 描述 |
|--------|------|
| `639b1a2` | feat: add V16 classifier management APIs |
| `bb66cb1` | test: add unit tests for V16 classifier API endpoints |
| `3a88534` | feat: add Prometheus metrics for V16 classifier |

---

## 新增API端点

### 1. V16健康检查
```
GET /api/v1/health/v16-classifier
GET /api/v1/v16-classifier/health
```

**响应字段:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | ok/unavailable/disabled/error |
| `loaded` | bool | 分类器是否加载 |
| `speed_mode` | string | 当前速度模式 |
| `cache_enabled` | bool | 缓存是否启用 |
| `cache_size` | int | 当前缓存大小 |
| `cache_max_size` | int | 最大缓存容量 |
| `cache_hits` | int | 缓存命中次数 |
| `cache_misses` | int | 缓存未命中次数 |
| `cache_hit_ratio` | float | 缓存命中率 |
| `v6_model_loaded` | bool | V6模型加载状态 |
| `v14_model_loaded` | bool | V14模型加载状态 |
| `dwg_converter_available` | bool | DWG转换器可用性 |
| `categories` | list | 支持的分类类别 |

### 2. V16缓存清除
```
POST /api/v1/v16-classifier/cache/clear
```
**需要**: Admin Token

**响应字段:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | ok/unavailable/error |
| `cleared_entries` | int | 清除的条目数 |
| `previous_hits` | int | 清除前命中数 |
| `previous_misses` | int | 清除前未命中数 |

### 3. V16速度模式切换
```
GET  /api/v1/v16-classifier/speed-mode
POST /api/v1/v16-classifier/speed-mode
```
**POST需要**: Admin Token

**速度模式:**
| 模式 | V14折数 | 快速渲染 | 适用场景 |
|------|---------|----------|----------|
| `accurate` | 5 | 否 | 最高精度 |
| `balanced` | 3 | 是 | 平衡(推荐) |
| `fast` | 1 | 是 | 高吞吐量 |
| `v6_only` | 0 | - | 仅V6模型 |

### 4. 批量分类
```
POST /api/v1/analyze/batch-classify
```

**请求参数:**
| 参数 | 类型 | 说明 |
|------|------|------|
| `files` | File[] | CAD文件列表(DXF/DWG) |
| `max_workers` | int | 并行线程数(可选) |

**响应字段:**
| 字段 | 类型 | 说明 |
|------|------|------|
| `total` | int | 总文件数 |
| `success` | int | 成功分类数 |
| `failed` | int | 失败数 |
| `processing_time` | float | 处理时间(秒) |
| `results` | list | 每个文件的分类结果 |

---

## Prometheus指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `v16_classifier_loaded` | Gauge | 加载状态(1=已加载) |
| `v16_classifier_cache_hits_total` | Counter | 缓存命中总数 |
| `v16_classifier_cache_misses_total` | Counter | 缓存未命中总数 |
| `v16_classifier_cache_size` | Gauge | 当前缓存大小 |
| `v16_classifier_cache_max_size` | Gauge | 最大缓存容量 |
| `v16_classifier_inference_seconds` | Histogram | 单次推理延迟 |
| `v16_classifier_batch_seconds` | Histogram | 批量推理延迟 |
| `v16_classifier_batch_size` | Histogram | 批量大小分布 |
| `v16_classifier_predictions_total` | Counter | 预测计数(按类别/模式) |
| `v16_classifier_speed_mode` | Gauge | 当前速度模式(0-3) |
| `v16_classifier_needs_review_total` | Counter | 需人工复核计数 |

---

## 单元测试

**文件**: `tests/unit/test_v16_classifier_endpoints.py`
**测试数**: 17

| 测试类 | 测试数 | 覆盖内容 |
|--------|--------|----------|
| TestV16HealthEndpoint | 4 | 健康检查各状态 |
| TestV16CacheClearEndpoint | 3 | 缓存清除和权限 |
| TestV16SpeedModeEndpoint | 5 | 速度模式获取/切换 |
| TestBatchClassifyEndpoint | 5 | 批量分类各场景 |

---

## 相关文件变更

| 文件 | 变更 |
|------|------|
| `src/api/v1/health.py` | +235行 新增V16管理端点 |
| `src/api/v1/analyze.py` | +136行 新增批量分类端点 |
| `src/utils/analysis_metrics.py` | +50行 新增V16 Prometheus指标 |
| `tests/unit/test_v16_classifier_endpoints.py` | +328行 新增单元测试 |

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DISABLE_V16_CLASSIFIER` | - | 设为1/true禁用V16 |
| `V16_SPEED_MODE` | fast | 默认速度模式 |
| `V16_CACHE_SIZE` | 1000 | 缓存最大条目数 |

---

## 使用示例

### 检查V16状态
```bash
curl -H "X-API-Key: your-key" \
  http://localhost:8000/api/v1/health/v16-classifier
```

### 切换速度模式
```bash
curl -X POST \
  -H "X-API-Key: your-key" \
  -H "X-Admin-Token: your-admin-token" \
  -H "Content-Type: application/json" \
  -d '{"speed_mode": "fast"}' \
  http://localhost:8000/api/v1/v16-classifier/speed-mode
```

### 批量分类
```bash
curl -X POST \
  -H "X-API-Key: your-key" \
  -F "files=@part1.dxf" \
  -F "files=@part2.dxf" \
  http://localhost:8000/api/v1/analyze/batch-classify
```

### 清除缓存
```bash
curl -X POST \
  -H "X-API-Key: your-key" \
  -H "X-Admin-Token: your-admin-token" \
  http://localhost:8000/api/v1/v16-classifier/cache/clear
```
