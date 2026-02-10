# 测试覆盖率提升计划

**日期**: 2026-02-10
**当前覆盖率**: ~65%
**目标覆盖率**: 85%

---

## 已完成/已提升

| 模块 | 原覆盖率 | 新覆盖率 | 新增测试 |
|------|----------|----------|----------|
| `src/core/providers/classifier.py` | 54% | **80%** | 23个 |
| `src/core/providers/base.py` | 93% | **100%** | 23个 |
| `src/core/providers/readiness.py` | 78% | **100%** | 28个 |
| `src/core/providers/vision.py` | 85% | **100%** | 14个 |
| `src/core/providers/registry.py` | 96% | **100%** | 19个 |
| `src/core/providers/ocr.py` | 86% | **100%** | 20个 |
| `src/core/providers/knowledge.py` | 91% | **100%** | 16个 |
| `src/core/providers/bootstrap.py` | 92% | **100%** | 10个 |
| `src/ml/hybrid_classifier.py` | 71% | **82%** | 28个 |
| `src/utils/dxf_features.py` | 71% | **100%** | 19个 |
| `src/ml/part_classifier.py` | 1% | **37%** | 34个 (torch mock限制) |
| `src/utils/logging.py` | 69% | **86%** | 25个 |
| `src/utils/safe_eval.py` | 18% | **99%** | 54个 |
| `src/utils/cache.py` | 38% | **94%** | 33个 |
| `src/utils/circuit_breaker.py` | 27% | **100%** | 已有71个测试 |

---

## 优先级1: 核心Provider模块 (当前~98%)

| 模块 | 覆盖率 | 缺失行 | 建议 |
|------|--------|--------|------|
| base.py | ~~93%~~ | **100%** | ✅ 完成 |
| readiness.py | ~~78%~~ | **100%** | ✅ 完成 |
| vision.py | ~~85%~~ | **100%** | ✅ 完成 |
| registry.py | ~~96%~~ | **100%** | ✅ 完成 |
| ocr.py | ~~86%~~ | **100%** | ✅ 完成 |
| knowledge.py | ~~91%~~ | **100%** | ✅ 完成 |
| bootstrap.py | ~~92%~~ | **100%** | ✅ 完成 |
| classifier.py | 80% | 144,151,168等 | 主要是provider内部分支 |

---

## 优先级2: ML模块 (当前~50%)

| 模块 | 覆盖率 | 说明 |
|------|--------|------|
| `src/ml/part_classifier.py` | ~~1%~~ **37%** | torch mock限制，需完整torch环境达到更高覆盖 |
| `src/ml/hybrid_classifier.py` | ~~71%~~ **82%** | ✅ 已优化 |
| `src/ml/vision_2d.py` | 26% | Graph2D分类器 |
| `src/ml/vision_3d.py` | 28% | 3D分析 |

---

## 优先级3: 工具模块 (当前~95%)

| 模块 | 覆盖率 | 说明 |
|------|--------|------|
| `src/utils/dxf_features.py` | ~~71%~~ **100%** | ✅ 已完成 |
| `src/utils/logging.py` | ~~69%~~ **86%** | ✅ 已优化 |
| `src/utils/safe_eval.py` | ~~18%~~ **99%** | ✅ 已完成 |
| `src/utils/cache.py` | ~~38%~~ **94%** | ✅ 已优化 |
| `src/utils/circuit_breaker.py` | ~~27%~~ **100%** | ✅ 已完成 |

---

## 优先级4: API模块 (当前~80%)

| 模块 | 覆盖率 | 说明 |
|------|--------|------|
| `src/api/v1/health.py` | 85% | 已有V16测试 |
| `src/api/v1/analyze.py` | 78% | 核心分析API |

---

## 排除模块 (低优先级)

以下模块覆盖率低但非核心功能，可暂不处理：

- `src/core/api_versioning/` - 0%，但未启用
- `src/core/assistant/server.py` - 0%，独立服务
- `src/ml/tuning/` - 16-36%，训练相关
- `src/api/openapi_config.py` - 0%，配置文件

---

## 执行计划

### 第1周
- [x] classifier.py (54% → 80%)
- [x] base.py (93% → 100%) ✅
- [x] readiness.py (78% → 100%) ✅

### 第2周
- [x] vision.py (85% → 100%) ✅
- [x] ocr.py (86% → 100%) ✅
- [x] registry.py (96% → 100%) ✅
- [x] knowledge.py (91% → 100%) ✅
- [x] bootstrap.py (92% → 100%) ✅
- [x] hybrid_classifier.py (71% → 82%) ✅

### 第3周
- [x] part_classifier.py (1% → 37%) ✅ (torch mock限制)
- [x] dxf_features.py (71% → 100%) ✅

### 第4周
- [ ] 清理和文档
- [ ] 达成85%总覆盖率

---

## 测试策略

### 单元测试
- Mock外部依赖 (torch, ezdxf, Redis)
- 覆盖所有公共API
- 覆盖错误路径和边界条件

### 集成测试
- 需要运行服务器的测试标记为 `@pytest.mark.integration`
- Contract测试优先使用 `localhost:8000`；端口绑定受限环境中，manual contract 断言可回退到 TestClient

### 性能测试
- 基准测试放在 `tests/benchmark/`
- 不计入覆盖率

---

## 覆盖率排除

以下文件已在 `.coveragerc` 中排除：
- `*/__pycache__/*`
- `*/migrations/*`
- `*_test.py`
- `conftest.py`

---

*Updated by Claude Code - 2026-02-10*
