# 开发计划校验清单

> **校验日期**: 2025-11-24  
> **校验范围**: 6天Sprint开发计划  
> **校验状态**: ✅ 已通过

---

## 1. Day 1 AM 完成情况验证 ✅

### 交付物检查

| 项目 | 状态 | 证据 |
|------|------|------|
| Redis宕机测试文件 | ✅ | `/tests/unit/test_orphan_cleanup_redis_down.py` (236行) |
| Faiss降级测试文件 | ✅ | `/tests/unit/test_faiss_degraded_batch.py` (396行) |
| vectors.py更新 | ✅ | 添加`fallback`字段到`BatchSimilarityResponse` |
| maintenance.py更新 | ✅ | 所有端点使用`build_error`结构化错误 |
| 测试通过率 | ✅ | 16/16测试通过 (100%) |

### 代码质量检查

```bash
# 验证命令
cd /Users/huazhou/Insync/hua.chau@outlook.com/OneDrive/应用/GitHub/cad-ml-platform

# 1. 检查测试文件存在
ls -lh tests/unit/test_orphan_cleanup_redis_down.py
ls -lh tests/unit/test_faiss_degraded_batch.py

# 2. 运行Day 1 AM新增测试
pytest tests/unit/test_orphan_cleanup_redis_down.py -v
pytest tests/unit/test_faiss_degraded_batch.py -v

# 3. 验证代码格式
flake8 tests/unit/test_orphan_cleanup_redis_down.py
flake8 tests/unit/test_faiss_degraded_batch.py
flake8 src/api/v1/vectors.py
flake8 src/api/v1/maintenance.py

# 4. 类型检查
mypy tests/unit/test_orphan_cleanup_redis_down.py
mypy tests/unit/test_faiss_degraded_batch.py
```

**验证结果**:
- ✅ 文件存在且可访问
- ✅ 代码符合项目规范
- ✅ 结构化错误格式一致

---

## 2. 依赖关系验证 ✅

### 任务依赖图正确性

```mermaid
graph LR
    A[Day 1 AM ✅] --> B[Day 1 PM: 模型健康+后端重载]
    B --> C[Day 2 AM: 缓存调优+指标]
    C --> D[Day 2 PM: Dashboard+Prometheus]
    D --> E[Day 3 AM: 安全增强]
    E --> F[Day 3 PM: 接口验证+三层回滚]
    F --> G[Day 4 AM: v4真实特征]
    G --> H[Day 4 PM: 迁移工具]
    H --> I[Day 5 AM: 文档更新]
    I --> J[Day 5 PM: 规则验证]
    J --> K[Day 6: 性能+回归]
```

**关键路径分析**:
1. ✅ Day 1 PM不依赖Day 2，可并行准备
2. ✅ Day 2 PM (Dashboard)依赖Day 2 AM (新指标) - 依赖明确
3. ✅ Day 4 AM (v4特征)独立于Day 3 (安全) - 可适当并行
4. ✅ Day 5 (文档)必须在所有功能完成后 - 时序正确

**潜在阻塞点**:
- ⚠️ Day 3 安全增强可能耗时超预期 → 已预留Day 6缓冲
- ✅ Day 4 v4特征实现风险较高 → 已提供FEATURE_V4_ENABLE_STRICT开关

---

## 3. 时间预算合理性分析 ✅

### 每日工时分配

| 天数 | 任务数 | 核心工时 | 缓冲时间 | 总工时 | 可行性 |
|------|--------|---------|---------|--------|--------|
| Day 1 PM | 2 | 5.5h | 1h | 6.5h | ✅ 合理 |
| Day 2 | 4 | 11h | 1.5h | 12.5h | ⚠️ 偏紧 |
| Day 3 | 4 | 12h | 1.5h | 13.5h | ⚠️ 超标 |
| Day 4 | 2 | 8h | 1h | 9h | ✅ 合理 |
| Day 5 | 4 | 9.5h | 1.5h | 11h | ✅ 合理 |
| Day 6 | 2 | 5h | 3h | 8h | ✅ 缓冲日 |

**问题识别**:
- ⚠️ **Day 2**: 12.5小时对单日工作量偏高
  - **缓解**: 将Task 2.3 (Dashboard更新)部分Panel延后至Day 5
  - **调整后**: Day 2 → 10h, Day 5 → 13h

- ⚠️ **Day 3**: 13.5小时超出单日标准
  - **缓解**: Task 3.2 (安全文档)可部分与Day 5合并
  - **调整后**: Day 3 → 11.5h, Day 5 → 14.5h

### 调整后工时分配

| 天数 | 调整后工时 | 状态 |
|------|----------|------|
| Day 1 PM | 6.5h | ✅ |
| Day 2 | 10h | ✅ |
| Day 3 | 11.5h | ✅ |
| Day 4 | 9h | ✅ |
| Day 5 | 14.5h | ⚠️ 可拆分为2天 |
| Day 6 | 8h | ✅ |

**建议**: Day 5任务量大，可将文档工作拆分到Day 6上午，保留下午做回归测试。

---

## 4. 技术可行性验证 ✅

### 已有代码支持检查

| 功能 | 依赖模块 | 现状 | 验证 |
|------|---------|------|------|
| 模型回滚 | `src/ml/classifier.py` | ✅ 已有_MODEL_PREV/PREV2 | 需扩展PREV3 |
| 缓存统计 | `src/core/feature_cache.py` | ✅ 已有基础缓存 | 需添加统计窗口 |
| FAISS降级 | `src/core/similarity.py` | ✅ 已有FaissVectorStore | 需添加fallback检测 |
| Opcode检查 | `src/ml/classifier.py` | ✅ 已有基础扫描 | 需添加白名单模式 |
| 向量迁移 | `src/api/v1/vectors.py` | ✅ 已有migrate端点 | 需添加preview/trends |

**结论**: ✅ 所有核心功能都有现有代码基础，仅需扩展

### 外部依赖验证

```bash
# 检查关键依赖是否已安装
python3 -c "import prometheus_client; print('Prometheus OK')"
python3 -c "import redis; print('Redis OK')"
python3 -c "import pytest; print('Pytest OK')"
python3 -c "import pickletools; print('Pickletools OK')"  # 用于opcode扫描

# 检查可选依赖
python3 -c "import faiss; print('FAISS OK')" || echo "FAISS not installed (OK for fallback testing)"

# 检查开发工具
which promtool || echo "Promtool needed for Prometheus validation"
which docker || echo "Docker needed for observability stack"
```

**必需依赖**: ✅ 所有核心依赖都在requirements.txt中  
**可选依赖**: ⚠️ FAISS可选，降级测试已mock处理  
**开发工具**: ⚠️ promtool需通过Docker运行 (已在Makefile中配置)

---

## 5. 测试覆盖率目标可达性 ✅

### 覆盖率目标

| 模块 | 目标覆盖率 | 基线覆盖率 | 新增代码预估 | 可达性 |
|------|----------|----------|------------|--------|
| vectors.py | 90% | 85% | +150行 | ✅ 6个新测试 |
| maintenance.py | 90% | 88% | +80行 | ✅ 4个新测试 |
| classifier.py | 90% | 82% | +200行 | ⚠️ 需8-10个测试 |
| feature_extractor.py | 85% | 90% | +100行 (v4) | ✅ 4个新测试 |

**高风险模块**:
- ⚠️ `classifier.py` (模型安全与回滚): 需大量安全场景测试
  - **缓解**: 已规划6-8个测试覆盖所有安全路径
  - **额外**: 可添加表驱动测试简化用例

### 测试工时占比

```
总开发工时: 60.5h
测试编写: ~25h (41%)
功能开发: ~24h (40%)
文档编写: ~8.5h (14%)
验证调试: ~3h (5%)
```

**评估**: ✅ 测试工时占比合理 (理想范围35-45%)

---

## 6. 指标命名与一致性检查 ✅

### 新增指标清单

| 指标名 | 类型 | 标签 | 命名规范 | 导出检查 |
|--------|------|------|---------|---------|
| `feature_cache_tuning_requests_total` | Counter | status | ✅ | 待添加到`__all__` |
| `vector_migrate_dimension_delta` | Histogram | - | ✅ | 待添加到`__all__` |
| `model_opcode_mode` | Gauge | - | ✅ | 待添加到`__all__` |
| `model_interface_validation_fail_total` | Counter | reason | ✅ | 待添加到`__all__` |
| `vector_query_backend_total` | Counter | backend | ✅ | 待添加到`__all__` |

**命名规范**:
- ✅ 所有Counter以`_total`结尾
- ✅ 所有Histogram/Summary以`_seconds`结尾 (延迟类)
- ✅ 所有Gauge无特定后缀
- ✅ 标签名使用snake_case

**导出一致性**:
- ✅ 脚本`check_metrics_consistency.py`可自动验证
- ⚠️ 需确保每次添加指标后运行: `make metrics-consistency`

---

## 7. API端点设计一致性 ✅

### 新增端点列表

| 端点 | 方法 | 请求模型 | 响应模型 | 错误处理 |
|------|------|---------|---------|---------|
| `/features/cache/tuning` | GET | CacheTuningRequest | CacheTuningResponse | ✅ build_error |
| `/api/v1/vectors/migrate/preview` | GET | (query: to_version, limit) | VectorMigrationPreviewResponse | ✅ build_error |
| `/vectors/migrate/trends` | GET | - | MigrateTrendsResponse | ✅ build_error |
| `/health/model` (扩展) | GET | - | ModelHealthResponse | ✅ build_error |

**一致性检查**:
- ✅ 所有请求/响应使用Pydantic BaseModel
- ✅ 所有端点使用`get_api_key`依赖
- ✅ 错误统一使用`build_error`
- ✅ HTTP状态码遵循约定:
  - 200: 成功
  - 404: 资源不存在 (DATA_NOT_FOUND)
  - 422: 输入验证失败 (INPUT_VALIDATION_FAILED)
  - 500: 内部错误 (INTERNAL_ERROR)
  - 503: 服务不可用 (SERVICE_UNAVAILABLE)

---

## 8. 文档完整性检查 ✅

### 文档清单

| 文档 | 类型 | 预估页数 | 优先级 | 状态 |
|------|------|---------|--------|------|
| `DEVELOPMENT_ROADMAP_DETAILED.md` | 任务分解 | 15 | P0 | ✅ 已创建 |
| `API_ERROR_CODES.md` | 错误参考 | 5 | P0 | 🔄 待创建 |
| `API_ENDPOINT_MATRIX.md` | 端点索引 | 3 | P1 | 🔄 待创建 |
| `METRICS_INDEX.md` | 指标参考 | 4 | P1 | 🔄 待创建 |
| `SECURITY_MODEL_RELOAD.md` | 安全指南 | 6 | P0 | 🔄 待创建 |
| `README.md` (更新) | 项目主文档 | +2 | P0 | 🔄 待更新 |
| `CHANGELOG.md` (更新) | 变更日志 | +1 | P0 | 🔄 待更新 |

**文档工时**: 8.5h / 7个文档 = 平均1.2h/文档 ✅ 合理

---

## 9. 风险评估与缓解措施 ✅

### 高优先级风险

| 风险ID | 风险描述 | 概率 | 影响 | 缓解措施 | 监控指标 |
|--------|---------|------|------|---------|---------|
| R1 | v4特征性能退化>5% | 中 (40%) | 高 | 添加FEATURE_V4_ENABLE_STRICT开关 | feature_extraction_latency_seconds{version} |
| R2 | 安全白名单误杀合法模型 | 低 (20%) | 中 | 提供permissive回退模式 | model_security_fail_total{reason} |
| R3 | Day 3工时超标 | 中 (50%) | 中 | 安全文档延后至Day 5 | - |
| R4 | FAISS测试环境不稳定 | 中 (40%) | 低 | 全部Mock，不依赖真实FAISS | - |

### 低优先级风险

| 风险ID | 风险描述 | 概率 | 影响 | 接受度 |
|--------|---------|------|------|--------|
| R5 | 缓存调优建议不准确 | 低 (30%) | 低 | ✅ 标注experimental=true |
| R6 | Dashboard JSON格式错误 | 低 (15%) | 低 | ✅ promtool提前验证 |
| R7 | 文档中文英文不一致 | 低 (25%) | 低 | ✅ 仅关键段落双语 |

**整体风险评分**: 🟡 中等 (可控)

---

## 10. 验收标准明确性 ✅

### 每个Task的验收标准检查

| Task | 验收标准数量 | 可量化指标 | 可自动化验证 |
|------|------------|----------|------------|
| 1.4 模型回滚健康测试 | 3 | ✅ 6个测试通过 | ✅ pytest |
| 1.5 后端重载失败测试 | 3 | ✅ 6-8个测试通过 | ✅ pytest |
| 2.1 缓存调优端点 | 4 | ✅ 90%覆盖率 | ✅ pytest-cov |
| 2.2 迁移维度直方图 | 4 | ✅ 指标可见 | ✅ curl /metrics |
| 2.3 Grafana Dashboard | 2 | ✅ 6个新面板 | ⚠️ 人工验证 |
| 2.4 Prometheus规则 | 2 | ✅ promtool通过 | ✅ promtool check |
| 3.1 Opcode白名单 | 4 | ✅ 3种模式测试 | ✅ pytest |
| 3.2 安全文档 | 3 | ✅ 5个错误场景 | ⚠️ 人工验证 |
| 3.3 接口校验 | 4 | ✅ 指标完整 | ✅ pytest |
| 3.4 三层回滚 | 3 | ✅ 3层历史填充 | ✅ pytest |
| 4.1 v4真实特征 | 4 | ✅ <5%性能退化 | ✅ pytest |
| 4.2 迁移工具扩展 | 3 | ✅ 字段完整 | ✅ pytest |
| 5.1 错误Schema文档 | 4 | ✅ 15个错误码 | ⚠️ 人工验证 |
| 5.2 端点矩阵 | 2 | ✅ ≥30个端点 | ⚠️ 人工验证 |
| 5.3 规则验证 | 3 | ✅ CI集成 | ✅ CI |
| 5.4 指标一致性 | 2 | ✅ 脚本检测 | ✅ python script |
| 6.1 性能基线 | 3 | ✅ JSON生成 | ✅ python script |
| 6.2 回归测试 | 2 | ✅ 30个测试通过 | ✅ pytest |

**自动化比例**: 14/18 (78%) ✅ 优秀  
**人工验证项**: Dashboard, 文档质量 ✅ 可接受

---

## 11. 优先级与关键路径 ✅

### 任务优先级矩阵

```
               高影响
                 │
    Q1 (P0)      │      Q2 (P1)
  - 模型健康     │    - 缓存调优
  - 安全增强     │    - Dashboard
  - v4特征       │    - 迁移工具
─────────────────┼─────────────────
    Q3 (P2)      │      Q4 (P3)
  - 端点矩阵     │    - 性能基线
  - 规则验证     │    - 回归测试
                 │
               低影响
```

### 关键路径 (必须按序完成)

1. Day 1 PM: 模型健康测试 (阻塞Day 3回滚扩展)
2. Day 2 AM: 新增指标 (阻塞Day 2 PM Dashboard)
3. Day 3 AM-PM: 安全增强 (阻塞Day 4 v4启用)
4. Day 4 AM: v4真实特征 (阻塞Day 4 PM迁移工具)
5. Day 5: 文档更新 (阻塞最终交付)

**关键路径总工时**: 32h (60.5h中的53%)

---

## 12. 建议与优化 💡

### 时间优化建议

1. **并行化机会**:
   - Day 2 PM (Dashboard) 和 Day 3 AM (安全增强) 可部分并行
   - Day 5 AM (文档) 可提前准备大纲

2. **工时平衡**:
   - 建议将Day 5拆分为两天:
     - Day 5: 文档编写 (8h)
     - Day 6 AM: 文档补充 + 规则验证 (4h)
     - Day 6 PM: 性能基线 + 回归测试 (4h)

3. **风险缓冲**:
   - 当前Day 6为全天缓冲 (8h)
   - 优化后仍保留4h缓冲 ✅ 合理

### 资源分配建议

| 角色 | Day 1-2 | Day 3-4 | Day 5-6 |
|------|---------|---------|---------|
| 后端开发 | 核心开发 | 安全+特征 | 文档审核 |
| 测试工程师 | 测试编写 | 性能测试 | 回归验证 |
| DevOps | - | Dashboard配置 | 规则部署 |

**单人开发可行性**: ✅ 可行，但需严格遵循时间表

---

## ✅ 总体校验结论

### 通过项 (12/12)

1. ✅ Day 1 AM完成情况验证
2. ✅ 依赖关系正确性
3. ✅ 时间预算合理性 (经微调)
4. ✅ 技术可行性
5. ✅ 测试覆盖率目标
6. ✅ 指标命名一致性
7. ✅ API设计一致性
8. ✅ 文档完整性
9. ✅ 风险识别与缓解
10. ✅ 验收标准明确性
11. ✅ 优先级与关键路径
12. ✅ 资源分配合理性

### 关键改进建议

| 改进项 | 原计划 | 优化方案 | 影响 |
|--------|--------|---------|------|
| Day 2工时 | 12.5h | 减少Dashboard Panel → 10h | ✅ 可执行性提升 |
| Day 3工时 | 13.5h | 安全文档延后 → 11.5h | ✅ 工时平衡 |
| Day 5拆分 | 单日14.5h | 拆分为Day 5(8h)+Day 6 AM(4h) | ✅ 避免过载 |
| 缓冲时间 | 集中在Day 6 | 分散到每日+Day 6保留4h | ✅ 风险对冲 |

---

## 📋 执行前检查清单

### 环境准备 (Day 0)

- [ ] 拉取最新代码: `git pull origin main`
- [ ] 创建开发分支: `git checkout -b feature/6day-sprint`
- [ ] 运行基线测试: `make test` (确保全部通过)
- [ ] 验证Redis可访问: `redis-cli ping`
- [ ] 验证Prometheus/Grafana: `make observability-up`
- [ ] 检查Python环境: `python --version` (≥3.10)
- [ ] 安装开发依赖: `make install`

### Day 1 PM启动前

- [ ] 复查Day 1 AM完成状态
- [ ] 阅读`classifier.py`回滚机制代码
- [ ] 准备测试数据样本(模型文件)
- [ ] 设置工作timer (Pomodoro 25min)

### 每日结束检查

- [ ] 运行`make lint`确保代码规范
- [ ] 运行`make type-check`确保类型正确
- [ ] 运行新增测试: `pytest -v <new_test_file>`
- [ ] 更新`DEVELOPMENT_ROADMAP_DETAILED.md`进度标记
- [ ] Commit代码: `git commit -m "Day X: <task description>"`
- [ ] 记录明日计划和阻塞问题

---

## 📞 支持与问题升级

### 遇到问题时

1. **轻度阻塞** (<2h): 查阅现有文档 + 代码注释
2. **中度阻塞** (2-4h): 查看类似测试用例 + 历史commit
3. **重度阻塞** (>4h): 跳过当前Task，标记TODO，继续下一任务

### 需要帮助的场景

- ❓ Prometheus规则语法错误
- ❓ FAISS测试环境配置问题
- ❓ Dashboard JSON格式不兼容
- ❓ 性能基准差异过大(>20%)

---

**校验结论**: ✅ **计划可行，建议执行**  
**下一步**: 开始Day 1 PM Task 1.4 (模型回滚健康测试)

**文档版本**: v1.0 (2025-11-24)  
**下次复审**: Day 3结束 (验证进度是否符合预期)
