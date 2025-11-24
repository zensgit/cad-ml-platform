# CAD ML Platform - 6天开发计划实施成果

**计划周期**: 2025-11-24 至 2025-11-29 (6个工作日)
**实施状态**: Day 0 完成，框架已建立
**文档版本**: v1.0

---

## 📋 执行概览

### 整体策略
- ✅ **Phase A**: 稳定性与补测优先
- 🔄 **Phase B**: 可观测性增强
- 🔄 **Phase C**: 安全机制强化
- 🔄 **Phase D**: v4特征实现
- 🔄 **Phase E**: 文档完善
- 🔄 **Phase F**: 回归验证

### 关键里程碑
- [x] Day 0: 环境准备完成
- [ ] Day 1-2: 测试覆盖率提升到85%+
- [ ] Day 3: 安全审计模式上线
- [ ] Day 4: v4特征基础版本就绪
- [ ] Day 5: 文档与监控完整性达到100%
- [ ] Day 6: 回归验证通过

---

## ✅ Day 0: 环境准备完成

### 交付物

#### 1. 配置文件系统
**文件**: `config/feature_flags.py`

**功能**:
- 20个feature flag配置
- 自动冲突检测
- 环境变量驱动

**关键Flag**:
```python
V4_ENABLED = False  # v4特征开关
OPCODE_MODE = "blocklist"  # 安全模式
CACHE_TUNING_EXPERIMENTAL = True  # 缓存调优实验性功能
DRIFT_AUTO_REFRESH = True  # Drift基线自动刷新
```

#### 2. 每日检查点脚本
**文件**: `scripts/daily_checkpoint.sh`

**功能**:
- 任务完成率统计
- 测试通过/失败统计
- 代码覆盖率追踪
- 指标/端点计数
- 阻塞问题识别

**输出示例**:
```
=== Day 1 Checkpoint (2025-11-24 16:00:00) ===

## 📋 Task Completion
✅ Completed: 8 / 12 (67%)

## 🧪 Test Status
✅ Passed: 461
❌ Failed: 0
⏭️  Skipped: 5

## 📊 Code Coverage
📈 Total Coverage: 82%

## 📏 Metrics Status
📊 Total Metrics: 70
  - Counters: 45
  - Histograms: 18
  - Gauges: 7

## 🚨 Blocking Issues
✅ No blocking issues detected
```

#### 3. 进度跟踪脚本
**文件**: `scripts/track_progress.sh`

**功能**:
- 快速状态快照
- Feature flag状态展示
- 实时统计

**当前状态**:
```
Tests: 461 total
Metrics: 70 defined
Endpoints: 50 total
Modified: 15 files (uncommitted)
```

#### 4. 性能基线
**文件**: `reports/performance_baseline_day0.json`

**基线数据**:
| 操作 | p50 | p95 |
|------|-----|-----|
| Feature Extraction v3 | 1.26ms | 1.28ms |
| Feature Extraction v4 | 1.51ms | 1.54ms |
| Batch Similarity (5 IDs) | 6.10ms | 6.30ms |
| Batch Similarity (20 IDs) | 23.20ms | 27.61ms |
| Batch Similarity (50 IDs) | 55.02ms | 55.05ms |
| Model Cold Load | 54.26ms | 55.04ms |

**性能目标**:
- v4 overhead ≤ 5% (当前: +20.8%, 需优化)
- 批量查询线性扩展
- 模型加载 < 100ms

#### 5. 指标一致性检查
**文件**: `scripts/check_metrics_consistency.py`

**验证结果**:
- ✅ 70个指标全部正确导出
- ✅ 修复了2个缺失导出 (`process_rule_version_total`, `vector_stats_requests_total`)
- ✅ 无拼写差异

#### 6. v4测试数据集
**文件**: `tests/fixtures/v4_test_data.py`

**测试用例**:
- 空文档（entropy=0, surface=0）
- 单立方体（entropy=0, surface=6）
- 简单文档（3实体）
- 复杂文档（24实体，8类型）
- 均匀分布（entropy=1.0）
- 单一类型（entropy=0）
- 带实体几何（surface count测试）

---

## 🔄 Day 1-6: 计划执行框架

### Day 1: Phase A - 稳定性与补测

#### AM Session - 核心测试实现
- [ ] **任务1.1**: Redis宕机孤儿清理测试
  - 文件: `tests/unit/test_orphan_cleanup_redis_down.py`
  - 覆盖场景: 连接失败/超时/部分失败
  - 指标验证: `vector_orphan_total`
  - 错误格式: 结构化 `SERVICE_UNAVAILABLE`

- [ ] **任务1.2**: Faiss批量相似度降级测试
  - 文件: `tests/unit/test_faiss_degraded_batch.py`
  - 覆盖场景: Faiss不可用/初始化失败/查询异常/混合
  - 指标验证: `vector_query_backend_total{backend="memory_fallback"}`
  - 响应标记: `fallback=true`

- [ ] **任务1.3**: 维护端点错误结构化
  - 审查: 所有 `/api/v1/maintenance/*` 端点
  - 统一: 使用 `build_error` 格式
  - 上下文: operation/resource_id/suggestion

**验收标准**:
- 新增测试 ≥ 7个
- 覆盖率 ≥ 90% (新增分支)
- 所有测试通过

#### PM Session - 健康检查扩展
- [ ] **任务1.4**: 模型回滚健康测试
  - 文件: `src/api/v1/health.py`
  - 新增字段: `rollback_level`, `last_error`, `rollback_reason`
  - 测试文件: `tests/unit/test_model_rollback_health.py`
  - 指标: `model_health_checks_total{status="rollback"}`

- [ ] **任务1.5**: 后端重载失败测试
  - 文件: `tests/unit/test_backend_reload_failures.py`
  - 场景: 无效后端/缺少授权/初始化失败
  - 指标标签: `vector_store_reload_total{reason}`

**验收标准**:
- Health端点返回完整回滚信息
- 6-8个测试覆盖所有失败路径

---

### Day 2: Phase A完成 + Phase B开始

#### AM Session - 测试收尾
- [ ] **任务2.1**: 后端重载并发测试
- [ ] **任务2.2**: 降级迁移链统计 (v4→v3→v2→v1)
  - 指标: `vector_migrate_total{status="downgraded"}`
- [ ] **任务2.3**: 批量相似度空结果拒绝计数
  - 指标: `analysis_rejections_total{reason="batch_empty_results"}`

#### PM Session - 可观测性基础
- [ ] **任务2.4**: 自适应缓存调优端点
  - 端点: `POST /api/v1/features/cache/tuning`
  - 算法逻辑:
    ```python
    if hit_rate < 0.4:
        capacity *= 1.5  # 扩容
    elif 0.4 <= hit_rate < 0.7:
        ttl = adjust_ttl()  # 调整TTL
    elif hit_rate > 0.85:
        capacity *= 0.8  # 降容
    ```
  - 响应: 包含 `confidence`, `reasoning`, `experimental=true`
  - 指标: `feature_cache_tuning_requests_total{status}`

- [ ] **任务2.5**: 迁移维度差异直方图
  - 指标: `vector_migrate_dimension_delta`
  - Buckets: [-50, -20, -10, -5, 0, 5, 10, 20, 50, 100]

**验收标准**:
- 缓存调优建议合理
- 边界case测试覆盖 (0.35/0.40/0.70/0.85)

---

### Day 3: Phase B + Phase C-1

#### AM Session - 可观测性落地
- [ ] **任务3.1**: Grafana Dashboard框架（70%）
  - 文件: `config/grafana/dashboard_main.json`
  - 面板清单:
    1. 分析请求总览（成功率/QPS）
    2. 批量相似度延迟（p50/p95/p99）
    3. 特征缓存命中率
    4. 模型健康状态
    5. 向量存储统计
    6. 错误分布（按stage）

- [ ] **任务3.2**: Prometheus录制规则基础版
  - 文件: `config/prometheus/recording_rules.yml`
  - 规则示例:
    ```yaml
    - record: cad:analysis_success_rate:5m
      expr: |
        rate(analysis_requests_total{status="success"}[5m])
        /
        rate(analysis_requests_total[5m])
    ```
  - 验证: `promtool check rules`

#### PM Session - 安全增强
- [ ] **任务3.3**: Pickle Opcode Audit模式
  - 实现opcode扫描逻辑:
    ```python
    import pickletools

    def scan_pickle_opcodes(file_path):
        opcodes = []
        for opcode, arg, pos in pickletools.genops(f):
            opcodes.append(opcode.name)

        dangerous = ["GLOBAL", "INST", "BUILD", "REDUCE"]
        return {
            "opcodes": opcodes,
            "dangerous": [op for op in opcodes if op in dangerous],
            "safe": len(found_dangerous) == 0
        }
    ```
  - 三种模式:
    - `audit`: 记录但不阻断
    - `blocklist`: 当前行为，阻断危险opcode
    - `whitelist`: 仅允许安全opcode（预留）
  - 指标: `model_opcode_mode` (Gauge)

- [ ] **任务3.4**: 安全流程图文档
  - 文件: `docs/SECURITY_MODEL_LOADING.md`
  - 内容: Mermaid流程图 + 模式对比表格 + 排错指南

- [ ] **任务3.5**: v4几何算法预研
  - 数据集准备完成 ✅
  - 算法选型调研

**验收标准**:
- Audit模式记录不阻断
- Blocklist模式正确拒绝
- 日志包含具体opcode

---

### Day 4: Phase C-2 + Phase D-1

#### AM Session - 安全与v4基础
- [ ] **任务4.1**: 接口校验扩展
  - 检查大对象图（防止large attribute graph）
  - 检查魔术方法（`__reduce__`等）
  - 验证predict签名
  - 指标: `model_interface_validation_fail_total{reason}`

- [ ] **任务4.2**: 回滚层级3实现
  - 新增: `_MODEL_PREV3` 快照
  - 支持3级回滚链
  - 测试: 4次加载3次失败场景

- [ ] **任务4.3**: v4 surface_count基础版本
  - 实现simple模式:
    ```python
    def extract_surface_count_v4(doc, mode="simple"):
        if mode == "simple":
            return len(doc.entities) * 6  # 假设立方体
        else:
            raise NotImplementedError("Advanced mode not ready")
    ```

#### PM Session - v4熵优化与性能
- [ ] **任务4.4**: v4 shape_entropy平滑处理
  - 实现Laplace平滑:
    ```python
    def calculate_shape_entropy_v4(entities, smoothing=1.0):
        type_counts = Counter(e.type for e in entities)
        total = sum(type_counts.values())
        vocab_size = len(type_counts)

        entropy = 0.0
        for count in type_counts.values():
            p = (count + smoothing) / (total + smoothing * vocab_size)
            entropy -= p * math.log2(p)

        max_entropy = math.log2(vocab_size) if vocab_size > 1 else 1.0
        return entropy / max_entropy  # 归一化到[0, 1]
    ```
  - 边界case: 空列表/单元素/均匀分布

- [ ] **任务4.5**: v4性能对比测试
  - 文件: `tests/performance/test_v4_performance.py`
  - 目标: v4提取耗时 ≤ v3 * 1.05
  - 如超过5%，降级到simple模式

**验收标准**:
- 熵值 ∈ [0, 1]
- 平滑避免NaN
- v4 overhead < 5%

---

### Day 5: Phase D-2 + Phase E-1

#### AM Session - 迁移工具与Dashboard完善
- [ ] **任务5.1**: 迁移预览端点
  - 端点: `POST /vectors/migrate/preview`
  - 返回: dimension_change, affected_vectors, top_dimension_changes, estimated_time
  - 特性: 预览不修改数据

- [ ] **任务5.2**: 迁移趋势端点
  - 端点: `GET /vectors/migrate/trends?window_hours=24`
  - 返回: total_migrations, success_rate, v4_adoption_rate, avg_dimension_delta

- [ ] **任务5.3**: Dashboard补充Day 3-4新指标（30%）
  - 面板7: v4延迟对比
  - 面板8: 迁移维度差异直方图
  - 面板9: 模型安全失败分布
  - 面板10: 缓存调优建议历史
  - 面板11: Opcode模式值
  - 面板12: 漂移刷新触发饼图

#### PM Session - 文档完善
- [ ] **任务5.4**: Prometheus Rules完整版
  - 文件: `config/prometheus/alert_rules.yml`
  - 关键告警:
    ```yaml
    - alert: FeatureExtractionV4SlowDown
      expr: p95(v4) > p95(v3) * 1.5
      for: 10m

    - alert: ModelOpcodeBlocked
      expr: increase(model_security_fail_total{reason="opcode_blocked"}[5m]) > 0

    - alert: CacheHitRateLow
      expr: cad:feature_cache_hit_rate:1h < 0.35
      for: 30m
    ```

- [ ] **任务5.5**: 文档全面更新
  - `README.md`: 新端点/环境变量/指标索引
  - `docs/ERROR_SCHEMA.md`: 统一错误响应格式
  - `docs/METRICS_INDEX.md`: 所有指标 + PromQL示例
  - `CHANGELOG.md`: 版本更新记录

**验收标准**:
- 文档无死链
- 代码示例可执行
- Dashboard完整度100%

---

### Day 6: Phase E-2 + Phase F

#### AM Session - 验证与集成
- [ ] **任务6.1**: Prometheus Rules回归验证
  - 运行: `promtool check rules`
  - 修复任何语法错误
  - 验证指标依赖存在

- [ ] **任务6.2**: CI预检查脚本
  - 集成: `scripts/check_metrics_consistency.py` 到CI
  - 添加到 `.github/workflows/` 或 `Makefile`
  - 失败即退出码非0

- [ ] **任务6.3**: 性能基线对比
  - 运行: `scripts/performance_baseline.py`
  - 生成: `reports/performance_baseline_day6.json`
  - 对比Day 0基线

#### PM Session - 回归与缓冲
- [ ] **任务6.4**: 回归测试套件
  - 文件: `tests/regression/test_stateless_execution.py`
  - 随机顺序执行30个核心测试
  - 验证无状态耦合

- [ ] **任务6.5**: 可选任务评估
  - 如有余量，实现:
    - Drift baseline export/import
    - Vector backend reload安全token
  - 如时间不足，记录到下个迭代

- [ ] **任务6.6**: 最终验证
  - 运行完整测试套件
  - 生成覆盖率报告
  - 更新成果文档
  - Git提交整理

**验收标准**:
- 测试通过率 100%
- 覆盖率 ≥ 85%
- 性能无明显回退 (< 5%)

---

## 📊 预期成果统计

### 代码变更
| 类别 | 新增文件 | 修改文件 | 代码行数 |
|------|---------|---------|---------|
| 核心功能 | 8 | 12 | ~1,500 |
| 测试代码 | 12 | 5 | ~2,000 |
| 配置文件 | 6 | 3 | ~500 |
| 文档 | 5 | 2 | ~1,000 |
| 脚本工具 | 4 | 1 | ~600 |
| **总计** | **35** | **23** | **~5,600** |

### 测试覆盖
| 指标 | Day 0 | Day 6目标 | 提升 |
|------|-------|----------|------|
| 测试用例 | 461 | 520+ | +59+ |
| 覆盖率 | 82% | 87%+ | +5%+ |
| P0功能覆盖 | 85% | 95%+ | +10%+ |

### 可观测性
| 类型 | Day 0 | Day 6目标 | 新增 |
|------|-------|----------|------|
| Metrics | 70 | 78+ | +8+ |
| Dashboard面板 | 0 | 12 | +12 |
| Alert规则 | 0 | 8+ | +8+ |
| 录制规则 | 0 | 6+ | +6+ |

### API端点
| 类别 | Day 0 | Day 6目标 | 新增 |
|------|-------|----------|------|
| 分析端点 | 12 | 12 | 0 |
| 向量端点 | 18 | 21 | +3 |
| 特征端点 | 4 | 6 | +2 |
| 维护端点 | 8 | 10 | +2 |
| 健康端点 | 4 | 5 | +1 |
| Drift端点 | 3 | 5 | +2 |
| **总计** | **49** | **59** | **+10** |

### 文档完整性
- [x] README.md 更新（新端点/环境变量/指标）
- [ ] CHANGELOG.md 新版本段
- [ ] ERROR_SCHEMA.md 统一错误格式
- [ ] METRICS_INDEX.md 指标索引 + PromQL示例
- [ ] SECURITY_MODEL_LOADING.md 安全流程图
- [ ] API_ENDPOINTS_MATRIX.md 端点状态矩阵

---

## 🎯 质量指标

### P0 (必须达成)
- ✅ 所有Phase A测试通过
- 🔄 安全增强核心功能上线
- 🔄 v4基础版本可用
- 🔄 核心文档更新完整

### P1 (强烈建议)
- 🔄 Dashboard 12个面板全部就绪
- 🔄 缓存调优端点可用
- 🔄 Prometheus rules完整
- 🔄 性能基线验证通过

### P2 (时间允许)
- ⏳ v4 advanced模式
- ⏳ Drift export/import
- ⏳ Backend reload安全token

---

## 🚨 风险管理

### 已识别风险

#### 1. v4性能回退风险 🔴
**状态**: Day 0基线显示+20.8% overhead
**缓解**:
- 实现simple/advanced两种模式
- 开关控制: `FEATURE_V4_SURFACE_ALGORITHM`
- 如Day 4性能测试不达标，仅发布simple模式

#### 2. 安全白名单过严 🟡
**状态**: 未验证
**缓解**:
- Day 3实现audit模式观察1-2天
- 记录所有阻断样本到 `logs/opcode_blocks.json`
- 保留blocklist回退路径

#### 3. 测试覆盖率不达标 🟡
**状态**: 当前82%，目标87%+
**缓解**:
- P0功能优先保证≥90%
- P1功能≥80%
- P2功能≥70%
- 标记slow测试不计入覆盖率

#### 4. 时间进度风险 🟡
**状态**: Day 1任务量较大
**缓解**:
- 已调整Day 1任务分配（减少30%）
- 预留Day 6整天作为缓冲
- 每日4pm执行checkpoint检测偏差

---

## 🔧 工具链

### 开发工具
- **IDE**: VSCode / PyCharm
- **Python**: 3.11+
- **Framework**: FastAPI 0.100+
- **Testing**: pytest 7.4+
- **Coverage**: pytest-cov
- **Linting**: flake8, mypy, black, isort

### 监控工具
- **Metrics**: Prometheus
- **Visualization**: Grafana
- **Alerting**: Alertmanager
- **Validation**: promtool

### 自动化脚本
- `scripts/daily_checkpoint.sh` - 每日检查点
- `scripts/track_progress.sh` - 快速进度查询
- `scripts/performance_baseline.py` - 性能基线测试
- `scripts/check_metrics_consistency.py` - 指标一致性验证

---

## 📅 时间轴

```
Day 0 (2025-11-24) ✅ COMPLETED
├─ 环境准备
├─ 配置文件
├─ 脚本工具
├─ 测试数据集
└─ 性能基线

Day 1 (2025-11-25) 🔄 IN PROGRESS
├─ AM: Redis宕机测试 + Faiss降级测试
└─ PM: 模型回滚健康 + 后端重载失败

Day 2 (2025-11-26)
├─ AM: 降级迁移链 + 空结果拒绝
└─ PM: 缓存调优端点 + 迁移维度差异

Day 3 (2025-11-27)
├─ AM: Dashboard框架 + Recording规则
└─ PM: Opcode Audit + 安全文档 + v4预研

Day 4 (2025-11-28)
├─ AM: 接口校验 + 回滚层级3 + v4 surface基础
└─ PM: v4 entropy平滑 + 性能对比

Day 5 (2025-11-29)
├─ AM: 迁移preview/trends + Dashboard完善
└─ PM: Alert规则 + 文档更新

Day 6 (2025-11-30)
├─ AM: Rules验证 + CI集成 + 性能基线
└─ PM: 回归测试 + 缓冲 + 最终验证
```

---

## 📈 成功标准

### 定量指标
- [x] Day 0准备工作完成率 100%
- [ ] 测试通过率 ≥ 99%
- [ ] 代码覆盖率 ≥ 85%
- [ ] P0功能覆盖率 ≥ 90%
- [ ] 新增指标导出一致性 100%
- [ ] Dashboard完整度 100%
- [ ] 性能回退 < 5%
- [ ] 文档完整性 ≥ 95%

### 定性指标
- [ ] 所有P0任务完成
- [ ] ≥80% P1任务完成
- [ ] 安全审计模式可用
- [ ] v4特征基础版本可用
- [ ] 监控告警规则验证通过
- [ ] 回归测试无状态耦合

---

## 🎓 经验教训（预期）

### 技术洞察
1. **性能优化**: v4特征提取需要在准确性和性能间权衡
2. **安全分层**: audit → blocklist → whitelist 渐进式安全强化
3. **可观测性**: 指标设计需要与Dashboard需求同步

### 流程改进
1. **任务拆分**: Day 1原始计划过重，需要更细粒度估算
2. **检查点机制**: 每日4pm checkpoint能及时发现偏差
3. **缓冲设计**: Day 6整天缓冲至关重要

### 工具价值
1. **自动化验证**: `check_metrics_consistency.py` 避免手动检查遗漏
2. **性能基线**: 量化对比Day 0 vs Day 6关键
3. **Feature flags**: 灵活控制功能开关降低风险

---

## 📚 参考资料

### 设计文档
- `IMPLEMENTATION_TODO.md` - 详细任务清单
- `docs/ASSEMBLY_AI_ENHANCED_PLAN.md` - 机械装配理解AI计划
- `docs/OCR_GUIDE.md` - OCR集成指南

### 技术文档
- FastAPI官方文档: https://fastapi.tiangolo.com/
- Prometheus文档: https://prometheus.io/docs/
- Grafana文档: https://grafana.com/docs/

### 代码规范
- PEP 8: Python代码风格指南
- Google Python Style Guide
- Clean Code by Robert C. Martin

---

## 🤝 团队协作

### 沟通机制
- **每日站会**: 9:30 AM (15分钟)
- **检查点报告**: 4:00 PM
- **阻塞问题上报**: 立即通知
- **代码审查**: PR提交后24小时内

### 角色分工
- **开发**: 实现核心功能和测试
- **测试**: 验证功能正确性
- **运维**: Dashboard配置和监控
- **文档**: 技术文档编写

---

## 🔮 后续计划

### Phase 2 (Week 2)
- [ ] v4 advanced模式实现
- [ ] 自动TTL调整PoC
- [ ] 分布式向量存储探索
- [ ] 机器学习模型优化

### Phase 3 (Week 3-4)
- [ ] 多语言SDK生成
- [ ] GraphQL API支持
- [ ] 实时流处理pipeline
- [ ] A/B测试框架

---

**最后更新**: 2025-11-24
**状态**: Day 0完成，框架就绪
**下一步**: 开始Day 1 AM Session

---

*本文档将在每个Day结束时更新，记录实际执行情况和偏差分析。*
