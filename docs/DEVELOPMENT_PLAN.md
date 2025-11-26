# CAD ML Platform 开发计划（更新版）

> 版本：v2.1
> 更新时间：2025-11-25
> 时间范围：预计 4–5.5 天有效开发（不含评审与部署）
> 目标：在保持最小迭代风险的前提下，完成 v4 真实特征、结构化错误统一、安全模式增强、可观测性与运维能力提升，并建立性能与稳定性基线。

---

## 1. 总览 (Executive Summary)

本周期聚焦六大阶段（拆分 Phase 1A/1B）：
1. v4 真实特征实现（surface_count + shape_entropy）
2. 错误响应结构化统一 + 迁移预览统计增强
3. 模型安全升级（opcode audit → whitelist 模式）
4. 自动恢复与降级可观测性、缓存调优与预热、并行批量相似度
5. 文档与一致性工具（README、指标校验脚本）
6. 压力与稳定性测试（内存、GC、并发锁、降级抖动）

核心原则：
- 最小增量：单次提交 ≤300 行核心逻辑 / ≤5 新文件
- 全面测试：每个功能配套单元或集成测试
- 结构化响应：所有错误形式统一（code / stage / message / context / timestamp）
- 可回滚：缓存调优与潜在风险功能提供回滚窗口

---

## 2. 范围说明 (In Scope / Out of Scope)

| 类别 | In Scope | Out of Scope |
|------|----------|--------------|
| 特征 | v4 surface/entropy 实现 | v5 拓扑高阶特征 |
| 安全 | opcode audit / whitelist | 模型签名哈希在线轮换系统 |
| 存储 | Faiss 降级与恢复机制 | 多数据中心复制 |
| 可观测性 | 新指标、Dashboard 示例、规则样稿 | 完整报警体系部署 |
| 性能 | 并行批量相似度优化（只读） | GPU 加速 / SIMD 优化 |
| 运维 | 缓存调优 apply/rollback | 自动调优策略闭环 |
| 文档 | 指标/错误码全量 README，CHANGELOG | 多语言文档翻译 |
| 稳定性 | 压力 & 并发测试脚本 | 持续性能基线平台 |

---

## 2.1 阶段依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                        开发阶段依赖图                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Phase 1A ──┐                                                  │
│   (v4特征)    ├──► Phase 2 ──► Phase 3 ──┐                      │
│   Phase 1B ──┘    (安全模式)   (自动恢复)  │                      │
│   (错误统一)                              │                      │
│                                          ▼                      │
│                                      Phase 4 ──► Phase 5 ──► Phase 6
│                                      (缓存/并行)  (文档)     (压力测试)
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  ● Phase 1A/1B 可并行开发（无依赖）                               │
│  ● Phase 2 依赖 Phase 1B（错误码定义）                            │
│  ● Phase 3 依赖 Phase 2（指标框架）                               │
│  ● Phase 4 依赖 Phase 3（降级状态）                               │
│  ● Phase 5 依赖 Phase 1-4（文档汇总）                             │
│  ● Phase 6 依赖全部（集成验证）                                   │
└─────────────────────────────────────────────────────────────────┘
```

**关键路径**：Phase 1A/1B → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6

**并行机会**：
- Phase 1A 与 Phase 1B 可同时进行
- Phase 5 文档可与 Phase 4 后半段并行

---

## 3. 阶段计划与验收标准

### Phase 1A：v4 真实特征实现（状态：✅ 已完成）
- 内容：
  - surface_count：统计几何对象或面片数量（DXF/STL/STEP/IGES 适配器扩展）
  - shape_entropy：基于面/实体类型频率熵 (H = -Σ p_i log p_i)，Laplace 平滑：p_i = (freq_i + 1) / (N + K)
  - 归一化：H_norm = H / log(K_eff)，K_eff 为不同类型总数
- 边界条件：

| 输入场景 | surface_count | shape_entropy | 说明 |
|---------|---------------|---------------|------|
| 空文档 | 0 | 0.0 | 无几何数据 |
| 单一类型（N个） | N | 0.0 | 确定性最大，熵为零 |
| 均匀分布（K类） | N | ≈1.0 | 最大不确定性 |
| 极端偏斜（90%单类） | N | <0.5 | 低多样性 |
| 高多样性（10+类均匀） | N | >0.9 | 接近理论上界 |
- 测试文件：`tests/unit/test_feature_extractor_v4_real.py`
- 测试用例 ≥12：
  1. 空文档
  2. 单类型
  3. 均匀多类型
  4. 极端偏斜（单一占 90%）
  5. 大量重复
  6. 高类型多样性
  7. 异常输入（缺失字段）
  8. 并发提取一致性（多线程）
  9. 熵结果 ∈[0,1]
  10. 性能对比 v3（p95 耗时差 ≤25%）
  11. 特征升级/降级对维度变动正确
  12. 适配器失败回退
- 验收：
  - v4 输出 24 维（保持向量长度稳定）：22 geometric + 2 semantic
  - 熵不出现 NaN / Inf
  - feature_extraction_latency_seconds{version="v4"} 样本数 ≥ 测试调用数
  - p95 延迟差 ≤25%

### Phase 1B：错误结构化统一 + Preview 增强（状态：✅ 已完成）
- 统一模型重载错误：size_exceeded, magic_invalid, hash_mismatch, opcode_blocked → build_error()
- 新增错误码：`OPCODE_WHITELIST_VIOLATION`（预留）、`CACHE_TUNING_SNAPSHOT_ACTIVE`、`CACHE_TUNING_ROLLBACK_EXPIRED`
- Migration Preview：
  - 新字段：`avg_delta`, `median_delta`, `warnings[]`
  - 负向维度变化占比 >50% → 添加 `large_negative_skew`
  - 正向大幅增长（>100% 增量）样本占比 >30% → 添加 `growth_spike`
- 测试：
  - `tests/unit/test_model_reload_errors_structured.py`
  - `tests/unit/test_migration_preview_stats.py`
- 验收：
  - 所有重载失败 HTTP 响应一致结构
  - Preview 返回统计字段准确（人工设定样本）

### Phase 2：安全模式扩展（opcode audit/whitelist）（状态：✅ 已完成）
- 模式：`MODEL_OPCODE_MODE` = blacklist | audit | whitelist
- 数据结构：
  - `_OPCODE_AUDIT_SET: set[str]`
  - `_OPCODE_AUDIT_COUNT: dict[str,int]`
  - `_OPCODE_AUDIT_SAMPLES: int`
- 新端点：`GET /api/v1/model/opcode-audit`
- 指标：
  - `model_opcode_audit_total{opcode="..."}`
  - `model_security_fail_total{reason="opcode_whitelist_violation"}`
- 测试：`tests/unit/test_model_opcode_modes.py`
  - audit 模式记录且不阻断
  - whitelist 拒绝非白名单
  - 模式切换后 audit 集合数据保留
- 验收：
  - 三模式行为与预期
  - 拒绝时结构化错误响应包含 disallowed_opcode

### Phase 3：自动恢复与降级可观测性（状态：✅ 已完成）
- 标志：
  - `_FAISS_RECOVERY_LOCK`
  - `_FAISS_MANUAL_RECOVERY_IN_PROGRESS`
- 参数：
  - `FAISS_RECOVERY_INTERVAL_SECONDS=300`
  - `FAISS_RECOVERY_MAX_BACKOFF=3600`
  - `FAISS_RECOVERY_BACKOFF_MULTIPLIER=2`
- 指标：
  - `degraded_duration_seconds` Gauge（降级→恢复归零）
  - `faiss_recovery_attempts_total{result="success|failure"}`
  - `similarity_degraded_total{event="degraded|restored"}`
- 新端点（可选）：`POST /api/v1/health/faiss/recover`（手动请求）
- 测试：`tests/unit/test_faiss_auto_recovery.py`
  - 降级→恢复路径
  - 手动恢复占用标志跳过自动探测
  - 抖动：恢复后立即再降级不超过历史上限
- 验收：
  - 历史 ≤10 条
  - restored 事件计数正确
  - 手动恢复期间自动探测暂停

### Phase 4：缓存调优、预热与并行批量相似度（状态：✅ 已完成）
- apply & rollback：
  - `PATCH /features/cache/apply?dry_run=true|false`
  - 快照结构：`_CACHE_TUNING_SNAPSHOT = {previous_capacity, previous_ttl, applied_at, expires_at}`
  - 回滚窗口：5 分钟
- 预热策略优先级：
  1. 最近 N 次命中（LRU 命中轨迹）
  2. 最近 M 次批量查询中的 reference IDs
  3. 低维度高频访问向量（访问频次/存储成本比）
- 预热指标：`feature_cache_prewarm_total{result="success|skipped|error"}`  
- 并行批量相似度：
  - 阈值：批量大小 > 50
  - 仅并行计算阶段（不并行写）
  - 指标：`batch_similarity_parallel_savings_seconds`
- 测试：
  - `tests/unit/test_cache_apply_rollback.py`
  - `tests/unit/test_cache_prewarm_priority.py`
  - `tests/unit/test_parallel_similarity_savings.py`
- 验收：
  - 回滚返回原参数
  - 并行节省时间（模拟延迟）>0
  - 预热后命中率提升（模拟数据）

### Phase 5：文档与一致性工具（详细）
- 目标：补齐端到端文档（API/错误码/指标/运维）、提供一致性检查工具与最小示例仪表盘/告警样例。
- 文档扩展（README.md）：
  - API 参考补齐（模型重载错误结构化、预览统计 avg/median/warnings、缓存调优/回滚/预热、opcode 审计/模式切换）
  - 运维指南（降级/恢复 Runbook、管理员令牌生成/轮换、Prometheus 规则与 Grafana 面板）
- 一致性工具：
  - scripts/metrics_consistency_check.py（已提交）：校验 metrics 是否完整导出至 __all__
  - scripts/generate_error_code_table.py：从 src/core/errors_extended.py 生成错误码表格（MD）并与 README 对齐
- 示例仪表盘/告警：
  - prometheus/rules/*.yaml（示例规则）
  - grafana/dashboards/observability.json（核心面板：降级、熵延迟、维度直方图、违规次数）
- 验收标准（DoD）：
  - README 新增章节完整、示例可执行（curl 或等价）
  - 一致性脚本在本地通过（CI 可 warning）
  - 示例规则/面板存在且语法有效（格式校验）

### Phase 6：压力与稳定性（详细）
- 目标：提供最小可行的压力/并发/内存与降级抖动验证工具与用例，形成可重复的稳定性检查。
- 并发/锁验证：
  - scripts/stress_concurrency_reload.py：并发触发 /api/v1/model/reload，验证 _MODEL_LOCK 有效性与 load_seq 单调性
  - 记录状态分布，确保无 data race
- 内存/GC 验证：
  - scripts/stress_memory_gc_check.py：提供 STRESS_STRICT=1 选项；默认仅报告
- 降级抖动：
  - 模拟 Faiss 可用/不可用切换并观察 degraded/restored 与持续时间指标
- 并行批量相似度（可选）：
  - 记录 analysis_parallel_savings_seconds 样例并给出 baseline 对比
- 集成测试：`tests/integration/test_stress_stability.py`
- 验收：
  - 并发 100 次重载无异常/死锁，_MODEL_LOAD_SEQ 单调递增
  - STRESS_STRICT=1 下 RSS 增长率 <10%（或输出关注提示）
  - 降级/恢复脚本能观察到事件/持续时间指标变化

---

## 4. 新增/修改文件清单（预估）

| 文件类型 | 预期路径 | 描述 |
|----------|----------|------|
| 新增测试 | tests/unit/test_feature_extractor_v4_real.py | v4 实特征与熵测试 |
| 新增测试 | tests/unit/test_model_reload_errors_structured.py | 重载错误结构化 |
| 新增测试 | tests/unit/test_migration_preview_stats.py | Preview 扩展 |
| 新增测试 | tests/unit/test_model_opcode_modes.py | opcode 三模式 |
| 新增测试 | tests/unit/test_faiss_auto_recovery.py | 自动恢复+锁 |
| 新增测试 | tests/unit/test_cache_apply_rollback.py | 缓存调优与回滚 |
| 新增测试 | tests/unit/test_cache_prewarm_priority.py | 预热策略 |
| 新增测试 | tests/unit/test_parallel_similarity_savings.py | 并行批量相似度 |
| 新增测试 | tests/unit/test_metrics_consistency.py | 指标一致性校验 |
| 新增测试 | tests/integration/test_stress_stability.py | 压力+并发集成 |
| 新脚本 | scripts/metrics_consistency_check.py | 指标对比工具 |
| 新脚本 | scripts/stress_migration.py | 迁移压力测试 |
| 新脚本 | scripts/stress_concurrency_reload.py | 并发 reload 压力 |
| 修改 | src/core/feature_extractor.py | v4 真实逻辑 |
| 修改 | src/api/v1/vectors.py | Preview 扩展 + 维度统计 |
| 修改 | src/ml/classifier.py | opcode 模式 + 审计收集 |
| 修改 | src/core/similarity.py | 降级恢复尝试逻辑 |
| 修改 | src/api/v1/health.py | 新指标暴露 & 恢复端点 |
| 修改 | src/utils/analysis_metrics.py | 新增若干指标 |
| 修改 | README.md | 文档更新 |
| 修改 | src/core/feature_cache.py | 调优 apply/rollback |
| 修改 | src/api/v1/features.py | 调优 apply 端点（PATCH） |

---

## 5. 结构化错误规范

统一格式：
```json
{
  "code": "OPCODE_WHITELIST_VIOLATION",
  "stage": "model_reload",
  "message": "Disallowed opcode detected during whitelist enforcement",
  "context": {
    "opcode": "GLOBAL",
    "mode": "whitelist",
    "allowed_set_size": 12
  },
  "timestamp": "2025-11-26T09:31:11.456Z"
}
```

新增错误码：
- OPCODE_WHITELIST_VIOLATION
- MODEL_INTERFACE_INVALID
- CACHE_TUNING_SNAPSHOT_ACTIVE
- CACHE_TUNING_ROLLBACK_EXPIRED
- FAISS_RECOVERY_CONFLICT
- MODEL_OPCODE_AUDIT_UNAVAILABLE

---

## 6. 新增指标列表（需加入 __all__）

| 指标名 | 类型 | Labels | 描述 |
|--------|------|--------|------|
| model_opcode_audit_total | Counter | opcode | audit 模式收集的各 opcode 频次 |
| model_security_fail_total | Counter | reason | 已有，新增 whitelist_violation |
| degraded_duration_seconds | Gauge | - | 当前降级持续秒数 |
| faiss_recovery_attempts_total | Counter | result | 自动恢复尝试结果（success/failure） |
| feature_cache_prewarm_total | Counter | result | 预热执行结果 |
| batch_similarity_parallel_savings_seconds | Histogram/Gauge | - | 并行批处理节约时间 |
| vector_migrate_dimension_delta | Histogram | - | 已有，继续使用 |
| stress_memory_growth_ratio | Gauge | phase | 压力测试前后内存增长比 |
| feature_extraction_latency_seconds (v4) | Histogram | version | 已有版本维度，将新增真实 v4 样本 |

---

## 7. 性能与稳定性基线

| 项目 | 当前估计 | 目标 / 阈值 |
|------|---------|-------------|
| v3 特征提取 p95 | 待采集 | v4 p95 ≤ v3 p95 *1.25 |
| 批量相似度（100 向量） | 待采集 | 并行节约 ≥10% |
| 模型重载并发（10 线程） | 无竞态 | load_seq 严格递增 |
| 内存增长 (迁移 1000 向量) | 待采集 | RSS 增幅 <50% 峰值，结束 <10% |
| 降级恢复时间 | 手动 | 自动恢复尝试间隔递增至 ≤1h |

---

## 8. 安全策略补充

- opcode 模式迁移路径：默认 blacklist → audit 收集 1–2 周 → whitelist（基于采集集合）。
- hash whitelist：sha256 校验失败 → 结构化 error + 提示更新配置。
- 管理员操作（reload/apply/rollback/recover）必须同时具备 API Key + X-Admin-Token。
- 定期提醒：响应中附 `next_rotation_hint`（若最后旋转时间 ≥7d）。

---

## 9. 缓存调优与回滚逻辑

流程：
1. GET /health/features/cache/tuning → 获取建议
2. PATCH /features/cache/apply（dry_run=true）→ 查看将变更
3. PATCH /features/cache/apply（dry_run=false）→ 应用并生成快照（5 min 窗口）
4. POST /features/cache/rollback → 回滚（若过期返回 CACHE_TUNING_ROLLBACK_EXPIRED）

快照结构：
```json
{
  "previous_capacity": 1000,
  "previous_ttl": 3600,
  "applied_at": "...",
  "expires_at": "...",
  "can_rollback_until": "..."
}
```

规则：
- 窗口内再次 apply → 拒绝（CACHE_TUNING_SNAPSHOT_ACTIVE）
- 回滚后可再次 apply
- 重启丢失快照（README 说明）

---

## 10. 预热策略

优先顺序：
1. 最近 N 次命中列表（LRU 命中轨迹）
2. 最近 M 次批量 similarity 的 reference IDs
3. 低维度且高频访问向量（命中 / 长度 比值）

执行：
- 限制最大预热数量 PREWARM_MAX（默认 500）
- 每项加载错误计数 feature_cache_prewarm_total{result="error"}

---

## 11. 压力测试脚本逻辑概述

`scripts/stress_migration.py`：
- 生成模拟向量（多版本）→ dry_run 迁移 → 正式迁移 → 记录耗时/成功率 → 输出 JSON 报告

`scripts/stress_concurrency_reload.py`：
- 多线程触发 reload（部分正常、部分故意破坏）→ 验证 load_seq 严格递增 → 汇总 last_error 统计

内存与 GC：
- 使用 `tracemalloc.start()` + RSS（/proc/self/statm 或 psutil）
- 前后差值、GC 次数与耗时统计（简单采样）

---

## 12. 测试矩阵总览

| 模块 | 文件 | 场景数 | 目的 |
|------|------|--------|------|
| v4 特征 | test_feature_extractor_v4_real.py | ≥12 | 真实熵+面计数边界 |
| 重载错误 | test_model_reload_errors_structured.py | 6 | 错误结构化 |
| 迁移预览 | test_migration_preview_stats.py | 5 | delta统计 + 警告 |
| opcode 模式 | test_model_opcode_modes.py | 6 | 三模式行为 |
| 自动恢复 | test_faiss_auto_recovery.py | 5 | 恢复/抖动/锁 |
| 缓存调优 | test_cache_apply_rollback.py | 6 | apply/rollback |
| 预热策略 | test_cache_prewarm_priority.py | 5 | 优先级 & 命中提升 |
| 并行批量 | test_parallel_similarity_savings.py | 3 | 节约指标 |
| 指标一致性 | test_metrics_consistency.py | 1 | 指标与 README 对齐 |
| 压力稳定 | test_stress_stability.py | 5 | 并发 & 内存 & 降级循环 |

---

## 13. 风险与缓解

| 风险 | 描述 | 缓解 |
|------|------|------|
| v4 熵计算复杂度过高 | 大文档面枚举耗时超预算 | 分层：先类型频率，后几何精细；必要时采样 |
| audit 数据丢失 | 重启清空内存集合 | 定期日志 dump（JSON）或后续持久化 |
| 自动恢复抖动 | 频繁降级与恢复 | 指数退避 + 手动恢复标志 |
| 并行批处理竞态 | 多线程访问共享结构 | 只读数据快照 + 不修改全局状态 |
| 调优误配置 | 容量/TTL 极端调整 | dry_run + 回滚窗口 |
| 回滚过期调用 | 用户延迟操作失败 | 明确错误码 + README 提示 |
| 内存膨胀 | 预热 + 压力测试引起 | PREWARM_MAX 上限 & RSS 校验 |

---

## 14. 提交与质量守则

- 每阶段功能完成 → 对应测试文件全部通过 → 再提交 PR
- PR 模板需附：
  - 变更摘要
  - 新增指标
  - 新增错误码
  - cURL 示例（若新增端点）
  - 风险与回滚策略
- 本周期不处理既有 121 个遗留失败测试（标记 legacy）

---

## 15. 时间表 (参考)

| Day | 上午 | 下午 |
|-----|------|------|
| Day 1 | Phase 1A | Phase 1B |
| Day 2 | Phase 2 | Phase 3 |
| Day 3 | Phase 4 (调优+预热) | Phase 4 (并行批处理) |
| Day 4 | Phase 5 (文档/一致性) | Phase 6 (压力/回归) |
| Day 5 (buffer) | 修复/调优 & 评审 | 发布准备 |

---

## 16. 启动指令与后续

- 开始 Phase 1A：实现 v4 真实特征 + 测试
- 若需先生成测试骨架，请执行：“开始 Phase 1A”
- 若需 opcode 白名单建议初始集合（基于 audit），请执行：“opcode 白名单”

---

## 17. v4 熵计算：数学定义与实现

### 17.1 数学定义

**香农熵（Shannon Entropy）**：
```
H = -Σ pᵢ · log(pᵢ)
```

**Laplace 平滑**（避免零概率）：
```
pᵢ = (freqᵢ + 1) / (N + K)

其中：
  - freqᵢ = 类型 i 的出现次数
  - N = 总实体数量 = Σ freqᵢ
  - K = 不同类型总数
```

**归一化**（映射到 [0, 1]）：
```
H_norm = H / log(K)

其中：
  - log(K) 为均匀分布时的理论最大熵
  - K = 1 时，H_norm = 0（单一类型无不确定性）
```

### 17.2 实现伪代码

```python
def compute_shape_entropy(type_counts: dict[str, int]) -> float:
    """计算形状类型分布熵，归一化到 [0, 1]"""
    if not type_counts:
        return 0.0

    K = len(type_counts)  # 类型数
    N = sum(type_counts.values())  # 总数

    if K == 1:
        return 0.0  # 单一类型，熵为零

    # Laplace 平滑概率
    probs = [(c + 1) / (N + K) for c in type_counts.values()]

    import math
    H = -sum(p * math.log(p) for p in probs)
    max_H = math.log(K)  # 理论最大熵

    return H / max_H if max_H > 0 else 0.0
```

### 17.3 示例计算

| 输入 | type_counts | K | N | H | H_norm |
|-----|-------------|---|---|---|--------|
| 单一类型 | {"plane": 100} | 1 | 100 | 0 | 0.0 |
| 二元均匀 | {"plane": 50, "cylinder": 50} | 2 | 100 | 0.693 | 1.0 |
| 二元偏斜 | {"plane": 90, "cylinder": 10} | 2 | 100 | 0.325 | 0.47 |
| 多类型均匀 | {"A": 25, "B": 25, "C": 25, "D": 25} | 4 | 100 | 1.386 | 1.0 |

---

## 18. 维护与回滚策略摘要

| 功能 | 回滚方式 | 记录点 |
|------|----------|--------|
| v4 特征 | 切换 FEATURE_VERSION 环境变量 | 无状态（向量存储需手动迁移） |
| 缓存调优 | rollback API | _CACHE_TUNING_SNAPSHOT |
| 降级模式 | 自动恢复或手动 recover | _DEGRADATION_HISTORY |
| opcode 模式 | 改 MODE 变量 | audit 集合保留 |

---

## 19. DONE 判定标准

满足以下全部即视为周期完成：
- v4 真实实现与测试通过
- 所有新增指标在 README、__all__、一致性脚本无差异
- 所有新增端点提供 cURL 示例
- 所有结构化错误符合统一格式
- 压力测试满足内存与错误率阈值
- 降级/恢复完整链路指标可见
- CHANGELOG 更新完毕

---

## 20. 附录：错误码总表（增量部分）

| Code | 描述 | 触发阶段 |
|------|------|----------|
| OPCODE_WHITELIST_VIOLATION | 非白名单 opcode | model_reload |
| MODEL_INTERFACE_INVALID | predict 接口不符合要求 | model_reload |
| CACHE_TUNING_SNAPSHOT_ACTIVE | 存在未回滚快照 | cache_tuning |
| CACHE_TUNING_ROLLBACK_EXPIRED | 回滚窗口过期 | cache_tuning |
| FAISS_RECOVERY_CONFLICT | 手动恢复冲突 | faiss_recovery |
| MODEL_OPCODE_AUDIT_UNAVAILABLE | audit 集合未初始化 | opcode_audit |

---

## 21. 环境变量汇总

| 变量名 | 默认值 | Phase | 描述 |
|--------|--------|-------|------|
| `FEATURE_VERSION` | v3 | 1A | 特征提取版本 (v1/v2/v3/v4) |
| `MODEL_OPCODE_MODE` | blacklist | 2 | opcode 检查模式 (blacklist/audit/whitelist) |
| `MODEL_OPCODE_WHITELIST` | (内置集合) | 2 | 白名单允许的 opcode 列表（逗号分隔） |
| `FAISS_RECOVERY_INTERVAL_SECONDS` | 300 | 3 | 自动恢复探测间隔（秒） |
| `FAISS_RECOVERY_MAX_BACKOFF` | 3600 | 3 | 恢复失败最大退避时间（秒） |
| `FAISS_RECOVERY_BACKOFF_MULTIPLIER` | 2 | 3 | 退避时间乘数 |
| `CACHE_ROLLBACK_WINDOW_SECONDS` | 300 | 4 | 缓存调优回滚窗口（秒） |
| `PREWARM_MAX` | 500 | 4 | 预热最大向量数 |
| `BATCH_SIMILARITY_PARALLEL_THRESHOLD` | 50 | 4 | 触发并行计算的批量大小阈值 |
| `BATCH_SIMILARITY_PARALLEL_WORKERS` | 4 | 4 | 并行批量计算的最大线程数 |
| `ADMIN_TOKEN` | (必须设置) | - | 管理员操作令牌 |
| `OPCODE_AUDIT_PERSIST_INTERVAL` | 3600 | 2 | audit 数据持久化间隔（秒，0=禁用） |
| `OPCODE_AUDIT_PERSIST_PATH` | /var/log/cad-ml/opcode_audit.jsonl | 2 | audit 数据持久化路径 |

---

## 22. API 端点汇总

### 22.1 新增端点

| 端点 | 方法 | Phase | 认证 | 描述 |
|------|------|-------|------|------|
| `/api/v1/model/opcode-audit` | GET | 2 | API Key | 查询 audit 收集结果 |
| `/api/v1/health/faiss/recover` | POST | 3 | API Key + Admin Token | 手动触发 Faiss 恢复 |
| `/api/v1/health/features/cache/apply` | PATCH | 4 | API Key + Admin Token | 应用缓存调优建议 |
| `/api/v1/health/features/cache/rollback` | POST | 4 | API Key + Admin Token | 回滚缓存配置 |
| `/api/v1/health/features/cache/prewarm` | POST | 4 | API Key | 触发缓存预热 |

### 22.2 修改端点

| 端点 | 方法 | Phase | 变更说明 |
|------|------|-------|---------|
| `/api/v1/vectors/migrate/preview` | GET | 1B | 已新增 avg_delta, median_delta, warnings 字段 (完成) |
| `/api/v1/model/reload` | POST | 1B | 错误响应结构化统一 |
| `/api/v1/health/faiss/health` | GET | 3 | 新增 recovery_status, last_recovery_attempt 字段 |

### 22.3 cURL 示例

**查询 opcode 审计数据：**
```bash
curl -H "X-API-Key: your_key" \
  http://localhost:8000/api/v1/model/opcode-audit
```

**应用缓存调优（dry_run）：**
```bash
curl -X PATCH \
  -H "X-API-Key: your_key" \
  -H "X-Admin-Token: your_admin_token" \
  "http://localhost:8000/api/v1/health/features/cache/apply?dry_run=true"
```

**手动触发 Faiss 恢复：**
```bash
curl -X POST \
  -H "X-API-Key: your_key" \
  -H "X-Admin-Token: your_admin_token" \
  http://localhost:8000/api/v1/health/faiss/recover
```

---

## 23. Prometheus 告警规则示例

```yaml
# prometheus/alerts/cad-ml-alerts.yaml
groups:
  - name: cad-ml-platform
    rules:
      # 向量存储降级告警
      - alert: VectorStoreDegraded
        expr: degraded_duration_seconds > 300
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Vector store degraded for > 5min"
          description: "Faiss unavailable, using memory fallback. Duration: {{ $value }}s"

      # 模型重载失败激增
      - alert: ModelReloadFailureSpike
        expr: rate(model_security_fail_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High rate of model reload failures"
          description: "Security validation failures: {{ $value }}/s"

      # opcode 白名单违规
      - alert: OpcodeWhitelistViolation
        expr: increase(model_security_fail_total{reason="opcode_whitelist_violation"}[5m]) > 0
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Opcode whitelist violation detected"
          description: "Unauthorized opcode in model file"

      # 缓存命中率过低
      - alert: LowCacheHitRatio
        expr: |
          (feature_cache_hits_total / (feature_cache_hits_total + feature_cache_miss_total)) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Feature cache hit ratio below 50%"
          description: "Consider adjusting cache capacity or TTL"

      # v4 特征提取延迟过高
      - alert: V4FeatureExtractionSlow
        expr: |
          histogram_quantile(0.95, rate(feature_extraction_latency_seconds_bucket{version="v4"}[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "v4 feature extraction p95 latency > 500ms"
          description: "Current p95: {{ $value }}s"

      # 内存使用过高（压力测试用）
      - alert: HighMemoryGrowth
        expr: stress_memory_growth_ratio > 1.5
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Memory growth exceeded 50% during stress test"
          description: "Growth ratio: {{ $value }}"

      # 自动恢复连续失败
      - alert: FaissRecoveryFailing
        expr: increase(faiss_recovery_attempts_total{result="failure"}[30m]) > 5
        for: 0m
        labels:
          severity: warning
        annotations:
          summary: "Faiss recovery failing repeatedly"
          description: "{{ $value }} failures in last 30 minutes"
```

---

## 24. 版本兼容性说明

| 变更 | 向后兼容 | 迁移说明 |
|------|---------|----------|
| v4 特征维度 | ✅ 保持24维 | 无需客户端修改 |
| Preview 新字段 | ✅ 新增字段 | 客户端可忽略新字段 |
| opcode audit 端点 | ✅ 新增端点 | 可选使用 |
| 错误响应结构化 | ⚠️ 字段变更 | 更新错误解析逻辑（新增 context, timestamp） |
| 缓存调优 API | ✅ 新增端点 | 可选使用 |
| 手动恢复端点 | ✅ 新增端点 | 可选使用 |
| 批量相似度并行 | ✅ 透明优化 | 无需客户端修改 |
| 预热功能 | ✅ 内部优化 | 无需客户端修改 |

### 24.1 错误响应迁移指南

**旧格式**（部分端点）：
```json
{"error": "Model file not found"}
```

**新格式**（统一）：
```json
{
  "code": "DATA_NOT_FOUND",
  "stage": "model_reload",
  "message": "Model file not found",
  "context": {"path": "/path/to/model.pkl"},
  "timestamp": "2025-11-25T10:30:00.000Z"
}
```

**客户端适配建议**：
```python
# 兼容新旧格式
def parse_error(response):
    data = response.json()
    if "code" in data:
        # 新格式
        return data["code"], data["message"], data.get("context", {})
    elif "error" in data:
        # 旧格式
        return "UNKNOWN", data["error"], {}
    else:
        return "UNKNOWN", str(data), {}
```

---

## 25. 并发控制与审计持久化

### 25.1 并发限制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIMILARITY_PARALLEL_WORKERS` | 4 | 批量相似度并行线程数 |
| 模型重载并发 | 受 `_MODEL_LOCK` 保护 | 同一时刻仅允许一个重载操作 |
| Faiss 恢复并发 | 受 `_FAISS_RECOVERY_LOCK` 保护 | 防止自动/手动恢复冲突 |
| 缓存调优并发 | 受 `_CACHE_TUNING_LOCK` 保护 | 防止并发 apply/rollback |

### 25.2 审计数据持久化

**目的**：避免重启丢失 opcode 审计数据

**机制**：
```python
# 定期持久化（默认每小时）
# 格式：JSON Lines
# 路径：OPCODE_AUDIT_PERSIST_PATH

# 示例输出 /var/log/cad-ml/opcode_audit.jsonl
{"timestamp": "2025-11-25T10:00:00Z", "opcodes": {"BININT": 150, "TUPLE": 80, ...}, "samples": 42}
{"timestamp": "2025-11-25T11:00:00Z", "opcodes": {"BININT": 312, "TUPLE": 156, ...}, "samples": 89}
```

**配置**：
```bash
# 启用持久化（默认禁用）
export OPCODE_AUDIT_PERSIST_INTERVAL=3600  # 每小时
export OPCODE_AUDIT_PERSIST_PATH=/var/log/cad-ml/opcode_audit.jsonl

# 禁用持久化
export OPCODE_AUDIT_PERSIST_INTERVAL=0
```

**重启恢复**：
- 启动时自动加载最近一条持久化记录
- 合并到内存集合，避免数据丢失

### 25.3 线程安全保证

```python
# 关键锁定义（src/ml/classifier.py, src/core/similarity.py）
_MODEL_LOCK = threading.Lock()           # 模型重载
_FAISS_RECOVERY_LOCK = threading.Lock()  # Faiss 恢复
_CACHE_TUNING_LOCK = threading.Lock()    # 缓存调优
_OPCODE_AUDIT_LOCK = threading.Lock()    # 审计数据写入

# 使用模式
with _MODEL_LOCK:
    # 原子操作
    pass
```

---

## 26. 测试执行顺序

### 26.1 CI 流水线建议顺序

```yaml
# .github/workflows/test.yml
jobs:
  test:
    steps:
      # 1. 快速反馈（<2分钟）
      - name: Unit Tests (Fast)
        run: pytest tests/unit/ -x --ignore=tests/unit/test_stress* -q

      # 2. 集成测试（<5分钟）
      - name: Integration Tests
        run: pytest tests/integration/ -x -q

      # 3. 压力测试（仅发布前，<10分钟）
      - name: Stress Tests
        if: github.event_name == 'release'
        run: |
          pytest tests/unit/test_stress* -v
          python scripts/stress_migration.py --vectors 1000
          python scripts/stress_concurrency_reload.py --threads 10
```

### 26.2 本地开发建议

```bash
# 开发阶段：仅运行相关测试
pytest tests/unit/test_feature_extractor_v4_real.py -v

# 提交前：运行完整单元测试
pytest tests/unit/ -x

# 发布前：运行全部测试
pytest tests/ -v --tb=short
```

---

*文档版本：v2.1 | 更新时间：2025-11-25*
