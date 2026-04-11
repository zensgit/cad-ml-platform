# CAD ML Platform - 6天开发计划实施清单

**计划周期**: 6个工作日
**开始日期**: 2025-11-24
**策略**: 稳定性优先 → 可观测性 → 安全 → 功能扩展

---

## 📋 准备工作（Day 0 - 开始前）

### ✅ 环境准备
- [ ] 创建 `config/feature_flags.py` 配置文件
- [ ] 建立 `scripts/daily_checkpoint.sh` 检查点脚本
- [ ] 创建 `scripts/track_progress.sh` 进度跟踪脚本
- [ ] 准备 v4 几何算法测试数据集
- [ ] 配置 CI 预检查钩子
- [ ] 设置优先级标签系统 (P0/P1/P2)
- [ ] 建立性能基线快照

### 验收标准
- 所有准备脚本可执行
- 测试数据集就位
- 基线性能数据记录完成

---

## Day 1: Phase A - 稳定性与补测（调整版）

### 🌅 AM Session (4h) - Phase A-1

#### 任务 1.1: Redis宕机孤儿清理测试
- [ ] 实现 `tests/unit/test_orphan_cleanup_redis_down.py`
  - 模拟 Redis 连接失败
  - 验证孤儿清理降级逻辑
  - 检查 `vector_orphan_total` 指标
- [ ] 扩展 `/api/v1/maintenance/orphans/cleanup` 错误处理
  - 返回结构化错误 `SERVICE_UNAVAILABLE`
  - 添加 fallback 提示信息

**验收**:
- 3个测试用例通过（连接失败/超时/部分失败）
- 指标正确递增
- 错误响应符合 `build_error` 格式

#### 任务 1.2: Faiss批量相似度降级测试
- [ ] 创建 `tests/unit/test_faiss_degraded_batch.py`
  - Faiss 不可用时批量查询降级到内存
  - 验证 `vector_query_backend_total{backend="memory_fallback"}`
  - 检查响应包含 `fallback=true` 标记
- [ ] 更新 `src/api/v1/vectors.py` batch_similarity
  - 增加 fallback 标记字段
  - 记录降级指标

**验收**:
- 4个测试用例通过（Faiss不可用/初始化失败/查询异常/混合场景）
- 降级标记正确返回
- 性能无明显劣化（< 10%）

#### 任务 1.3: 维护端点错误结构化
- [ ] 审查所有 `/api/v1/maintenance/*` 端点
- [ ] 统一错误响应格式（使用 `build_error`）
- [ ] 添加上下文字段（operation/resource_id/suggestion）

**验收**:
- 所有维护端点错误格式一致
- 单测覆盖错误路径

---

### 🌆 PM Session (4h) - Phase A-2

#### 任务 1.4: 模型回滚健康测试
- [ ] 扩展 `src/api/v1/health.py` 的 `/health/model` 端点
  - 增加 `rollback_level: int` 字段（0/1/2）
  - 增加 `last_error: str | None` 字段
  - 增加 `rollback_reason: str | None` 字段
- [ ] 创建 `tests/unit/test_model_rollback_health.py`
  - 模拟安全失败触发回滚
  - 验证 health 端点返回 rollback 信息
  - 检查 `model_health_checks_total{status="rollback"}` 指标

**验收**:
- 6个测试用例通过（无回滚/level1/level2/连续失败/恢复后清除）
- Health 响应包含完整回滚信息
- 指标正确分类 ok/absent/error/rollback

#### 任务 1.5: 后端重载失败测试（部分）
- [ ] 创建 `tests/unit/test_backend_reload_failures.py`
  - 无效后端名称
  - 缺少授权头
  - 后端初始化失败
- [ ] 更新 `vector_store_reload_total` 指标
  - 增加 `reason` 标签（invalid_backend/auth_failed/init_error）

**验收**:
- 3个核心测试用例完成
- 错误响应结构化
- 指标标签齐全

---

## Day 2: Phase A完成 + Phase B开始

### 🌅 AM Session (4h) - Phase A收尾

#### 任务 2.1: 完成后端重载失败测试
- [ ] 补充 `tests/unit/test_backend_reload_failures.py`
  - 并发重载冲突
  - 配置文件缺失
  - 权限不足场景

**验收**:
- 后端重载测试套件完整（6-8个用例）
- 覆盖率 ≥ 90%

#### 任务 2.2: 降级迁移链统计测试
- [ ] 创建 `tests/unit/test_migrate_downgrade_chain.py`
  - 模拟 v4→v3→v2→v1 降级链
  - 验证 `vector_migrate_total{status="downgraded"}` 计数
  - 检查维度递减统计

**验收**:
- 降级链路清晰可追踪
- 统计指标精确

#### 任务 2.3: 批量相似度空结果拒绝计数
- [ ] 扩展 `src/api/v1/vectors.py` batch_similarity
  - 检测所有ID无结果情况
  - 记录 `analysis_rejections_total{reason="batch_empty_results"}`
- [ ] 添加测试用例

**验收**:
- 空结果场景指标正确递增
- 不影响正常查询

---

### 🌆 PM Session (4h) - Phase B-1

#### 任务 2.4: 自适应缓存调优端点
- [ ] 创建 `src/api/v1/features.py` 新端点 `/api/v1/features/cache/tuning`
  - 输入模型：`CacheTuningRequest` (hit_rate/capacity/ttl/window_hours)
  - 输出模型：`CacheTuningRecommendation`
    - `recommended_capacity: int`
    - `recommended_ttl: int`
    - `confidence: float`
    - `reasoning: List[str]`
    - `experimental: bool = True`
- [ ] 实现调优算法
  ```python
  if hit_rate < 0.4:
      # 命中率过低，容量不足
      capacity *= 1.5
      reasoning.append("Low hit rate suggests insufficient capacity")
  elif 0.4 <= hit_rate < 0.7:
      # 中等命中率，调整TTL
      ttl = adjust_ttl_based_on_access_pattern()
      reasoning.append("Moderate hit rate, optimize TTL")
  elif hit_rate > 0.85:
      # 命中率过高，可能过度缓存
      capacity *= 0.8
      reasoning.append("High hit rate, capacity can be reduced")
  ```
- [ ] 添加指标 `feature_cache_tuning_requests_total{status}`
- [ ] 创建 `tests/unit/test_cache_tuning.py`

**验收**:
- 端点返回合理建议
- 边界case测试覆盖（0.35/0.39/0.40/0.70/0.85/0.90）
- 可解释性强（reasoning清晰）

#### 任务 2.5: 迁移维度差异直方图
- [ ] 添加指标 `vector_migrate_dimension_delta`
  ```python
  vector_migrate_dimension_delta = Histogram(
      "vector_migrate_dimension_delta",
      "Dimension difference during migration (positive = expansion, negative = reduction)",
      buckets=[-50, -20, -10, -5, 0, 5, 10, 20, 50, 100],
  )
  ```
- [ ] 在 `src/core/similarity.py` 或迁移逻辑中记录
- [ ] 导出到 `__all__`

**验收**:
- 指标在 `/metrics` 可见
- 记录维度变化分布

---

## Day 3: Phase B + Phase C-1

### 🌅 AM Session (4h) - Phase B-2

#### 任务 3.1: Grafana Dashboard框架（70%）
- [ ] 创建 `config/grafana/dashboard_main.json`
  - 面板1: 分析请求总览（成功率/QPS）
  - 面板2: 批量相似度延迟（p50/p95/p99）
  - 面板3: 特征缓存命中率
  - 面板4: 模型健康状态
  - 面板5: 向量存储统计
  - 面板6: 错误分布（按stage）
- [ ] 使用现有指标，不依赖Day 3-4新增指标

**验收**:
- Dashboard可导入Grafana
- 面板数据正确展示
- 时间范围选择器工作正常

#### 任务 3.2: Prometheus录制规则基础版
- [ ] 创建 `config/prometheus/recording_rules.yml`
  ```yaml
  groups:
    - name: cad_analysis_aggregations
      interval: 30s
      rules:
        - record: cad:analysis_requests:rate5m
          expr: rate(analysis_requests_total[5m])

        - record: cad:analysis_success_rate:5m
          expr: |
            rate(analysis_requests_total{status="success"}[5m])
            /
            rate(analysis_requests_total[5m])

        - record: cad:feature_cache_hit_rate:1h
          expr: |
            sum(rate(feature_cache_hits_total[1h]))
            /
            (sum(rate(feature_cache_hits_total[1h])) + sum(rate(feature_cache_miss_total[1h])))
  ```
- [ ] 运行 `promtool check rules config/prometheus/recording_rules.yml`

**验收**:
- promtool验证通过
- 规则语法正确

---

### 🌆 PM Session (4h) - Phase C-1

#### 任务 3.3: Pickle Opcode Audit模式
- [ ] 扩展 `src/ml/classifier.py` reload_model
  - 增加 `MODEL_OPCODE_MODE` 环境变量支持
    - `audit`: 扫描但不阻断，记录日志
    - `blocklist`: 当前行为，阻断危险opcode
    - `whitelist`: 仅允许安全opcode（预留）
  - 实现 opcode 扫描逻辑
    ```python
    import pickletools

    def scan_pickle_opcodes(file_path: Path) -> Dict[str, Any]:
        """扫描pickle文件中的opcode"""
        opcodes = []
        with file_path.open("rb") as f:
            for opcode, arg, pos in pickletools.genops(f):
                opcodes.append(opcode.name)

        dangerous = ["GLOBAL", "INST", "BUILD", "REDUCE"]
        found_dangerous = [op for op in opcodes if op in dangerous]

        return {
            "opcodes": opcodes,
            "dangerous": found_dangerous,
            "safe": len(found_dangerous) == 0
        }
    ```
  - 在 audit 模式记录但继续加载
  - 在 blocklist 模式拒绝并返回 `opcode_blocked`
- [ ] 添加指标
  ```python
  model_opcode_mode = Gauge(
      "model_opcode_mode",
      "Current opcode validation mode (0=audit, 1=blocklist, 2=whitelist)"
  )
  ```
- [ ] 更新 `model_security_fail_total{reason="opcode_blocked"}`

**验收**:
- Audit模式记录但不阻断
- Blocklist模式正确拒绝
- 日志包含具体opcode信息

#### 任务 3.4: 安全流程图文档
- [ ] 创建 `docs/SECURITY_MODEL_LOADING.md`
  - 模型加载安全流程图（Mermaid）
  - 三种模式对比表格
  - 快速排错指南（hash mismatch / opcode blocked / magic invalid）
- [ ] 更新 README 安全章节

**验收**:
- 流程图清晰易懂
- 排错指南可操作

#### 任务 3.5: v4几何算法预研
- [ ] 准备测试数据集
  - 空实体CAD文件
  - 单实体简单几何
  - 多实体复杂几何（高多样性）
- [ ] 调研几何细分算法
  - 选择库：OCC/trimesh/其他
  - 评估性能影响

**验收**:
- 测试数据集就绪
- 算法选型文档

---

## Day 4: Phase C-2 + Phase D-1

### 🌅 AM Session (4h) - Phase C-2 + Phase D基础

#### 任务 4.1: 接口校验扩展
- [ ] 扩展 `src/ml/classifier.py` reload_model
  - 检查模型对象属性数量（防止large attribute graph）
  - 检查 `__reduce__` 等魔术方法
  - 验证predict方法签名
- [ ] 添加指标
  ```python
  model_interface_validation_fail_total = Counter(
      "model_interface_validation_fail_total",
      "Model interface validation failures",
      ["reason"],  # large_graph|suspicious_method|invalid_signature
  )
  ```
- [ ] 创建测试 `tests/unit/test_model_interface_validation.py`

**验收**:
- 大对象图被拒绝
- 可疑方法被标记
- 指标正确记录

#### 任务 4.2: 回滚层级3实现
- [ ] 扩展模型快照系统
  ```python
  # 已有: _MODEL_PREV, _MODEL_PREV2
  # 新增: _MODEL_PREV3
  _MODEL_PREV3: Dict[str, Any] | None = None
  _MODEL_PREV3_HASH: str | None = None
  _MODEL_PREV3_VERSION: str | None = None
  _MODEL_PREV3_PATH: Path | None = None
  ```
- [ ] 更新回滚逻辑支持3级
- [ ] 创建测试 `tests/unit/test_model_rollback_level3.py`
  - 模拟4次加载，3次失败
  - 验证层级推进/回退

**验收**:
- 3级回滚逻辑正确
- 快照链完整
- 测试覆盖所有层级

#### 任务 4.3: v4 surface_count 基础版本
- [ ] 在 `src/core/feature_extractor.py` 实现
  ```python
  def extract_surface_count_v4(doc: CadDocument, mode: str = "simple") -> int:
      """提取表面数量

      Args:
          doc: CAD文档对象
          mode: simple | advanced（几何细分）
      """
      if mode == "simple":
          # 基于实体数量估算
          return len(doc.entities) * 6  # 假设每实体6面（立方体）
      else:
          # TODO: 实现高级几何细分
          raise NotImplementedError("Advanced surface counting not ready")
  ```
- [ ] 添加单测 `tests/unit/test_v4_surface_count.py`
  - 空实体 → 0
  - 单立方体 → 6
  - 复杂模型（已知面数）

**验收**:
- Simple模式工作正常
- 单测通过
- 性能可接受

---

### 🌆 PM Session (4h) - Phase D-1

#### 任务 4.4: v4 shape_entropy 平滑处理
- [ ] 实现Laplace平滑
  ```python
  def calculate_shape_entropy_v4(entities: List[Entity], smoothing: float = 1.0) -> float:
      """计算形状熵（带平滑）

      Args:
          entities: 实体列表
          smoothing: Laplace平滑参数（默认1.0）
      """
      from collections import Counter
      import math

      if not entities:
          return 0.0

      type_counts = Counter(e.type for e in entities)
      total = sum(type_counts.values())
      vocab_size = len(type_counts)

      # Laplace平滑
      entropy = 0.0
      for count in type_counts.values():
          p = (count + smoothing) / (total + smoothing * vocab_size)
          entropy -= p * math.log2(p)

      # 归一化到[0, 1]
      max_entropy = math.log2(vocab_size) if vocab_size > 1 else 1.0
      return entropy / max_entropy
  ```
- [ ] 添加单测验证
  - 单一类型 → 0.0（完全确定）
  - 均匀分布 → 接近1.0（最大不确定性）
  - 边界case（空列表/单元素）

**验收**:
- 熵值 ∈ [0, 1]
- 平滑避免NaN
- 单测边界case通过

#### 任务 4.5: v4性能对比测试
- [ ] 创建 `tests/performance/test_v4_performance.py`
  ```python
  @pytest.mark.slow
  def test_v4_extraction_overhead():
      """测试v4特征提取性能开销"""
      # 准备测试数据
      test_files = load_test_cad_files(count=20)

      # 测试v3
      v3_times = []
      for f in test_files:
          start = time.time()
          extract_features_v3(f)
          v3_times.append(time.time() - start)

      # 测试v4
      v4_times = []
      for f in test_files:
          start = time.time()
          extract_features_v4(f)
          v4_times.append(time.time() - start)

      v3_p95 = np.percentile(v3_times, 95)
      v4_p95 = np.percentile(v4_times, 95)
      overhead = (v4_p95 - v3_p95) / v3_p95

      assert overhead < 0.05, f"v4 overhead {overhead:.1%} exceeds 5% limit"
  ```
- [ ] 记录基线数据

**验收**:
- v4 提取耗时 ≤ v3 * 1.05
- 如超过5%，降级到simple模式

---

## Day 5: Phase D-2 + Phase E-1

### 🌅 AM Session (4h) - Phase D-2

#### 任务 5.1: 迁移工具preview端点
- [ ] 扩展 `src/api/v1/vectors.py` 或新建 `src/api/v1/migrate.py`
  ```python
  @router.get("/migrate/preview")  # 已更新: 预览迁移改为GET并挂载在 /api/v1/vectors/migrate/preview
  async def migrate_preview(
      from_version: str,
      to_version: str,
      sample_ids: List[str] = Query(default=[], max_length=10)
  ):
      """预览迁移影响

      Returns:
          - dimension_change: int (delta)
          - affected_vectors: int
          - top_dimension_changes: List[Tuple[slot_idx, old_val, new_val]]
          - estimated_time: float (seconds)
      """
  ```
- [ ] 实现预览逻辑（不实际迁移）
- [ ] 添加测试

**验收**:
- 预览不修改数据
- 维度变化统计正确
- 响应时间 < 2s

#### 任务 5.2: 迁移趋势端点
- [ ] 实现 `/vectors/migrate/trends`
  ```python
  @router.get("/vectors/migrate/trends")
  async def migrate_trends(window_hours: int = 24):
      """获取迁移趋势

      Returns:
          - total_migrations: int
          - success_rate: float
          - v4_adoption_rate: float
          - avg_dimension_delta: float
          - hourly_breakdown: List[dict]
      """
  ```
- [ ] 从指标查询历史数据
- [ ] 添加测试

**验收**:
- 趋势数据准确
- 可用于Dashboard集成

#### 任务 5.3: Dashboard补充Day 3-4新指标
- [ ] 更新 `config/grafana/dashboard_main.json`
  - 面板7: v4特征提取延迟对比（v3 vs v4）
  - 面板8: 迁移维度差异直方图
  - 面板9: 模型安全失败分布（按reason）
  - 面板10: 缓存调优建议历史
  - 面板11: Opcode模式当前值
  - 面板12: 漂移刷新触发饼图
- [ ] 验证所有面板数据源

**验收**:
- Dashboard完整度100%
- 所有面板正常展示

---

### 🌆 PM Session (4h) - Phase E-1

#### 任务 5.4: Prometheus Rules完整版
- [ ] 创建 `config/prometheus/alert_rules.yml`
  ```yaml
  groups:
    - name: cad_analysis_alerts
      rules:
        - alert: FeatureExtractionV4SlowDown
          expr: |
            histogram_quantile(0.95,
              rate(feature_extraction_latency_seconds_bucket{version="v4"}[5m])
            ) >
            histogram_quantile(0.95,
              rate(feature_extraction_latency_seconds_bucket{version="v3"}[5m])
            ) * 1.5
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "v4 feature extraction significantly slower than v3"

        - alert: ModelOpcodeBlocked
          expr: increase(model_security_fail_total{reason="opcode_blocked"}[5m]) > 0
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "Dangerous pickle opcode detected and blocked"

        - alert: CacheHitRateLow
          expr: cad:feature_cache_hit_rate:1h < 0.35
          for: 30m
          labels:
            severity: warning
          annotations:
            summary: "Feature cache hit rate below 35% for 30 minutes"
  ```
- [ ] 运行 `promtool check rules` 验证

**验收**:
- 所有告警规则语法正确
- 阈值合理
- 告警分级明确（critical/warning/info）

#### 任务 5.5: 文档全面更新
- [ ] 更新 `README.md`
  - 新增端点文档（cache/tuning, migrate/preview, migrate/trends）
  - 环境变量表格新增（MODEL_OPCODE_MODE, FEATURE_V4_*）
  - 指标索引更新（新增8个指标）
- [ ] 创建 `docs/ERROR_SCHEMA.md`
  ```markdown
  # 统一错误响应Schema

  | Field | Type | Description |
  |-------|------|-------------|
  | code | str | 错误代码（大写下划线） |
  | stage | str | 发生阶段 |
  | message | str | 人类可读描述 |
  | context | dict | 上下文信息 |

  ## 常见Stage
  - `routing`: 路由层
  - `batch_similarity`: 批量相似度
  - `vector_migrate`: 向量迁移
  - `feature_slots`: 特征槽位
  - `model_reload`: 模型重载
  - `security`: 安全验证
  - `drift`: 漂移检测
  ```
- [ ] 创建 `docs/METRICS_INDEX.md`
  - 所有指标列表
  - PromQL查询示例
  - 可视化建议
- [ ] 更新 `CHANGELOG.md`

**验收**:
- 文档无死链
- 代码示例可执行
- 渲染无格式错误

---

## Day 6: Phase E-2 + Phase F

### 🌅 AM Session (4h) - Phase E-2

#### 任务 6.1: Prometheus Rules回归验证
- [ ] 运行完整验证
  ```bash
  promtool check rules config/prometheus/recording_rules.yml
  promtool check rules config/prometheus/alert_rules.yml
  ```
- [ ] 修复任何语法错误
- [ ] 验证指标依赖存在

**验收**:
- 所有规则验证通过
- 无missing metric警告

#### 任务 6.2: CI预检查脚本
- [ ] 创建 `scripts/check_metrics_consistency.py`
  ```python
  #!/usr/bin/env python3
  """验证metrics定义与__all__导出一致性"""
  import ast
  import re
  from pathlib import Path

  def extract_metric_definitions(file_path: Path):
      """提取Counter/Histogram/Gauge定义"""
      with open(file_path) as f:
          content = f.read()

      # 正则匹配 metric_name = Counter(...) 等
      pattern = r'(\w+)\s*=\s*(Counter|Histogram|Gauge)\('
      return [m.group(1) for m in re.finditer(pattern, content)]

  def extract_all_exports(file_path: Path):
      """提取__all__列表"""
      with open(file_path) as f:
          tree = ast.parse(f.read())

      for node in ast.walk(tree):
          if isinstance(node, ast.Assign):
              for target in node.targets:
                  if isinstance(target, ast.Name) and target.id == '__all__':
                      return [elt.s for elt in node.value.elts]
      return []

  def main():
      metrics_file = Path("src/utils/analysis_metrics.py")

      defined = set(extract_metric_definitions(metrics_file))
      exported = set(extract_all_exports(metrics_file))

      missing = defined - exported
      extra = exported - defined

      if missing:
          print(f"❌ Metrics defined but not exported: {missing}")
          return 1
      if extra:
          print(f"⚠️  Metrics exported but not defined: {extra}")

      print(f"✅ All {len(defined)} metrics consistent")
      return 0

  if __name__ == "__main__":
      exit(main())
  ```
- [ ] 添加到 `.github/workflows/` 或 Makefile

**验收**:
- 脚本检测到不一致时退出码非0
- CI集成完成

#### 任务 6.3: 性能基线测试
- [ ] 运行 `scripts/performance_baseline.py`
  ```bash
  # 测试场景
  - 单文件分析 (小/中/大)
  - 批量相似度 (5/20/50 IDs)
  - 特征迁移 (v3→v4, 100 vectors)
  - 模型加载 (冷启动/热重载)
  ```
- [ ] 记录p50/p95/p99延迟
- [ ] 与Day 0基线对比

**验收**:
- 性能无回退 (< 5%)
- 生成性能报告表格

---

### 🌆 PM Session (4h) - Phase F

#### 任务 6.4: 回归测试套件
- [ ] 创建 `tests/regression/test_stateless_execution.py`
  ```python
  @pytest.mark.parametrize("order", [
      list(range(30)),
      list(range(30))[::-1],
      random.sample(range(30), 30)
  ])
  def test_critical_path_random_order(order):
      """随机顺序执行关键测试，验证无状态耦合"""
      tests = load_critical_tests()  # 30个核心测试
      for i in order:
          run_test_isolated(tests[i])
  ```
- [ ] 运行3次验证无顺序依赖

**验收**:
- 随机顺序测试全部通过
- 无状态泄漏

#### 任务 6.5: 缓冲与延后任务评估
- [ ] 评估时间余量
- [ ] 如有余量，实现可选任务：
  - [ ] Drift baseline 导出/导入端点
    ```python
    @router.post("/drift/baseline/export")
    async def export_baseline():
        """导出当前baseline快照"""

    @router.post("/drift/baseline/import")
    async def import_baseline(data: BaselineSnapshot):
        """导入baseline快照"""
    ```
  - [ ] Vector backend reload 安全token
    ```python
    @router.post("/vectors/backend/reload")
    async def reload_backend(
        backend: str,
        token: str = Header(None, alias="X-Admin-Token")
    ):
        """重载向量后端（需要管理员token）"""
        if token != os.getenv("ADMIN_TOKEN"):
            raise HTTPException(403, "Invalid admin token")
    ```

**验收**:
- 如实现，测试覆盖齐全
- 如未实现，记录到下个迭代

#### 任务 6.6: 最终验证与文档
- [ ] 运行完整测试套件
  ```bash
  pytest -v --cov=src --cov-report=html
  ```
- [ ] 生成覆盖率报告
- [ ] 更新 `IMPLEMENTATION_RESULTS.md`（开发成果文档）
- [ ] Git提交整理

**验收**:
- 测试通过率 100%
- 覆盖率 ≥ 85%
- 成果文档完整

---

## 📊 每日检查点

每天下午4点执行：
```bash
./scripts/daily_checkpoint.sh
```

输出内容：
- [ ] 当日任务完成率
- [ ] 测试通过数/失败数
- [ ] 代码覆盖率变化
- [ ] 新增指标数量
- [ ] 性能对比（如适用）
- [ ] 阻塞问题列表

---

## 🎯 优先级标签

**P0 (必须完成)**:
- 所有Phase A测试
- 安全增强核心功能 (Opcode audit/blocklist)
- v4基础版本 (simple模式)
- 核心文档更新 (README/ERROR_SCHEMA)

**P1 (强烈建议)**:
- Dashboard完整版
- 缓存调优端点
- Prometheus rules完整版
- 性能基线测试

**P2 (时间允许)**:
- v4 advanced模式
- Drift export/import
- 自动TTL调整PoC
- Backend reload安全token

---

## 🚨 风险缓解

### 如Day 4 v4实现延期
- **降级方案**: v4仅实现entropy优化，surface_count标记experimental
- **开关控制**: `FEATURE_V4_SURFACE_ALGORITHM=simple`

### 如安全白名单过严
- **回退策略**: `MODEL_OPCODE_MODE=audit` 模式运行1-2天观察
- **记录机制**: 所有阻断样本日志到 `logs/opcode_blocks.json`

### 如测试覆盖率不达标
- **最小阈值**: P0功能 ≥90%, P1功能 ≥80%, P2功能 ≥70%
- **豁免机制**: 性能测试、集成测试可标记 `@pytest.mark.slow` 不计入覆盖率

---

## ✅ 最终交付物清单

- [ ] 更新代码模块（~15个文件）
- [ ] 新增测试套件（~12个文件）
- [ ] 完整README更新
- [ ] CHANGELOG.md新版本段
- [ ] ERROR_SCHEMA.md文档
- [ ] METRICS_INDEX.md文档
- [ ] SECURITY_MODEL_LOADING.md文档
- [ ] Dashboard JSON (config/grafana/)
- [ ] Prometheus rules (config/prometheus/)
- [ ] 测试覆盖率报告 (htmlcov/)
- [ ] 性能基线报告 (reports/performance_baseline.md)
- [ ] 回归验证记录 (reports/regression_validation.md)
- [ ] **开发成果总结** (`IMPLEMENTATION_RESULTS.md`)

---

**Last Updated**: 2025-11-24
**Status**: Ready to start
**Estimated Completion**: 6 working days
