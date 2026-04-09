# CAD ML Platform - 2026 Q2 详细开发方案

**编制日期**: 2026-04-09
**计划周期**: 12 周（2026-04-14 ~ 2026-07-03）
**当前版本**: v2.0.1
**目标版本**: v3.0.0

---

## 一、现状评估

### 已完成（可直接使用）

| 模块 | 代码量 | 状态 |
|------|--------|------|
| 零件分类器 V2/V6/V8/V16 | 1,231 行 | 生产就绪 |
| 混合分类器（5路融合） | 1,514 行 | 生产就绪 |
| 2D GNN 图纸分析 | 412 行 | 代码完成，**配置关闭** |
| 3D UV-Net 几何分析 | 270 行 | 生产就绪 |
| 多模态融合 | 335 行 | 生产就绪 |
| 6种融合策略 | 541 行 | 生产就绪 |
| 5种置信度校准 | 498 行 | 生产就绪 |
| 可解释性引擎 | 639 行 | 生产就绪 |
| 训练器 + 3个数据集 | 3,000+ 行 | 生产就绪 |
| 模型压缩（剪枝/量化/蒸馏） | 1,200+ 行 | 代码完成，**蒸馏配置关闭** |
| 漂移检测（KS/PSI/ADWIN） | 500+ 行 | 生产就绪 |
| 评估框架 | 1,000+ 行 | 生产就绪 |
| 超参搜索（Optuna） | 800+ 行 | 生产就绪 |
| 实验追踪 + 模型注册表 | 2,000+ 行 | 生产就绪 |
| 模型服务（REST + gRPC） | 1,500+ 行 | 生产就绪 |
| 推理 API（双层缓存+限流） | 600+ 行 | 生产就绪 |
| 向量相似度搜索（Qdrant+FAISS） | 1,131 行 | 生产就绪 |
| LLM 助手（Claude/GPT/Qwen） | 5,000+ 行 | 基础对话可用 |
| 多租户 + RBAC | 1,000+ 行 | 生产就绪 |
| SSE 流式响应 | 200+ 行 | 生产就绪 |
| Web UI（Vanilla JS + Tailwind） | 694 行 | MVP 级别 |
| Docker + K8s Helm | 多文件 | 生产就绪 |
| CI/CD（33个 GitHub Workflows） | 多文件 | 生产就绪 |
| 测试套件 | 673 个文件 | 覆盖良好 |

### 待启用/补齐

| 项目 | 位置 | 说明 |
|------|------|------|
| Graph2D 分类分支 | `config/hybrid_classifier.yaml` line 14 | `enabled: false` |
| 历史序列分类分支 | `config/hybrid_classifier.yaml` line 42 | `enabled: false` |
| 拒绝机制 | `config/hybrid_classifier.yaml` line 51 | `enabled: false` |
| 知识蒸馏 | `config/hybrid_classifier.yaml` line 85 | `enabled: false` |
| V4 几何算法 | `config/feature_flags.py` | 需 `FEATURE_V4_ENABLE=1` |
| 漂移自动调优 | `src/ml/monitoring/drift.py` | 缺少自适应阈值 |
| 模型回滚 Level 3 | `IMPLEMENTATION_TODO.md` Day 4 | 未实现 |
| Pickle 安全白名单模式 | `IMPLEMENTATION_TODO.md` Day 3 | 仅 blocklist 模式 |

---

## 二、总体架构演进

```
v2.0 (当前)                    v3.0 (目标)
┌─────────────┐               ┌─────────────────────────┐
│  REST API   │               │  REST API + GraphQL      │
│  26 端点    │               │  30+ 端点                │
├─────────────┤               ├─────────────────────────┤
│  基础 LLM   │     ──→      │  LLM + Function Calling  │
│  对话问答    │               │  CAD 分析 Copilot        │
├─────────────┤               ├─────────────────────────┤
│  分类+相似度 │               │  分类+相似度+成本估算     │
│  工艺推荐    │               │  工艺推荐+图纸Diff       │
├─────────────┤               ├─────────────────────────┤
│  手动监控    │               │  智能异常检测+自动修复    │
├─────────────┤               ├─────────────────────────┤
│  Vanilla JS │               │  React 前端              │
│  MVP UI     │               │  完整可视化平台           │
└─────────────┘               └─────────────────────────┘
```

---

## 三、阶段一：收尾巩固（第 1-3 周，4/14 ~ 5/2）

> 目标：将完成度从 80% 提升到 100%，零新功能开发，只启用和验证

### Sprint 1.1 — 合并当前分支 + 安全加固（W1: 4/14 ~ 4/18）

#### 任务 1.1.1：合并 feat/hybrid-blind-drift-autotune-e2e
- **负责人**: 1 人
- **工时**: 1 天
- **操作**:
  ```bash
  git checkout main
  git merge feat/hybrid-blind-drift-autotune-e2e
  make test-unit
  ```
- **验收**: CI 全绿，无回归

#### 任务 1.1.2：Pickle Opcode 安全审计模式
- **负责人**: 1 人
- **工时**: 2 天
- **修改文件**:
  - `src/ml/classifier.py` — 新增 `scan_pickle_opcodes()` 函数
  - `config/feature_flags.py` — `MODEL_OPCODE_MODE` 已存在，实现 audit/blocklist/whitelist 三模式
  - `src/utils/analysis_metrics.py` — 新增 `model_opcode_mode` Gauge 指标
- **新增文件**:
  - `tests/unit/test_pickle_opcode_audit.py`
- **实现细节**:
  ```python
  # src/ml/classifier.py 新增
  import pickletools

  def scan_pickle_opcodes(file_path: Path) -> dict:
      """扫描 pickle 文件中的 opcode，检测危险操作"""
      DANGEROUS_OPS = {"GLOBAL", "INST", "BUILD", "REDUCE", "STACK_GLOBAL"}
      opcodes = []
      with file_path.open("rb") as f:
          for opcode, arg, pos in pickletools.genops(f):
              opcodes.append(opcode.name)
      found = [op for op in opcodes if op in DANGEROUS_OPS]
      return {"total": len(opcodes), "dangerous": found, "safe": len(found) == 0}

  def load_model_safe(file_path: Path, mode: str = "blocklist") -> Any:
      """安全加载模型，根据 opcode 模式决定行为"""
      scan = scan_pickle_opcodes(file_path)
      if mode == "audit":
          if not scan["safe"]:
              logger.warning("Dangerous opcodes found (audit mode): %s", scan["dangerous"])
          return torch.load(file_path, weights_only=True)
      elif mode == "blocklist":
          if not scan["safe"]:
              raise SecurityError(f"Blocked: dangerous opcodes {scan['dangerous']}")
          return torch.load(file_path, weights_only=True)
      elif mode == "whitelist":
          SAFE_OPS = {"PROTO", "FRAME", "SHORT_BINUNICODE", "MEMOIZE", ...}
          unsafe = [op for op in scan["total"] if op not in SAFE_OPS]
          if unsafe:
              raise SecurityError(f"Whitelist violation: {unsafe}")
          return torch.load(file_path, weights_only=True)
  ```
- **测试用例**: 6 个（每种模式正常/异常各 1 个）
- **验收**: 三种模式切换正常，指标可见

#### 任务 1.1.3：模型回滚 Level 3
- **负责人**: 1 人
- **工时**: 1 天
- **修改文件**:
  - `src/ml/classifier.py` — 扩展 `_MODEL_PREV3` 快照槽位
  - `src/api/v1/health.py` — health 端点增加 `rollback_level` 字段
- **新增文件**:
  - `tests/unit/test_model_rollback_level3.py`
- **实现细节**:
  ```python
  # 已有: _MODEL_PREV, _MODEL_PREV2
  # 新增:
  _MODEL_PREV3: Dict[str, Any] | None = None
  _MODEL_PREV3_HASH: str | None = None
  _MODEL_PREV3_VERSION: str | None = None

  def _push_snapshot(model, hash_val, version):
      """级联推送快照: current→prev→prev2→prev3"""
      global _MODEL_PREV, _MODEL_PREV2, _MODEL_PREV3
      _MODEL_PREV3 = _MODEL_PREV2
      _MODEL_PREV3_HASH = _MODEL_PREV2_HASH
      _MODEL_PREV2 = _MODEL_PREV
      _MODEL_PREV2_HASH = _MODEL_PREV_HASH
      _MODEL_PREV = model
      _MODEL_PREV_HASH = hash_val
  ```
- **测试用例**: 8 个（4次加载3次失败、层级推进/回退、快照链完整性）
- **验收**: 连续 3 次失败后能回滚到第 3 级

#### 任务 1.1.4：统一错误响应格式
- **负责人**: 1 人
- **工时**: 1 天
- **修改文件**:
  - `src/api/v1/maintenance.py` — 所有端点使用 `build_error()`
  - `src/api/v1/vectors.py` — batch_similarity 增加 fallback 标记
- **验收**: 所有维护端点错误格式一致

---

### Sprint 1.2 — 启用已关闭功能（W2: 4/21 ~ 4/25）

#### 任务 1.2.1：启用 Graph2D 分类分支
- **负责人**: 1 人
- **工时**: 2 天
- **操作步骤**:
  1. 确认模型文件存在:
     ```bash
     ls models/graph2d_*.pth  # 应有 48+ 个变体
     ```
  2. 修改配置:
     ```yaml
     # config/hybrid_classifier.yaml line 14
     graph2d:
       enabled: true        # false → true
       model_path: models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth
       min_confidence: 0.6  # 初始设置较低阈值，观察效果
     ```
  3. 运行融合评估:
     ```bash
     make hybrid-blind-eval  # 运行盲评，对比开启前后准确率
     ```
  4. 调参（根据盲评结果调整 `min_confidence` 和融合权重）
- **新增文件**:
  - `tests/integration/test_hybrid_with_graph2d.py` — 端到端融合测试
- **验收**: 混合分类准确率提升 ≥ 1%（或不低于当前），无延迟回归

#### 任务 1.2.2：启用历史序列分类分支
- **负责人**: 1 人
- **工时**: 2 天
- **操作步骤**:
  1. 修改配置:
     ```yaml
     # config/hybrid_classifier.yaml line 42
     history_sequence:
       enabled: true        # false → true
       min_confidence: 0.5
     ```
  2. 验证 HPSketch 数据集加载:
     ```bash
     python -c "from src.ml.train.hpsketch_dataset import HPSketchDataset; print('OK')"
     ```
  3. 运行盲评并对比
- **验收**: 对有历史数据的图纸分类准确率提升

#### 任务 1.2.3：启用拒绝机制
- **负责人**: 1 人
- **工时**: 1 天
- **操作步骤**:
  ```yaml
  # config/hybrid_classifier.yaml line 51
  rejection:
    enabled: true           # false → true
    min_confidence: 0.4     # 低于此阈值的预测被拒绝
    max_entropy: 0.85       # 高熵（不确定性大）被拒绝
  ```
- **新增文件**:
  - `tests/unit/test_rejection_mechanism.py`
- **验收**: 低置信度预测返回 `"status": "rejected"` 而非错误结果

#### 任务 1.2.4：启用知识蒸馏 + 模型压缩
- **负责人**: 1 人
- **工时**: 2 天（含训练时间）
- **操作步骤**:
  1. 配置蒸馏:
     ```yaml
     # config/hybrid_classifier.yaml line 85
     distillation:
       enabled: true
       teacher_model: models/cad_classifier_v16_config.json
       temperature: 4.0
       alpha: 0.7
     ```
  2. 执行蒸馏训练:
     ```bash
     python scripts/train_classifier_v16.py \
       --distill \
       --teacher models/cad_classifier_v15_ensemble.pt \
       --temperature 4.0 \
       --alpha 0.7 \
       --epochs 50
     ```
  3. 对比推理延迟:
     ```bash
     make perf-resilience-benchmark  # 对比蒸馏前后
     ```
- **验收**: 模型体积减小 ≥ 30%，准确率下降 < 0.5%，延迟降低 ≥ 20%

---

### Sprint 1.3 — V4 算法 + 监控补齐（W3: 4/28 ~ 5/2）

#### 任务 1.3.1：V4 几何算法 simple 模式
- **负责人**: 1 人
- **工时**: 2 天
- **修改文件**:
  - `src/core/feature_extractor.py` — 新增 `extract_surface_count_v4()` 和 `calculate_shape_entropy_v4()`
- **新增文件**:
  - `tests/unit/test_v4_surface_count.py`
  - `tests/unit/test_v4_shape_entropy.py`
  - `tests/performance/test_v4_performance.py`
- **实现细节**:
  ```python
  # src/core/feature_extractor.py
  def extract_surface_count_v4(doc, mode: str = "simple") -> int:
      if mode == "simple":
          count = 0
          for entity in doc.entities:
              kind = entity.dxftype() if hasattr(entity, 'dxftype') else ""
              if kind in ("3DSOLID", "BODY"):
                  count += 6  # 立方体基准
              elif kind in ("CIRCLE", "ARC"):
                  count += 3  # 圆柱面估算
              elif kind in ("LINE", "LWPOLYLINE"):
                  count += 2  # 板件估算
              else:
                  count += 1
          return max(count, 1)
      else:
          raise NotImplementedError("Advanced surface counting: Phase 2")

  def calculate_shape_entropy_v4(entities: list, smoothing: float = 1.0) -> float:
      from collections import Counter
      import math
      if not entities:
          return 0.0
      type_counts = Counter(getattr(e, 'dxftype', lambda: 'UNKNOWN')() for e in entities)
      total = sum(type_counts.values())
      vocab_size = len(type_counts)
      if vocab_size <= 1:
          return 0.0
      entropy = 0.0
      for count in type_counts.values():
          p = (count + smoothing) / (total + smoothing * vocab_size)
          entropy -= p * math.log2(p)
      max_entropy = math.log2(vocab_size)
      return entropy / max_entropy if max_entropy > 0 else 0.0
  ```
- **性能要求**: V4 提取耗时 ≤ V3 × 1.05
- **验收**: 单测通过，性能无回退，特征值合理

#### 任务 1.3.2：Grafana Dashboard 完整版
- **负责人**: 1 人
- **工时**: 1 天
- **修改文件**:
  - `config/grafana/dashboards/dashboard_main.json` — 补充至 12 个面板
- **面板清单**:
  1. 分析请求 QPS + 成功率（已有）
  2. 批量相似度延迟 p50/p95/p99（已有）
  3. 特征缓存命中率（已有）
  4. 模型健康状态（已有）
  5. 向量存储统计（已有）
  6. 错误分布（按 stage）（已有）
  7. **新增**: V4 特征提取延迟对比
  8. **新增**: 迁移维度差异直方图
  9. **新增**: 模型安全失败分布
  10. **新增**: 混合分类器各分支贡献度
  11. **新增**: 漂移检测状态时间线
  12. **新增**: 拒绝率趋势
- **验收**: Dashboard 可导入 Grafana 且面板数据正确

#### 任务 1.3.3：Prometheus Rules 完整版
- **负责人**: 1 人
- **工时**: 1 天
- **修改文件**:
  - `config/prometheus/recording_rules.yml` — 补充缺失的聚合规则
  - `config/prometheus/alerting_rules.yml` — 补充 V4/蒸馏/拒绝相关告警
- **新增告警规则**:
  ```yaml
  - alert: HybridClassifierRejectionRateHigh
    expr: rate(analysis_rejections_total{reason="low_confidence"}[5m]) / rate(analysis_requests_total[5m]) > 0.15
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "混合分类器拒绝率超过 15%"

  - alert: DistilledModelAccuracyDrop
    expr: cad:classification_accuracy:1h < 0.98
    for: 30m
    labels:
      severity: critical
    annotations:
      summary: "蒸馏模型准确率低于 98%"
  ```
- **验证**: `promtool check rules config/prometheus/*.yml`
- **验收**: 全部规则语法验证通过

#### 任务 1.3.4：阶段一集成验证
- **负责人**: 全员
- **工时**: 1 天
- **操作**:
  ```bash
  # 1. 全量测试
  make test-all-local

  # 2. 覆盖率报告
  pytest -v --cov=src --cov-report=html

  # 3. Docker 构建验证
  docker-compose up -d
  curl http://localhost:8000/health

  # 4. 盲评对比报告
  make hybrid-blind-eval > reports/phase1_blind_eval.md
  ```
- **验收标准**:
  - 测试通过率 100%
  - 覆盖率 ≥ 85%
  - 混合分类准确率 ≥ 99.5%（启用新分支后）
  - Docker 启动正常，健康检查通过
  - 生成阶段一总结报告

### 阶段一交付物

| 交付物 | 文件 |
|--------|------|
| 安全加固代码 | `src/ml/classifier.py` |
| 启用配置变更 | `config/hybrid_classifier.yaml` |
| V4 算法 | `src/core/feature_extractor.py` |
| 完整 Dashboard | `config/grafana/dashboards/dashboard_main.json` |
| 完整告警规则 | `config/prometheus/alerting_rules.yml` |
| 新增测试 ~15 个文件 | `tests/unit/test_*.py` |
| 盲评报告 | `reports/phase1_blind_eval.md` |
| 蒸馏模型 | `models/cad_classifier_v16_distilled.pt` |

---

## 四、阶段二：核心能力扩展（第 4-8 周，5/5 ~ 5/30）

> 目标：新增 3 个核心功能，建立市场差异化

### Sprint 2.1 — 制造成本自动估算（W4-W5: 5/5 ~ 5/16）

#### 2.1.1 架构设计

```
用户上传 CAD 文件
      │
      ▼
┌─ 特征提取 ─────────────────────────────────┐
│  几何复杂度 ← feature_extractor.py (已有)    │
│  材料属性   ← manufacturing_data.yaml (已有) │
│  工艺路线   ← process_rules.yaml (已有)      │
└─────────────────────────────────────────────┘
      │
      ▼
┌─ 成本模型 (新增) ──────────────────────────────────────┐
│                                                         │
│  材料成本 = 体积 × 密度 × 单价 × 废料系数               │
│  加工成本 = Σ(工序时间 × 机时单价)                       │
│  工序时间 = f(几何复杂度, 工序类型, 精度等级)             │
│  设置成本 = 固定成本 / 批量大小                          │
│  总成本   = (材料 + 加工 + 设置) × 管理费率              │
│                                                         │
└─────────────────────────────────────────────────────────┘
      │
      ▼
  返回: 总成本、成本明细、成本区间（乐观/期望/悲观）
```

#### 2.1.2 新增文件清单

```
src/ml/cost/
├── __init__.py
├── estimator.py          # 核心成本估算引擎 (~400行)
├── material_cost.py      # 材料成本计算 (~150行)
├── machining_cost.py     # 加工成本计算 (~250行)
├── complexity_scorer.py  # 几何复杂度评分 (~200行)
└── models.py             # Pydantic 数据模型 (~100行)

src/api/v1/cost.py        # API 端点 (~150行)

config/cost_model.yaml    # 成本模型参数配置

tests/unit/test_cost_estimator.py
tests/unit/test_material_cost.py
tests/unit/test_machining_cost.py
tests/integration/test_cost_api.py
```

#### 2.1.3 核心代码设计

```python
# src/ml/cost/models.py
from pydantic import BaseModel
from typing import Optional

class CostEstimateRequest(BaseModel):
    file_path: Optional[str] = None       # CAD 文件路径
    file_bytes: Optional[bytes] = None    # 或直接传字节
    material: str = "steel"               # 材料类型
    batch_size: int = 1                   # 生产批量
    tolerance_grade: str = "IT8"          # 精度等级
    surface_finish: str = "Ra3.2"         # 表面粗糙度

class CostBreakdown(BaseModel):
    material_cost: float                  # 材料费
    machining_cost: float                 # 加工费
    setup_cost: float                     # 装夹/编程费
    overhead: float                       # 管理费
    total: float                          # 总计
    currency: str = "CNY"

class CostEstimateResponse(BaseModel):
    estimate: CostBreakdown               # 期望成本
    optimistic: CostBreakdown             # 乐观（-20%）
    pessimistic: CostBreakdown            # 悲观（+30%）
    process_route: list[str]              # 推荐工艺路线
    complexity_score: float               # 几何复杂度 0~10
    confidence: float                     # 估算置信度 0~1
    reasoning: list[str]                  # 可解释性说明
```

```python
# src/ml/cost/estimator.py
class CostEstimator:
    def __init__(self, config_path: str = "config/cost_model.yaml"):
        self.config = load_yaml(config_path)
        self.material_calc = MaterialCostCalculator(self.config["materials"])
        self.machining_calc = MachiningCostCalculator(self.config["machines"])
        self.complexity_scorer = ComplexityScorer()

    async def estimate(self, request: CostEstimateRequest) -> CostEstimateResponse:
        # 1. 提取几何特征（复用已有 feature_extractor）
        features = await self.extract_features(request)

        # 2. 计算几何复杂度
        complexity = self.complexity_scorer.score(features)

        # 3. 确定工艺路线（复用已有 process_rules）
        process_route = self.determine_process_route(
            material=request.material,
            complexity=complexity,
            volume=features.get("bounding_volume", 0)
        )

        # 4. 计算各项成本
        material = self.material_calc.calculate(
            material=request.material,
            volume=features.get("bounding_volume", 0),
            waste_factor=1.0 + complexity.stock_removal_ratio
        )
        machining = self.machining_calc.calculate(
            process_route=process_route,
            complexity=complexity,
            tolerance=request.tolerance_grade,
            surface=request.surface_finish
        )
        setup = self.config["setup_base_cost"] / max(request.batch_size, 1)

        # 5. 汇总
        overhead_rate = self.config.get("overhead_rate", 0.15)
        subtotal = material + machining + setup
        total = subtotal * (1 + overhead_rate)

        return CostEstimateResponse(
            estimate=CostBreakdown(
                material_cost=material,
                machining_cost=machining,
                setup_cost=setup,
                overhead=subtotal * overhead_rate,
                total=total
            ),
            optimistic=self._apply_factor(total, 0.8),
            pessimistic=self._apply_factor(total, 1.3),
            process_route=process_route,
            complexity_score=complexity.score,
            confidence=self._calc_confidence(features, complexity),
            reasoning=complexity.reasoning
        )
```

```python
# src/api/v1/cost.py
from fastapi import APIRouter, UploadFile, File, Form

router = APIRouter(prefix="/api/v1/cost", tags=["cost"])

@router.post("/estimate")
async def estimate_cost(
    file: UploadFile = File(...),
    material: str = Form("steel"),
    batch_size: int = Form(1),
    tolerance_grade: str = Form("IT8"),
    surface_finish: str = Form("Ra3.2"),
):
    """估算 CAD 零件的制造成本"""
    estimator = get_cost_estimator()
    request = CostEstimateRequest(
        file_bytes=await file.read(),
        material=material,
        batch_size=batch_size,
        tolerance_grade=tolerance_grade,
        surface_finish=surface_finish,
    )
    return await estimator.estimate(request)

@router.post("/batch-estimate")
async def batch_estimate_cost(requests: list[CostEstimateRequest]):
    """批量成本估算"""
    estimator = get_cost_estimator()
    return [await estimator.estimate(req) for req in requests]

@router.get("/materials")
async def list_materials():
    """返回支持的材料及其单价"""
    return get_cost_estimator().config["materials"]
```

#### 2.1.4 成本模型配置

```yaml
# config/cost_model.yaml
version: "1.0"

materials:
  steel:
    price_per_kg: 6.5       # CNY/kg
    density: 7850            # kg/m^3
    machinability: 0.6       # 0~1, 影响加工时间系数
  stainless_steel:
    price_per_kg: 22.0
    density: 7930
    machinability: 0.4
  aluminum:
    price_per_kg: 28.0
    density: 2700
    machinability: 0.85
  titanium:
    price_per_kg: 280.0
    density: 4500
    machinability: 0.25
  plastic_abs:
    price_per_kg: 18.0
    density: 1040
    machinability: 0.95

machines:
  cnc_3axis:
    hourly_rate: 80          # CNY/hr
    setup_time_min: 30       # 分钟
    applicable: [milling, drilling, tapping]
  cnc_5axis:
    hourly_rate: 200
    setup_time_min: 60
    applicable: [milling, contouring, impeller]
  cnc_lathe:
    hourly_rate: 60
    setup_time_min: 20
    applicable: [turning, boring, threading]
  wire_edm:
    hourly_rate: 120
    setup_time_min: 45
    applicable: [cutting, slotting]
  grinding:
    hourly_rate: 100
    setup_time_min: 25
    applicable: [surface_finish, precision]

setup_base_cost: 200         # CNY（编程+装夹基础费用）
overhead_rate: 0.15          # 管理费率 15%

# 公差等级对加工时间的影响系数
tolerance_factor:
  IT6: 2.0
  IT7: 1.5
  IT8: 1.0
  IT9: 0.8
  IT10: 0.6
  IT11: 0.5
  IT12: 0.4

# 表面粗糙度对加工时间的影响系数
surface_factor:
  Ra0.8: 2.5
  Ra1.6: 1.8
  Ra3.2: 1.0
  Ra6.3: 0.7
  Ra12.5: 0.5
```

#### 2.1.5 测试计划

| 测试 | 场景 | 验收 |
|------|------|------|
| 单元 | 简单立方体 steel | 成本 > 0，明细加总 = 总计 |
| 单元 | 批量 1 vs 100 | setup_cost 相差 100 倍 |
| 单元 | 不同材料 | titanium > stainless > steel |
| 单元 | 不同精度 | IT6 > IT8 > IT12 |
| 集成 | API 端点调用 | HTTP 200，响应格式正确 |
| 集成 | 批量估算 | 3 个文件并行处理 |
| 性能 | 单次估算延迟 | < 500ms |

- **工时**: W4 开发（5天），W5 前 2 天测试 + 调参
- **验收**: API 可用，估算结果合理，有可解释性

---

### Sprint 2.2 — LLM Function Calling + CAD Copilot（W5-W7: 5/14 ~ 5/30）

#### 2.2.1 架构设计

```
用户自然语言输入
      │
      ▼
┌─ Query Analyzer (已有) ──────────────────────┐
│  意图分类: 查询 / 分析 / 比较 / 成本 / 通用    │
└──────────────────────────────────────────────┘
      │
      ▼
┌─ LLM + Function Calling (新增) ──────────────┐
│                                               │
│  系统提示词 + 可用工具定义                      │
│       │                                       │
│       ▼                                       │
│  LLM 决策: 调用哪个工具、传什么参数              │
│       │                                       │
│       ▼                                       │
│  工具执行器: 调用内部 API                       │
│       │                                       │
│       ▼                                       │
│  LLM 汇总: 将工具结果转为自然语言               │
│                                               │
└───────────────────────────────────────────────┘
      │
      ▼
  SSE 流式返回（已有 streaming.py）
```

#### 2.2.2 新增文件清单

```
src/core/assistant/
├── function_calling.py    # 工具定义与执行器 (~350行)
├── tools/
│   ├── __init__.py
│   ├── classify_tool.py   # 分类工具 (~80行)
│   ├── similarity_tool.py # 相似度搜索工具 (~80行)
│   ├── cost_tool.py       # 成本估算工具 (~80行)
│   ├── feature_tool.py    # 特征提取工具 (~80行)
│   ├── process_tool.py    # 工艺推荐工具 (~80行)
│   ├── quality_tool.py    # 质量评估工具 (~80行)
│   └── knowledge_tool.py  # 知识库查询工具 (~80行)
├── prompts/
│   ├── system_prompt.py   # 系统提示词 (~100行)
│   └── tool_schemas.py    # 工具 JSON Schema (~200行)
└── report_generator.py    # 分析报告自动生成 (~200行)

tests/unit/assistant/test_function_calling.py
tests/unit/assistant/test_tools.py
tests/integration/test_copilot_e2e.py
```

#### 2.2.3 核心代码设计

```python
# src/core/assistant/tools/tool_schemas.py
TOOL_DEFINITIONS = [
    {
        "name": "classify_part",
        "description": "对 CAD 图纸进行零件分类，识别零件类型（法兰盘、轴、壳体、支架等8类）",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "已上传文件的 ID"},
                "use_hybrid": {"type": "boolean", "default": True}
            },
            "required": ["file_id"]
        }
    },
    {
        "name": "search_similar",
        "description": "在向量库中搜索与指定图纸相似的零件",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
                "min_similarity": {"type": "number", "default": 0.7},
                "material_filter": {"type": "string", "description": "按材料过滤"}
            },
            "required": ["file_id"]
        }
    },
    {
        "name": "estimate_cost",
        "description": "估算零件的制造成本，包括材料费、加工费、管理费",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "material": {"type": "string", "default": "steel"},
                "batch_size": {"type": "integer", "default": 1},
                "tolerance_grade": {"type": "string", "default": "IT8"}
            },
            "required": ["file_id"]
        }
    },
    {
        "name": "extract_features",
        "description": "提取 CAD 图纸的几何特征向量（95维）",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "version": {"type": "string", "enum": ["v3", "v4"], "default": "v3"}
            },
            "required": ["file_id"]
        }
    },
    {
        "name": "recommend_process",
        "description": "根据零件特征推荐加工工艺路线",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "material": {"type": "string", "default": "steel"},
                "batch_size": {"type": "integer", "default": 1}
            },
            "required": ["file_id"]
        }
    },
    {
        "name": "assess_quality",
        "description": "评估图纸质量（标注完整性、尺寸一致性、图层规范性等）",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"}
            },
            "required": ["file_id"]
        }
    },
    {
        "name": "query_knowledge",
        "description": "查询制造业知识库（材料属性、焊接参数、GD&T规则等）",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "自然语言查询"},
                "category": {
                    "type": "string",
                    "enum": ["materials", "welding", "gdt", "standards", "all"],
                    "default": "all"
                }
            },
            "required": ["query"]
        }
    }
]
```

```python
# src/core/assistant/function_calling.py
from anthropic import Anthropic

class FunctionCallingEngine:
    """LLM Function Calling 引擎，连接自然语言与内部 API"""

    def __init__(self, llm_provider: str = "claude"):
        self.tools = self._register_tools()
        self.client = self._init_client(llm_provider)

    def _register_tools(self) -> dict:
        return {
            "classify_part": ClassifyTool(),
            "search_similar": SimilarityTool(),
            "estimate_cost": CostTool(),
            "extract_features": FeatureTool(),
            "recommend_process": ProcessTool(),
            "assess_quality": QualityTool(),
            "query_knowledge": KnowledgeTool(),
        }

    async def chat(self, user_message: str, conversation_history: list,
                   uploaded_files: list = None) -> AsyncGenerator[str, None]:
        """处理用户消息，支持多轮工具调用"""
        messages = conversation_history + [{"role": "user", "content": user_message}]

        while True:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            # 如果 LLM 要调用工具
            if response.stop_reason == "tool_use":
                tool_calls = [b for b in response.content if b.type == "tool_use"]
                tool_results = []
                for call in tool_calls:
                    result = await self.tools[call.name].execute(call.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            # LLM 生成最终回复
            for block in response.content:
                if hasattr(block, "text"):
                    yield block.text
            break
```

```python
# src/core/assistant/prompts/system_prompt.py
SYSTEM_PROMPT = """你是 CAD ML Platform 的智能分析助手，专门帮助制造业工程师分析 CAD 图纸。

你具备以下能力（通过工具调用实现）:
1. 零件分类 - 识别零件类型（法兰盘、轴、壳体、支架、齿轮、连接件、密封件、其他）
2. 相似度搜索 - 在零件库中查找相似零件，支持按材料/复杂度过滤
3. 成本估算 - 预估材料费、加工费、总成本，支持不同批量/精度
4. 特征提取 - 提取95维深度特征向量，用于几何分析
5. 工艺推荐 - 根据零件特征推荐最佳加工工艺路线
6. 质量评估 - 检查图纸标注完整性、尺寸一致性等
7. 知识查询 - 查询材料属性、焊接参数、GD&T规则等

使用规则:
- 用户上传文件后会获得 file_id，用该 ID 调用工具
- 优先使用工具获取数据，不要猜测或编造技术参数
- 回复使用中文，技术术语保留英文原文
- 成本数据单位为 CNY（人民币）
- 对于不确定的结果，明确告知置信度
- 如果工具返回错误，向用户解释原因并建议替代方案
"""
```

#### 2.2.4 报告自动生成

```python
# src/core/assistant/report_generator.py
class AnalysisReportGenerator:
    """将多个分析结果汇总为结构化报告"""

    async def generate_full_report(self, file_id: str) -> str:
        """生成完整分析报告（分类+特征+工艺+成本+质量）"""
        results = await asyncio.gather(
            self.tools["classify_part"].execute({"file_id": file_id}),
            self.tools["extract_features"].execute({"file_id": file_id}),
            self.tools["recommend_process"].execute({"file_id": file_id}),
            self.tools["estimate_cost"].execute({"file_id": file_id}),
            self.tools["assess_quality"].execute({"file_id": file_id}),
        )
        classification, features, process, cost, quality = results

        # 使用 LLM 将结构化数据转为可读报告
        prompt = f"""根据以下分析数据生成一份简洁的中文分析报告:

零件分类: {json.dumps(classification, ensure_ascii=False)}
几何特征: 共{len(features['vector'])}维特征
推荐工艺: {json.dumps(process, ensure_ascii=False)}
成本估算: {json.dumps(cost, ensure_ascii=False)}
质量评估: {json.dumps(quality, ensure_ascii=False)}

报告格式: Markdown，包含摘要、分类结果、工艺建议、成本明细、质量问题、改进建议。"""

        return await self.llm.generate(prompt)
```

#### 2.2.5 API 端点更新

```python
# src/api/v1/assistant.py 新增端点

@router.post("/api/v1/assistant/chat")
async def chat_with_copilot(
    message: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """Copilot 对话端点，支持文件上传 + Function Calling"""
    engine = get_function_calling_engine()
    history = get_conversation_history(conversation_id)

    file_id = None
    if file:
        file_id = await save_uploaded_file(file)

    async def event_stream():
        async for chunk in engine.chat(message, history, uploaded_files=[file_id]):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/api/v1/assistant/report")
async def generate_analysis_report(file_id: str):
    """一键生成完整分析报告"""
    generator = get_report_generator()
    report = await generator.generate_full_report(file_id)
    return {"report": report, "format": "markdown"}
```

#### 2.2.6 实施计划

| 天 | 任务 | 产出 |
|----|------|------|
| W5 D3 | 工具定义 + Schema | `tools/*.py`, `tool_schemas.py` |
| W5 D4-5 | Function Calling 引擎 | `function_calling.py` |
| W6 D1-2 | 各工具实现（对接已有 API） | 7 个工具全部可调用 |
| W6 D3 | 系统提示词 + 报告生成器 | `system_prompt.py`, `report_generator.py` |
| W6 D4-5 | API 端点 + SSE 流式集成 | `/assistant/chat`, `/assistant/report` |
| W7 D1-2 | 测试 + 多轮对话调优 | 测试文件 + 提示词迭代 |
| W7 D3 | 集成测试 | E2E 验证 |

- **验收**: 用户可以用自然语言查询并获得基于真实数据的回答

---

### Sprint 2.3 — OCR + 文档理解增强（W7-W8: 5/26 ~ 5/30）

#### 2.3.1 增强内容

| 功能 | 当前状态 | 目标 |
|------|---------|------|
| 标题栏 OCR | 基于 DXF 实体提取 | + 图像 OCR（Tesseract/PaddleOCR） |
| 语言支持 | 中文 + 英文 | + 日文 |
| GD&T 符号 | 不支持 | 自动识别几何公差符号 |
| 手写标注 | 不支持 | 基础手写文字识别 |

#### 2.3.2 新增文件

```
src/core/ocr/
├── engine.py             # OCR 引擎统一接口 (~200行)
├── providers/
│   ├── dxf_native.py     # 已有的 DXF 实体提取（迁移）
│   ├── paddle_ocr.py     # PaddleOCR 集成 (~150行)
│   └── tesseract.py      # Tesseract 备选 (~100行)
├── gdt_recognizer.py     # GD&T 符号识别 (~200行)
└── post_processor.py     # OCR 后处理（纠错、结构化）(~150行)

config/ocr_config.yaml    # OCR 配置
```

#### 2.3.3 核心设计

```python
# src/core/ocr/engine.py
class OCREngine:
    def __init__(self, config_path: str = "config/ocr_config.yaml"):
        self.config = load_yaml(config_path)
        self.providers = self._init_providers()

    def _init_providers(self) -> list:
        providers = [DXFNativeProvider()]  # 始终可用
        if self.config.get("paddle_ocr_enabled", False):
            providers.append(PaddleOCRProvider(
                lang=self.config.get("languages", ["ch", "en"])
            ))
        return providers

    async def extract_text(self, file_path: str, file_type: str) -> OCRResult:
        """多策略文字提取：DXF 原生 → 图像 OCR → 合并去重"""
        results = []
        for provider in self.providers:
            if provider.supports(file_type):
                result = await provider.extract(file_path)
                results.append(result)
        return self._merge_results(results)

    async def extract_gdt(self, file_path: str) -> list[GDTAnnotation]:
        """提取 GD&T 几何公差标注"""
        recognizer = GDTRecognizer()
        return await recognizer.recognize(file_path)
```

```yaml
# config/ocr_config.yaml
paddle_ocr_enabled: true
languages: ["ch", "en", "japan"]
gdt_recognition: true
confidence_threshold: 0.6
max_text_regions: 200
```

#### 2.3.4 实施计划

| 天 | 任务 | 产出 |
|----|------|------|
| W7 D4 | OCR 引擎接口 + PaddleOCR 集成 | `engine.py`, `paddle_ocr.py` |
| W7 D5 | GD&T 符号识别 | `gdt_recognizer.py` |
| W8 D1 | 后处理 + 多语言测试 | `post_processor.py` |
| W8 D2 | 与标题栏分类器集成 | 更新 `titleblock_extractor.py` |
| W8 D3 | 测试 + 验收 | 测试文件 |

- **验收**: 中英日三语 OCR 可用，GD&T 基础符号可识别

### 阶段二交付物

| 交付物 | 版本标记 |
|--------|---------|
| 制造成本估算 API | `/api/v1/cost/estimate` |
| 成本模型配置 | `config/cost_model.yaml` |
| LLM Function Calling 引擎 | `src/core/assistant/function_calling.py` |
| 7 个 Copilot 工具 | `src/core/assistant/tools/*.py` |
| 分析报告自动生成 | `/api/v1/assistant/report` |
| OCR 多语言增强 | `src/core/ocr/` |
| GD&T 符号识别 | `src/core/ocr/gdt_recognizer.py` |
| 新增测试 ~20 个文件 | `tests/` |

---

## 五、阶段三：平台化与智能自治（第 9-12 周，6/2 ~ 7/3）

### Sprint 3.1 — React 前端重构（W9-W10: 6/2 ~ 6/13）

#### 3.1.1 技术选型

| 项目 | 选择 | 理由 |
|------|------|------|
| 框架 | React 18 + TypeScript | 生态成熟，组件库丰富 |
| UI 库 | Ant Design 5 | 企业级，中文友好 |
| 状态管理 | Zustand | 轻量，适合中小项目 |
| 图表 | ECharts | 制造业仪表盘常用 |
| CAD 预览 | three.js + STEP 解析 | 3D 模型在线预览 |
| 构建 | Vite | 快速开发体验 |

#### 3.1.2 页面设计

```
web-app/
├── src/
│   ├── pages/
│   │   ├── Dashboard/          # 首页仪表盘
│   │   │   ├── index.tsx       # 概览统计（今日分析量、准确率、成本节省）
│   │   │   ├── RecentAnalysis.tsx
│   │   │   └── DriftStatus.tsx
│   │   ├── Analysis/           # 图纸分析
│   │   │   ├── Upload.tsx      # 文件上传（拖拽 + 点击）
│   │   │   ├── Result.tsx      # 分析结果展示
│   │   │   ├── FeatureView.tsx # 95维特征可视化（雷达图）
│   │   │   ├── CostView.tsx    # 成本明细（饼图 + 表格）
│   │   │   └── ProcessView.tsx # 工艺路线（流程图）
│   │   ├── Similarity/         # 相似度搜索
│   │   │   ├── Search.tsx      # 搜索界面
│   │   │   └── Compare.tsx     # 零件对比（并排展示）
│   │   ├── Copilot/            # AI 对话
│   │   │   ├── Chat.tsx        # 对话界面（SSE 流式）
│   │   │   └── Report.tsx      # 报告查看/导出
│   │   ├── Models/             # 模型管理
│   │   │   ├── List.tsx        # 模型列表（版本、状态、准确率）
│   │   │   ├── ABTest.tsx      # A/B 测试结果
│   │   │   └── Drift.tsx       # 漂移监控
│   │   └── Settings/           # 系统设置
│   │       ├── Tenant.tsx      # 租户管理
│   │       ├── Users.tsx       # 用户管理 (RBAC)
│   │       └── Config.tsx      # 系统配置
│   ├── components/
│   │   ├── CADViewer.tsx       # 3D CAD 预览组件
│   │   ├── FileUploader.tsx    # 通用上传组件
│   │   ├── ConfidenceBadge.tsx # 置信度标签
│   │   └── CostChart.tsx       # 成本图表组件
│   ├── hooks/
│   │   ├── useSSE.ts           # SSE 流式数据 hook
│   │   └── useAPI.ts           # API 调用 hook
│   └── api/
│       └── client.ts           # API 客户端（自动生成自 OpenAPI）
```

#### 3.1.3 关键组件设计

```typescript
// src/hooks/useSSE.ts — 复用已有 SSE streaming 后端
export function useSSE(url: string) {
  const [messages, setMessages] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const start = useCallback((body: any) => {
    setIsStreaming(true);
    const eventSource = new EventSource(url);
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data === "[DONE]") {
        eventSource.close();
        setIsStreaming(false);
      } else {
        setMessages(prev => [...prev, data.text]);
      }
    };
  }, [url]);

  return { messages, isStreaming, start };
}
```

#### 3.1.4 实施计划

| 天 | 任务 | 产出 |
|----|------|------|
| W9 D1 | 项目初始化 + 路由 + 布局 | `web-app/` 脚手架 |
| W9 D2-3 | Dashboard + Analysis（上传+结果） | 核心两页 |
| W9 D4-5 | Copilot 对话 + SSE 集成 | AI 对话页 |
| W10 D1-2 | Similarity 搜索 + 对比 | 相似度页 |
| W10 D3 | Cost 可视化 + Process 流程图 | 成本/工艺页 |
| W10 D4 | Models 管理 + Drift 监控 | 管理页面 |
| W10 D5 | Settings + 权限集成 | 设置页 + RBAC |

- **验收**: 所有页面可用，对接真实 API，流式响应正常

---

### Sprint 3.2 — 智能异常检测与自动修复（W11: 6/16 ~ 6/20）

#### 3.2.1 新增文件

```
src/ml/monitoring/
├── anomaly_detector.py    # ML 异常检测 (~300行)
├── alert_correlator.py    # 告警关联引擎 (~200行)
└── auto_remediation.py    # 自动修复执行器 (~250行)

config/anomaly_detection.yaml
```

#### 3.2.2 核心设计

```python
# src/ml/monitoring/anomaly_detector.py
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class MetricsAnomalyDetector:
    """基于 Isolation Forest 的时序指标异常检测"""

    def __init__(self, config_path: str = "config/anomaly_detection.yaml"):
        self.config = load_yaml(config_path)
        self.models: dict[str, IsolationForest] = {}
        self.scalers: dict[str, StandardScaler] = {}

    def fit(self, metric_name: str, historical_data: np.ndarray):
        """用历史数据训练异常检测模型"""
        scaler = StandardScaler()
        scaled = scaler.fit_transform(historical_data.reshape(-1, 1))
        model = IsolationForest(
            contamination=self.config.get("contamination", 0.05),
            n_estimators=self.config.get("n_estimators", 100),
            random_state=42
        )
        model.fit(scaled)
        self.models[metric_name] = model
        self.scalers[metric_name] = scaler

    def detect(self, metric_name: str, current_value: float) -> AnomalyResult:
        """检测当前值是否异常"""
        model = self.models.get(metric_name)
        if not model:
            return AnomalyResult(is_anomaly=False, reason="no model trained")
        scaler = self.scalers[metric_name]
        scaled = scaler.transform([[current_value]])
        score = model.decision_function(scaled)[0]
        prediction = model.predict(scaled)[0]
        return AnomalyResult(
            is_anomaly=(prediction == -1),
            anomaly_score=float(-score),
            severity=self._score_to_severity(score),
            metric_name=metric_name,
            current_value=current_value,
        )
```

```python
# src/ml/monitoring/auto_remediation.py
class AutoRemediation:
    """自动修复执行器"""

    REMEDIATION_RULES = {
        "model_accuracy_drop": {
            "action": "rollback_model",
            "condition": "accuracy < 0.95 for 30min",
            "max_auto_actions": 3,
        },
        "drift_detected_high": {
            "action": "refresh_baseline",
            "condition": "drift_severity == HIGH",
            "max_auto_actions": 1,
        },
        "cache_hit_rate_low": {
            "action": "expand_cache",
            "condition": "hit_rate < 0.3 for 1h",
            "max_auto_actions": 2,
        },
        "latency_spike": {
            "action": "scale_workers",
            "condition": "p95_latency > 5s for 10min",
            "max_auto_actions": 2,
        },
    }

    async def evaluate_and_act(self, anomaly: AnomalyResult) -> RemediationResult:
        rule = self._match_rule(anomaly)
        if not rule:
            return RemediationResult(action="none")
        if self._exceeds_limit(rule):
            return RemediationResult(action="escalate_to_human")
        return await self._execute(rule["action"], anomaly)
```

#### 3.2.3 实施计划

| 天 | 任务 | 产出 |
|----|------|------|
| W11 D1-2 | 异常检测器（Isolation Forest） | `anomaly_detector.py` |
| W11 D3 | 告警关联引擎 | `alert_correlator.py` |
| W11 D4 | 自动修复执行器 | `auto_remediation.py` |
| W11 D5 | 集成到现有 drift 监控 | 更新 `drift.py` + 测试 |

- **验收**: 异常检测准确率 > 90%，自动修复可触发（带人工确认）

---

### Sprint 3.3 — 图纸版本 Diff（W12: 6/23 ~ 6/27）

#### 3.3.1 新增文件

```
src/core/diff/
├── __init__.py
├── geometry_diff.py       # 几何差异检测 (~300行)
├── annotation_diff.py     # 标注差异检测 (~200行)
├── report.py              # 差异报告生成 (~150行)
└── visualizer.py          # 差异可视化数据 (~200行)

src/api/v1/diff.py         # Diff API 端点

tests/unit/test_geometry_diff.py
tests/unit/test_annotation_diff.py
```

#### 3.3.2 核心设计

```python
# src/core/diff/geometry_diff.py
class GeometryDiff:
    """两版图纸的几何差异检测"""

    def compare(self, file_a: str, file_b: str) -> DiffResult:
        """对比两个 CAD 文件的几何差异"""
        # 1. 提取两个文件的实体
        entities_a = self._extract_entities(file_a)
        entities_b = self._extract_entities(file_b)

        # 2. 实体匹配（基于位置+类型的最近邻匹配）
        matched, added, removed = self._match_entities(entities_a, entities_b)

        # 3. 检测修改（匹配实体的属性差异）
        modified = []
        for ea, eb in matched:
            changes = self._compare_entity_attrs(ea, eb)
            if changes:
                modified.append(EntityChange(
                    entity_id=ea.id,
                    entity_type=ea.dxftype,
                    changes=changes,
                    location=ea.centroid,
                ))

        return DiffResult(
            added=added,
            removed=removed,
            modified=modified,
            summary=self._generate_summary(added, removed, modified),
            change_regions=self._compute_change_regions(added + removed + modified),
        )

    def _match_entities(self, entities_a, entities_b):
        """基于空间距离 + 类型相似度的实体匹配"""
        from scipy.spatial import KDTree
        # 用 KDTree 做最近邻匹配
        points_a = np.array([e.centroid for e in entities_a])
        points_b = np.array([e.centroid for e in entities_b])
        tree_b = KDTree(points_b)
        distances, indices = tree_b.query(points_a)
        # ... 匹配逻辑
```

```python
# src/api/v1/diff.py
@router.post("/api/v1/diff/compare")
async def compare_drawings(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    """对比两张图纸的差异"""
    differ = GeometryDiff()
    result = differ.compare(
        await save_temp(file_a),
        await save_temp(file_b),
    )
    return result

@router.post("/api/v1/diff/ecn")
async def generate_ecn(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    part_number: str = Form(...),
):
    """基于图纸差异自动生成工程变更通知 (ECN)"""
    diff = GeometryDiff().compare(...)
    ecn = ECNGenerator().generate(diff, part_number)
    return ecn
```

#### 3.3.3 实施计划

| 天 | 任务 | 产出 |
|----|------|------|
| W12 D1-2 | 几何差异检测 | `geometry_diff.py` |
| W12 D3 | 标注差异检测 | `annotation_diff.py` |
| W12 D4 | API + ECN 生成 | `diff.py`, `report.py` |
| W12 D5 | 前端差异可视化 | React 组件 |

---

### Sprint 3.4 — 最终验收（W12 D5 + 缓冲）

```bash
# 1. 全量回归测试
make test-all-local

# 2. 性能基线对比
python scripts/performance_baseline.py --compare-with=v2.0.1

# 3. Docker 全栈验证
docker-compose -f docker-compose.yml -f docker-compose.observability.yml up -d
# 验证: API / 前端 / Prometheus / Grafana / Redis 全部健康

# 4. 安全审计
make security-check

# 5. 版本发布
git tag v3.0.0
```

### 阶段三交付物

| 交付物 | 说明 |
|--------|------|
| React 前端 | `web-app/`，完整 8 个页面 |
| ML 异常检测 | `src/ml/monitoring/anomaly_detector.py` |
| 自动修复引擎 | `src/ml/monitoring/auto_remediation.py` |
| 图纸 Diff | `src/core/diff/` |
| ECN 自动生成 | `/api/v1/diff/ecn` |
| v3.0.0 发布 | 全栈验证通过 |

---

## 六、资源需求与风险

### 人力预估

| 角色 | 人数 | 阶段一 | 阶段二 | 阶段三 |
|------|------|--------|--------|--------|
| 后端工程师 | 2 | 安全+启用+V4 | 成本+Copilot+OCR | 异常检测+Diff |
| 前端工程师 | 1 | — | — | React 前端 |
| ML 工程师 | 1 | 蒸馏+盲评 | 成本模型调参 | 异常检测模型 |
| 测试 | 1 | 全程参与 | 全程参与 | 全程参与 |
| **合计** | **3-5 人** | | | |

### 风险与缓解

| 风险 | 影响 | 概率 | 缓解 |
|------|------|------|------|
| Graph2D 启用后准确率下降 | 高 | 中 | 先低权重(0.1)接入，渐进调高 |
| 蒸馏模型精度损失过大 | 中 | 低 | 保留 teacher 模型，随时回切 |
| LLM API 成本过高 | 中 | 中 | 默认用 Qwen（国内便宜），Claude 仅高级租户 |
| PaddleOCR 日文效果差 | 低 | 中 | 日文走 Tesseract 备选通道 |
| React 重构工期超 | 中 | 高 | 砍 Settings 页，用已有 Web UI 过渡 |
| 成本估算不准确 | 高 | 中 | 明确标注"估算"，给出区间而非单值 |

### 依赖项

| 依赖 | 获取方式 | 阶段 |
|------|---------|------|
| 历史加工订单数据（成本校准） | 客户提供 | 阶段二 |
| LLM API Key（Claude/GPT） | 内部申请 | 阶段二 |
| PaddleOCR 模型包 | `pip install paddleocr` | 阶段二 |
| GD&T 标注样本数据集 | 内部标注 | 阶段二 |
| 图纸版本对样本（同零件两版） | 客户提供 | 阶段三 |

---

## 七、里程碑检查点

| 日期 | 里程碑 | 验收标准 |
|------|--------|---------|
| **5/2** (W3末) | 阶段一完成 | 所有关闭功能启用，V4 可用，监控完整，准确率 ≥ 99.5% |
| **5/16** (W5末) | 成本估算上线 | `/api/v1/cost/estimate` 可用，估算合理 |
| **5/30** (W8末) | 阶段二完成 | Copilot 可对话，OCR 三语可用，报告可生成 |
| **6/13** (W10末) | 前端上线 | React 前端 8 页面可用 |
| **6/27** (W12末) | v3.0.0 发布 | 全栈验证通过，异常检测 + 图纸 Diff 可用 |
| **7/3** | 缓冲完成 | 文档完善，交付物清单确认 |

---

## 八、版本号规划

```
v2.0.1 (当前)
  ↓
v2.1.0 — 阶段一完成（启用功能 + 安全加固 + V4）
  ↓
v2.2.0 — 制造成本估算
  ↓
v2.3.0 — LLM Copilot + Function Calling
  ↓
v2.4.0 — OCR 增强
  ↓
v3.0.0 — React 前端 + 智能自治 + 图纸 Diff（重大版本）
```

---

**文档维护**: 每个 Sprint 结束时更新本文档的完成状态
**状态跟踪**: 使用 GitHub Projects 或 Linear 看板管理任务
