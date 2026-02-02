# CAD-ML Platform - ML核心功能增强开发计划

> **版本**: 1.0.0
> **创建日期**: 2026-02-01
> **状态**: 实施中

---

## 📋 目录

1. [项目背景](#1-项目背景)
2. [当前状态分析](#2-当前状态分析)
3. [开发目标](#3-开发目标)
4. [模块详细设计](#4-模块详细设计)
5. [实施计划](#5-实施计划)
6. [验收标准](#6-验收标准)
7. [风险评估](#7-风险评估)

---

## 1. 项目背景

### 1.1 项目概述

CAD-ML Platform 是一个面向机械CAD文件的智能分析平台，提供：
- CAD文件分类 (HybridClassifier)
- 相似图纸去重 (Dedup2D Pipeline)
- 几何特征提取与向量化
- 图神经网络模型训练

### 1.2 当前挑战

| 挑战 | 描述 | 影响 |
|------|------|------|
| 实验管理分散 | 训练脚本独立运行，无统一跟踪 | 难以复现和比较实验 |
| 评估标准不一 | 各模型评估方式不同 | 无法客观对比模型性能 |
| 元数据提取有限 | 标题栏信息利用不足 | 分类准确率受限 |
| 部署效率低 | 缺乏统一推理服务框架 | 生产环境响应慢 |

---

## 2. 当前状态分析

### 2.1 已有ML能力矩阵

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAD-ML Platform 能力图                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ 数据层      │    │ 模型层      │    │ 服务层      │          │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤          │
│  │ ✅ DXF解析  │───▶│ ✅ Graph2D  │───▶│ ⚠️ 批推理   │          │
│  │ ✅ 几何提取 │    │ ✅ UVNet    │    │ ⚠️ 模型服务 │          │
│  │ ✅ 向量存储 │    │ ✅ Hybrid   │    │ ✅ API端点  │          │
│  │ ⚠️ 标题栏  │    │ ⚠️ 评估    │    │ ⚠️ 缓存    │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                   │
│  ✅ 完整实现  ⚠️ 需要增强  ❌ 缺失                              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 代码结构现状

```
src/ml/
├── classifier.py           # 基础分类器
├── hybrid_classifier.py    # ✅ 混合分类器 (4源融合)
├── filename_classifier.py  # ✅ 文件名分类
├── process_classifier.py   # ✅ 工艺特征分类
├── titleblock_extractor.py # ⚠️ 标题栏提取 (基础)
├── vision_2d.py            # ✅ 2D视觉模型
├── vision_3d.py            # ✅ 3D视觉模型
├── knowledge_distillation.py # ✅ 知识蒸馏
├── multimodal_fusion.py    # ✅ 多模态融合
├── train/
│   ├── model.py            # ✅ UVNet模型
│   ├── model_2d.py         # ✅ 2D图模型
│   ├── dataset.py          # ✅ 3D数据集
│   ├── dataset_2d.py       # ✅ 2D数据集
│   └── trainer.py          # ✅ 训练器
└── [需要新增的模块]
    ├── experiment_tracker.py    # M1
    ├── evaluation/              # M3
    ├── serving/                 # I1
    └── active_learning/         # F1
```

---

## 3. 开发目标

### 3.1 总体目标

建立完整的ML实验-训练-评估-部署闭环，提升模型开发效率和生产质量。

### 3.2 具体指标

| 指标 | 当前值 | 目标值 | 提升 |
|------|--------|--------|------|
| 实验可复现性 | 30% | 95% | +65% |
| 模型评估覆盖率 | 40% | 90% | +50% |
| 推理延迟 P99 | 500ms | 100ms | 5x |
| 模型迭代周期 | 2周 | 3天 | 5x |

---

## 4. 模块详细设计

### 4.1 M1 - 实验跟踪系统

#### 4.1.1 设计目标

提供统一的实验管理基础设施，支持：
- 实验参数记录
- 指标追踪与可视化
- 模型版本管理
- 实验对比分析

#### 4.1.2 架构设计

```
┌──────────────────────────────────────────────────────────────┐
│                    Experiment Tracker                         │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │ Experiment  │   │   Run       │   │  Artifact   │         │
│  │ Manager     │──▶│  Tracker    │──▶│  Store      │         │
│  └─────────────┘   └─────────────┘   └─────────────┘         │
│        │                 │                  │                  │
│        ▼                 ▼                  ▼                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │ Config      │   │  Metrics    │   │   Model     │         │
│  │ Registry    │   │  Logger     │   │  Registry   │         │
│  └─────────────┘   └─────────────┘   └─────────────┘         │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

#### 4.1.3 核心接口

```python
# src/ml/experiment_tracker.py

class ExperimentTracker:
    """统一实验跟踪器"""

    def start_run(self, experiment_name: str, config: Dict) -> str:
        """开始新的实验运行"""

    def log_params(self, params: Dict[str, Any]) -> None:
        """记录超参数"""

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """记录指标"""

    def log_artifact(self, local_path: str, artifact_type: str) -> None:
        """保存模型或其他文件"""

    def end_run(self, status: str = "completed") -> None:
        """结束运行"""

    def compare_runs(self, run_ids: List[str]) -> ComparisonReport:
        """对比多次运行"""


class ModelRegistry:
    """模型版本注册中心"""

    def register_model(self, name: str, version: str,
                       model_path: str, metrics: Dict) -> str:
        """注册新模型版本"""

    def get_model(self, name: str, version: str = "latest") -> ModelInfo:
        """获取模型信息"""

    def promote_model(self, name: str, version: str, stage: str) -> None:
        """提升模型阶段 (staging -> production)"""
```

#### 4.1.4 文件结构

```
src/ml/experiment/
├── __init__.py
├── tracker.py           # ExperimentTracker 实现
├── run.py              # Run 管理
├── metrics.py          # 指标日志
├── artifacts.py        # 产物存储
├── registry.py         # 模型注册
├── backends/
│   ├── __init__.py
│   ├── local.py        # 本地文件后端
│   ├── mlflow.py       # MLflow集成 (可选)
│   └── wandb.py        # W&B集成 (可选)
└── utils/
    ├── __init__.py
    ├── config.py       # 配置序列化
    └── comparison.py   # 实验对比
```

---

### 4.2 M3 - 模型评估框架

#### 4.2.1 设计目标

提供标准化的模型评估流程：
- 多维度指标计算
- 混淆矩阵分析
- 错误案例分析
- 自动化报告生成

#### 4.2.2 架构设计

```
┌──────────────────────────────────────────────────────────────┐
│                   Evaluation Framework                        │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │ Evaluator   │──▶│  Metrics    │──▶│  Reporter   │         │
│  │ Engine      │   │  Calculator │   │  Generator  │         │
│  └─────────────┘   └─────────────┘   └─────────────┘         │
│        │                                     │                 │
│        ▼                                     ▼                 │
│  ┌─────────────┐                      ┌─────────────┐         │
│  │ Confusion   │                      │   HTML/MD   │         │
│  │ Analyzer    │                      │   Reports   │         │
│  └─────────────┘                      └─────────────┘         │
│        │                                                       │
│        ▼                                                       │
│  ┌─────────────┐                                              │
│  │ Error Case  │                                              │
│  │ Collector   │                                              │
│  └─────────────┘                                              │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

#### 4.2.3 核心接口

```python
# src/ml/evaluation/evaluator.py

class ModelEvaluator:
    """模型评估器"""

    def evaluate(self, model: Any, dataset: Dataset,
                 config: EvalConfig) -> EvaluationResult:
        """执行完整评估"""

    def calculate_metrics(self, y_true: List, y_pred: List,
                          labels: List[str]) -> MetricsReport:
        """计算评估指标"""

    def analyze_confusion(self, y_true: List, y_pred: List,
                          labels: List[str]) -> ConfusionAnalysis:
        """分析混淆矩阵"""

    def collect_errors(self, predictions: List[Prediction],
                       top_k: int = 100) -> ErrorCaseCollection:
        """收集错误案例"""


@dataclass
class EvaluationResult:
    """评估结果"""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class_metrics: Dict[str, ClassMetrics]
    confusion_matrix: np.ndarray
    error_cases: List[ErrorCase]
    evaluation_time: float

    def to_report(self, format: str = "markdown") -> str:
        """生成报告"""


@dataclass
class ClassMetrics:
    """类别级指标"""
    precision: float
    recall: float
    f1: float
    support: int
    false_positives: int
    false_negatives: int
```

#### 4.2.4 文件结构

```
src/ml/evaluation/
├── __init__.py
├── evaluator.py        # 主评估器
├── metrics.py          # 指标计算
├── confusion.py        # 混淆矩阵分析
├── error_analysis.py   # 错误案例分析
├── reporter.py         # 报告生成
├── visualizer.py       # 可视化
└── templates/
    ├── report.md.j2    # Markdown模板
    └── report.html.j2  # HTML模板
```

---

### 4.3 C3 - 标题栏智能解析

#### 4.3.1 设计目标

从CAD图纸标题栏中结构化提取元数据：
- 自动定位标题栏区域
- 识别标准字段
- 支持多种模板格式
- OCR与规则混合解析

#### 4.3.2 架构设计

```
┌──────────────────────────────────────────────────────────────┐
│                  Titleblock Parser                            │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │ Region      │──▶│  Template   │──▶│   Field     │         │
│  │ Detector    │   │  Matcher    │   │  Extractor  │         │
│  └─────────────┘   └─────────────┘   └─────────────┘         │
│        │                 │                  │                  │
│        ▼                 ▼                  ▼                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │ Geometry    │   │  Template   │   │  OCR        │         │
│  │ Analysis    │   │  Library    │   │  Engine     │         │
│  └─────────────┘   └─────────────┘   └─────────────┘         │
│                                                                │
│  输出: TitleblockMetadata                                      │
│  ├── part_number: str                                         │
│  ├── drawing_title: str                                       │
│  ├── material: str                                            │
│  ├── author: str                                              │
│  ├── date: str                                                │
│  ├── revision: str                                            │
│  └── custom_fields: Dict[str, str]                            │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

#### 4.3.3 核心接口

```python
# src/core/cad/titleblock_parser.py

class TitleblockParser:
    """标题栏解析器"""

    def parse(self, dxf_path: Path,
              template: Optional[str] = None) -> TitleblockMetadata:
        """解析标题栏"""

    def detect_region(self, entities: List[DXFEntity]) -> BoundingBox:
        """检测标题栏区域"""

    def match_template(self, region: BoundingBox,
                       entities: List[DXFEntity]) -> TemplateMatch:
        """匹配模板"""

    def extract_fields(self, region: BoundingBox,
                       template: Template) -> Dict[str, str]:
        """提取字段值"""


@dataclass
class TitleblockMetadata:
    """标题栏元数据"""
    part_number: Optional[str] = None
    drawing_title: Optional[str] = None
    material: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    revision: Optional[str] = None
    scale: Optional[str] = None
    sheet: Optional[str] = None
    custom_fields: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    template_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""


class TemplateLibrary:
    """模板库"""

    def add_template(self, name: str, template: TitleblockTemplate) -> None:
        """添加模板"""

    def match(self, region_features: RegionFeatures) -> List[TemplateMatch]:
        """匹配模板"""

    def load_from_directory(self, path: Path) -> int:
        """从目录加载模板"""
```

#### 4.3.4 文件结构

```
src/core/cad/
├── __init__.py
├── titleblock_parser.py    # 主解析器
├── region_detector.py      # 区域检测
├── template_matcher.py     # 模板匹配
├── field_extractor.py      # 字段提取
├── templates/
│   ├── __init__.py
│   ├── library.py          # 模板库
│   ├── iso_standard.py     # ISO标准模板
│   ├── gb_standard.py      # 国标模板
│   └── custom.py           # 自定义模板
└── ocr_integration.py      # OCR集成
```

---

### 4.4 I1 - 模型服务化框架

#### 4.4.1 设计目标

提供高效可靠的模型推理服务：
- 多模型并行服务
- 动态模型加载
- 请求负载均衡
- 资源隔离与监控

#### 4.4.2 架构设计

```
┌──────────────────────────────────────────────────────────────┐
│                    Model Serving Framework                    │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────────────────────────────┐          │
│  │                   API Gateway                    │          │
│  │  (Request Routing / Rate Limiting / Auth)       │          │
│  └─────────────────────────────────────────────────┘          │
│                          │                                     │
│                          ▼                                     │
│  ┌─────────────────────────────────────────────────┐          │
│  │                 Load Balancer                    │          │
│  │  (Round Robin / Least Conn / Weighted)          │          │
│  └─────────────────────────────────────────────────┘          │
│           │              │              │                      │
│           ▼              ▼              ▼                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐             │
│  │  Model      │ │  Model      │ │  Model      │             │
│  │  Worker 1   │ │  Worker 2   │ │  Worker N   │             │
│  │  (Graph2D)  │ │  (Hybrid)   │ │  (UVNet)    │             │
│  └─────────────┘ └─────────────┘ └─────────────┘             │
│           │              │              │                      │
│           ▼              ▼              ▼                      │
│  ┌─────────────────────────────────────────────────┐          │
│  │              Model Registry Cache                │          │
│  │    (Hot Models / Warm Models / Cold Storage)    │          │
│  └─────────────────────────────────────────────────┘          │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

#### 4.4.3 核心接口

```python
# src/ml/serving/server.py

class ModelServer:
    """模型服务器"""

    def load_model(self, model_name: str, version: str = "latest") -> str:
        """加载模型到内存"""

    def unload_model(self, model_name: str) -> bool:
        """卸载模型"""

    def predict(self, model_name: str, inputs: List[Any],
                batch_size: int = 32) -> List[Prediction]:
        """执行预测"""

    def get_model_info(self, model_name: str) -> ModelInfo:
        """获取模型信息"""

    def list_models(self) -> List[ModelInfo]:
        """列出所有已加载模型"""


class ModelWorker:
    """模型工作进程"""

    def __init__(self, model_path: str, device: str = "cpu"):
        """初始化工作进程"""

    def predict_batch(self, inputs: List[Any]) -> List[Prediction]:
        """批量预测"""

    def warmup(self, sample_input: Any) -> float:
        """预热模型"""

    def get_stats(self) -> WorkerStats:
        """获取统计信息"""


class InferenceRequest:
    """推理请求"""
    request_id: str
    model_name: str
    inputs: List[Any]
    priority: int = 0
    timeout: float = 30.0
    created_at: datetime


class InferenceResponse:
    """推理响应"""
    request_id: str
    predictions: List[Prediction]
    latency_ms: float
    model_version: str
```

#### 4.4.4 文件结构

```
src/ml/serving/
├── __init__.py
├── server.py           # 主服务器
├── worker.py           # 工作进程
├── router.py           # 请求路由
├── balancer.py         # 负载均衡
├── cache.py            # 模型缓存
├── batch.py            # 动态批处理
├── health.py           # 健康检查
└── metrics.py          # 性能指标
```

---

## 5. 实施计划

### 5.1 总体时间线

```
┌────────────────────────────────────────────────────────────────┐
│                      实施时间线                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Week 1-2: M1 实验跟踪 + M3 评估框架                            │
│  ════════════════════════════════════                           │
│  Day 1-3:  M1 核心实现 (tracker, run, metrics)                  │
│  Day 4-5:  M1 模型注册 + 后端集成                               │
│  Day 6-8:  M3 评估器 + 指标计算                                 │
│  Day 9-10: M3 报告生成 + 可视化                                 │
│                                                                  │
│  Week 3-4: C3 标题栏解析 + I1 模型服务化                        │
│  ════════════════════════════════════════                       │
│  Day 11-13: C3 区域检测 + 模板库                                │
│  Day 14-15: C3 字段提取 + OCR集成                               │
│  Day 16-18: I1 服务器 + 工作进程                                │
│  Day 19-20: I1 负载均衡 + 批处理                                │
│                                                                  │
│  Week 5: 集成测试 + 文档                                        │
│  ══════════════════════════                                     │
│  Day 21-23: 集成测试                                            │
│  Day 24-25: 文档完善 + 最终验证                                 │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 详细任务分解

#### Week 1-2: M1 + M3

| 任务ID | 任务名称 | 依赖 | 预估工时 |
|--------|----------|------|---------|
| M1-1 | ExperimentTracker 基础类 | - | 4h |
| M1-2 | Run 管理实现 | M1-1 | 3h |
| M1-3 | Metrics 日志记录 | M1-2 | 3h |
| M1-4 | Artifact 存储 | M1-2 | 3h |
| M1-5 | ModelRegistry 实现 | M1-4 | 4h |
| M1-6 | 本地文件后端 | M1-5 | 3h |
| M1-7 | 实验对比功能 | M1-6 | 3h |
| M3-1 | ModelEvaluator 基础类 | - | 4h |
| M3-2 | 指标计算模块 | M3-1 | 4h |
| M3-3 | 混淆矩阵分析 | M3-2 | 3h |
| M3-4 | 错误案例收集 | M3-2 | 3h |
| M3-5 | 报告生成器 | M3-3, M3-4 | 4h |
| M3-6 | 可视化模块 | M3-5 | 3h |

#### Week 3-4: C3 + I1

| 任务ID | 任务名称 | 依赖 | 预估工时 |
|--------|----------|------|---------|
| C3-1 | 区域检测器 | - | 4h |
| C3-2 | 模板库基础 | C3-1 | 4h |
| C3-3 | 国标/ISO模板 | C3-2 | 3h |
| C3-4 | 字段提取器 | C3-2 | 4h |
| C3-5 | OCR集成 | C3-4 | 3h |
| C3-6 | 解析器整合 | C3-5 | 3h |
| I1-1 | ModelServer 基础 | - | 4h |
| I1-2 | ModelWorker 实现 | I1-1 | 4h |
| I1-3 | 请求路由 | I1-2 | 3h |
| I1-4 | 负载均衡 | I1-3 | 3h |
| I1-5 | 动态批处理 | I1-4 | 4h |
| I1-6 | 健康检查 | I1-5 | 2h |

---

## 6. 验收标准

### 6.1 功能验收

#### M1 实验跟踪
- [ ] 可创建/结束实验运行
- [ ] 可记录超参数和指标
- [ ] 可保存和加载模型产物
- [ ] 可对比多次运行结果
- [ ] 支持模型版本注册

#### M3 模型评估
- [ ] 支持准确率/精确率/召回率/F1计算
- [ ] 生成混淆矩阵分析
- [ ] 收集Top-K错误案例
- [ ] 生成Markdown/HTML报告
- [ ] 支持类别级指标分析

#### C3 标题栏解析
- [ ] 自动检测标题栏区域
- [ ] 支持至少3种标准模板
- [ ] 提取核心字段 (零件号/标题/材料等)
- [ ] 解析置信度 >= 80%
- [ ] 支持自定义模板扩展

#### I1 模型服务化
- [ ] 支持多模型并行加载
- [ ] 支持动态加载/卸载
- [ ] 推理延迟 P99 < 200ms
- [ ] 支持批量推理
- [ ] 健康检查端点可用

### 6.2 质量验收

| 指标 | 标准 |
|------|------|
| 单元测试覆盖率 | >= 80% |
| 类型注解覆盖率 | >= 90% |
| 文档完整性 | 100% 公共接口 |
| 代码风格 | 通过 ruff 检查 |

---

## 7. 风险评估

### 7.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| OCR准确率不足 | 中 | 中 | 多引擎备选，规则后处理 |
| 模型加载内存溢出 | 低 | 高 | 模型分片，LRU缓存 |
| 标题栏模板多样性 | 高 | 中 | 可扩展模板库，机器学习辅助 |

### 7.2 进度风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 依赖库兼容问题 | 中 | 低 | 提前锁定版本 |
| 需求变更 | 中 | 中 | 迭代开发，及时沟通 |

---

## 附录

### A. 相关文档

- [HybridClassifier 设计文档](./hybrid_classifier_design.md)
- [训练脚本使用指南](./training_guide.md)
- [API 接口文档](./api_reference.md)

### B. 参考资料

- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- PyTorch Model Serving: https://pytorch.org/serve/
- DXF File Format: https://images.autodesk.com/adsk/files/autocad_2012_pdf_dxf-reference_enu.pdf

---

*文档维护者: CAD-ML Platform Team*
*最后更新: 2026-02-01*
