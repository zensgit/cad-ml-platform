# 指标基数预算系统使用指南

**版本**: v1.0.0
**最后更新**: 2025-11-22
**工具集**: 基数追踪器、预算控制器、分析报告器、自动优化器

---

## 📚 目录

1. [快速开始](#快速开始)
2. [核心概念](#核心概念)
3. [工具详解](#工具详解)
4. [CI/CD集成](#cicd集成)
5. [配置与定制](#配置与定制)
6. [最佳实践](#最佳实践)
7. [故障排除](#故障排除)
8. [API参考](#api参考)

---

## 🚀 快速开始

### 安装依赖

```bash
# Python 3.8+ required
pip install -r requirements.txt

# 额外依赖
pip install pyyaml requests
```

### 快速体验

```bash
# 1. 分析当前指标基数
python scripts/metrics_cardinality_tracker.py \
  --prometheus-url http://localhost:9090 \
  --top 10

# 2. 检查预算状态
python scripts/metrics_budget_controller.py \
  --status

# 3. 生成分析报告
python scripts/cardinality_analysis_report.py \
  --format markdown \
  --output report.md

# 4. 自动优化建议
python scripts/metrics_auto_optimizer.py \
  --prometheus-url http://localhost:9090
```

---

## 📊 核心概念

### 什么是指标基数（Cardinality）？

指标基数是指一个指标的所有唯一时间序列组合数量。例如：
- 指标：`http_requests_total`
- 标签：`{method, status, path}`
- 如果有 5个方法 × 10个状态码 × 100个路径 = 5000个时间序列

### 为什么要管理基数？

1. **成本控制**: 每个时间序列都占用存储空间
2. **性能优化**: 高基数影响查询速度
3. **稳定性保障**: 防止基数爆炸导致OOM

### 基数预算系统架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Prometheus    │────▶│ Cardinality      │────▶│ Budget          │
│                 │     │ Tracker          │     │ Controller      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                          │
                               ▼                          ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ Analysis         │     │ Auto            │
                        │ Reporter         │     │ Optimizer       │
                        └──────────────────┘     └─────────────────┘
```

---

## 🔧 工具详解

### 1. 基数追踪器 (metrics_cardinality_tracker.py)

**功能**: 实时追踪每个指标的基数和成本

#### 基本使用

```bash
# 分析特定指标
python scripts/metrics_cardinality_tracker.py \
  --prometheus-url http://localhost:9090 \
  --metric http_requests_total \
  --analyze-labels

# 输出示例：
# 📊 Metric: http_requests_total
#    Cardinality: 5,234
#    Storage: 125.45 MB
#    Monthly Cost: $0.1254
#    Labels: {
#      "method": 5,
#      "status": 12,
#      "path": 87
#    }
```

#### Python API使用

```python
from metrics_cardinality_tracker import MetricsCardinalityTracker

tracker = MetricsCardinalityTracker("http://localhost:9090")

# 获取指标基数
info = tracker.get_metric_cardinality("http_requests_total")
print(f"Cardinality: {info.cardinality}")
print(f"Monthly Cost: ${info.monthly_cost:.2f}")

# 识别高基数标签
labels = tracker.identify_high_cardinality_labels("http_requests_total")
for label in labels:
    if label.unique_values > 100:
        print(f"Warning: {label.label_name} has {label.unique_values} unique values")
```

### 2. 预算控制器 (metrics_budget_controller.py)

**功能**: 管理团队/服务的基数预算

#### 配置预算

```python
# budget_config.json
{
  "global_max_series": 1000000,
  "team_budgets": {
    "platform": 300000,
    "api": 200000,
    "frontend": 100000
  },
  "service_budgets": {
    "cad-analyzer": 50000,
    "ocr-service": 30000
  }
}
```

#### 检查预算

```bash
# 检查团队预算
python scripts/metrics_budget_controller.py \
  --check platform \
  --type team

# 输出示例：
# 💰 Budget Status for team 'platform':
#    Status: WARNING
#    Budget: 300,000 series
#    Used: 245,000 series (81.7%)
#    Available: 55,000 series
```

#### 模拟变更影响

```bash
# 模拟新增指标的影响
python scripts/metrics_budget_controller.py \
  --simulate-change '{
    "metric_name": "new_feature_metric",
    "team": "platform",
    "service": "cad-analyzer",
    "estimated_cardinality_change": 10000,
    "labels_added": ["user_id", "feature_flag"],
    "reason": "New feature monitoring"
  }'

# 输出：
# 🎯 Budget Decision:
#    Decision: WARN
#    Reason: 团队预算使用率91.7%，接近上限
```

### 3. 分析报告器 (cardinality_analysis_report.py)

**功能**: 生成详细的基数分析报告

#### 生成报告

```bash
# Markdown格式报告
python scripts/cardinality_analysis_report.py \
  --prometheus-url http://localhost:9090 \
  --format markdown \
  --output cardinality_report.md

# JSON格式（用于API集成）
python scripts/cardinality_analysis_report.py \
  --format json \
  --output cardinality_report.json
```

#### 报告内容

- **Top高基数指标**: 成本最高的指标排名
- **异常检测**: 突发增长、标签爆炸
- **优化建议**: 具体的改进措施
- **成本分析**: 当前和预测成本
- **趋势分析**: 基数变化趋势

### 4. 自动优化器 (metrics_auto_optimizer.py)

**功能**: 自动生成优化配置和PR

#### 应用优化

```bash
# 生成优化建议
python scripts/metrics_auto_optimizer.py \
  --prometheus-url http://localhost:9090

# 应用优化并生成配置
python scripts/metrics_auto_optimizer.py \
  --apply \
  --generate-config \
  --output prometheus_optimized.yml

# 生成优化PR
python scripts/metrics_auto_optimizer.py \
  --apply \
  --pr \
  --output optimization_pr.json
```

#### 优化策略

1. **标签合并**: 将细粒度标签合并为粗粒度
2. **标签删除**: 删除ID类高基数标签
3. **降采样**: 对历史数据应用不同保留策略
4. **Recording Rules**: 预聚合常用查询

---

## 🔄 CI/CD集成

### GitHub Actions工作流

#### 自动预算检查 (.github/workflows/metrics-budget-check.yml)

**触发条件**:
- PR中修改了指标相关代码
- 手动触发

**功能**:
- 检测新增指标和标签
- 评估基数影响
- 计算成本增加
- 超预算自动阻断

#### 配置示例

```yaml
env:
  BUDGET_THRESHOLD: 90  # 预算使用超过90%时阻断
  PROMETHEUS_URL: http://prometheus:9090
```

### 集成到现有CI

```yaml
# 在PR检查中添加
jobs:
  metrics-check:
    steps:
      - name: Check Metrics Budget
        run: |
          python scripts/metrics_budget_controller.py \
            --check ${{ github.event.pull_request.head.ref }} \
            --type team

      - name: Generate Report
        if: failure()
        run: |
          python scripts/cardinality_analysis_report.py \
            --format markdown \
            --output pr_comment.md
```

---

## ⚙️ 配置与定制

### 预算配置

```python
# metrics_budget_config.py
BUDGET_CONFIG = {
    # 全局配置
    "global": {
        "max_total_series": 1000000,
        "max_per_metric": 10000,
        "cost_per_series": 0.001  # $/月
    },

    # 团队配额
    "teams": {
        "platform": {
            "budget": 300000,
            "alert_threshold": 0.8,
            "block_threshold": 0.95
        }
    },

    # 优先级配置
    "priorities": {
        "up": 100,  # 最高优先级
        "http_request_duration_seconds": 90,
        "custom_business_metric": 50
    }
}
```

### 优化规则

```yaml
# optimization_rules.yaml
rules:
  - name: merge_user_agents
    type: merge_labels
    pattern: "http_.*"
    config:
      user_agent: browser_family
      detailed_error: error_category

  - name: drop_trace_ids
    type: drop_labels
    pattern: ".*"
    labels:
      - trace_id
      - span_id
      - request_id

  - name: downsample_old_data
    type: retention
    config:
      1h: 1d   # 1小时精度保留1天
      5m: 7d   # 5分钟精度保留7天
      1m: 24h  # 1分钟精度保留24小时
```

---

## 💡 最佳实践

### 标签设计原则

1. **避免高基数标签**
   - ❌ `user_id`, `request_id`, `session_id`
   - ✅ `user_type`, `request_type`, `session_status`

2. **使用有界标签值**
   - ❌ `path="/users/12345/profile"`
   - ✅ `path_pattern="/users/{id}/profile"`

3. **合理的标签数量**
   - 理想: 2-5个标签
   - 警告: 5-8个标签
   - 危险: >8个标签

### 监控策略

1. **分层监控**
   ```
   原始指标 (高精度，短保留)
       ↓
   5分钟聚合 (中精度，中保留)
       ↓
   1小时聚合 (低精度，长保留)
   ```

2. **使用Recording Rules**
   ```yaml
   # 预聚合P95延迟
   - record: http:request_duration:p95:5m
     expr: |
       histogram_quantile(0.95,
         sum(rate(http_request_duration_seconds_bucket[5m]))
         by (service, method, le)
       )
   ```

3. **定期审查**
   - 每周: 检查Top 10高基数指标
   - 每月: 运行完整优化分析
   - 每季度: 审查预算分配

### 成本优化技巧

1. **使用采样**
   ```python
   # 对非关键指标采样
   if random.random() < 0.1:  # 10%采样率
       metric.inc()
   ```

2. **条件记录**
   ```python
   # 只记录错误和慢请求
   if response.status >= 400 or response.duration > 1.0:
       metric.observe(response.duration)
   ```

3. **批量优化**
   ```bash
   # 每月运行优化脚本
   0 0 1 * * python scripts/metrics_auto_optimizer.py --apply --pr
   ```

---

## 🔨 故障排除

### 常见问题

#### Q: Prometheus连接失败
```bash
# 检查Prometheus是否运行
curl http://localhost:9090/-/healthy

# 检查网络连接
telnet localhost 9090

# 使用环境变量
export PROMETHEUS_URL=http://your-prometheus:9090
```

#### Q: 基数计算不准确
```bash
# 强制刷新缓存
python scripts/metrics_cardinality_tracker.py \
  --prometheus-url http://localhost:9090 \
  --refresh-cache

# 增加查询超时
python scripts/metrics_cardinality_tracker.py \
  --timeout 60
```

#### Q: 优化后指标丢失
```bash
# 检查recording rules语法
promtool check rules recording_rules.yml

# 验证标签映射
python scripts/metrics_auto_optimizer.py \
  --dry-run \
  --verbose
```

### 调试模式

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG

# Python调试
python -m pdb scripts/metrics_budget_controller.py --status

# 查看中间数据
python scripts/metrics_cardinality_tracker.py \
  --export cardinality_debug.json
cat cardinality_debug.json | jq '.'
```

---

## 📖 API参考

### MetricsCardinalityTracker

```python
class MetricsCardinalityTracker:
    def __init__(self, prometheus_url: str)
    def get_metric_cardinality(self, metric_name: str) -> CardinalityInfo
    def track_cardinality_trends(self, metrics: List[str] = None) -> List[TrendInfo]
    def identify_high_cardinality_labels(self, metric_name: str, threshold: int = 100) -> List[LabelAnalysis]
    def estimate_total_cost(self) -> Dict[str, Any]
```

### MetricsBudgetController

```python
class MetricsBudgetController:
    def __init__(self, budget_config: BudgetConfig = None)
    def check_budget(self, entity: str, entity_type: str = "team") -> BudgetUsage
    def enforce_budget(self, metric_change: MetricChange) -> Tuple[Decision, str]
    def optimize_budget_allocation(self) -> AllocationPlan
    def get_global_status(self) -> Dict[str, Any]
```

### CardinalityAnalysisReporter

```python
class CardinalityAnalysisReporter:
    def __init__(self, tracker: MetricsCardinalityTracker, controller: MetricsBudgetController = None)
    def generate_top_offenders_report(self, top_n: int = 10) -> Dict[str, Any]
    def detect_anomalies(self) -> List[Anomaly]
    def generate_optimization_recommendations(self) -> List[Recommendation]
    def generate_full_report(self, output_format: str = "json") -> str
```

### MetricsAutoOptimizer

```python
class MetricsAutoOptimizer:
    def __init__(self, optimization_rules: List[OptimizationRule] = None)
    def apply_label_merging(self, metric_info: CardinalityInfo, label_mappings: Dict[str, str]) -> OptimizationResult
    def apply_label_dropping(self, metric_info: CardinalityInfo, labels_to_drop: List[str]) -> OptimizationResult
    def apply_downsampling(self, metric_info: CardinalityInfo, retention_policy: Dict[str, str]) -> Dict[str, Any]
    def generate_optimization_pr(self, optimizations: List[OptimizationResult]) -> Dict[str, Any]
```

---

## 📞 支持

- **问题反馈**: 创建GitHub Issue
- **功能建议**: 提交PR或Issue
- **技术支持**: platform-team@example.com

---

## 📝 更新日志

### v1.0.0 (2025-11-22)
- ✨ 初始版本发布
- ✨ 基数追踪器
- ✨ 预算控制器
- ✨ 分析报告器
- ✨ 自动优化器
- ✨ CI/CD集成
- 📝 完整文档

---

*本指南由CAD ML Platform治理团队维护*