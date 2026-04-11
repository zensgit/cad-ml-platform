# CAD ML Platform - 快速参考卡片

**版本**: v1.0 | **日期**: 2025-11-24 | **状态**: Day 0完成

---

## 🚀 快速命令

### 每日工作流
```bash
# 1. 早上9点 - 查看进度
./scripts/track_progress.sh 1

# 2. 开发工作...

# 3. 下午4点 - 执行检查点
./scripts/daily_checkpoint.sh 1

# 4. 提交代码前 - 验证指标
python3 scripts/check_metrics_consistency.py
```

### 测试命令
```bash
# 运行所有测试
pytest -v

# 带覆盖率
pytest --cov=src --cov-report=html

# 只运行快速测试
pytest -v -m "not slow"

# 运行特定模块
pytest tests/unit/test_batch_similarity.py -v
```

### 性能测试
```bash
# 建立基线（Day 0已完成）
python3 scripts/performance_baseline.py

# Day 6对比
python3 scripts/performance_baseline.py
# 手动对比: reports/performance_baseline_day0.json vs day6.json
```

---

## 📁 重要文件位置

### 配置文件
```
config/feature_flags.py                    # Feature开关配置
config/grafana/dashboard_main.json         # Dashboard配置（待创建）
config/prometheus/recording_rules.yml      # 录制规则（待创建）
config/prometheus/alert_rules.yml          # 告警规则（待创建）
```

### 脚本工具
```
scripts/daily_checkpoint.sh                # 每日检查点
scripts/track_progress.sh                  # 进度查询
scripts/performance_baseline.py            # 性能基线
scripts/check_metrics_consistency.py       # 指标验证
```

### 文档
```
IMPLEMENTATION_TODO.md                     # 详细任务清单
IMPLEMENTATION_RESULTS.md                  # 执行成果文档
DEVELOPMENT_SUMMARY.md                     # 开发摘要
QUICK_REFERENCE.md                         # 本文档
```

### 测试数据
```
tests/fixtures/v4_test_data.py             # v4测试用例
reports/performance_baseline_day0.json     # 性能基线
reports/daily_checkpoints/                 # 检查点报告
```

---

## 🎯 当前状态快照

### 项目统计
```
Tests: 461 total
Coverage: 82%
Metrics: 70 defined (100% 导出一致性)
Endpoints: 50 total
Files Modified: 15 (uncommitted)
```

### Feature Flags状态
```
✗ v4_enabled                    # v4特征未启用
✓ v4_surface_algorithm: simple  # 使用simple模式
✓ opcode_mode: blocklist        # 安全blocklist模式
✓ drift_auto_refresh           # Drift自动刷新启用
✓ cache_tuning_experimental    # 缓存调优实验性
• batch_similarity_max_ids: 200 # 批量上限200
```

### 性能基线（Day 0）
```
Feature Extraction v3:    p95 = 1.28ms
Feature Extraction v4:    p95 = 1.54ms (+20.8% ⚠️)
Batch Similarity (5):     p95 = 6.30ms
Batch Similarity (20):    p95 = 27.61ms
Batch Similarity (50):    p95 = 55.05ms
Model Cold Load:          p95 = 55.04ms
```

---

## 📋 6天路线图

### Day 1 (今日) - Phase A 稳定性
**AM**:
- Redis宕机孤儿清理测试
- Faiss批量相似度降级测试
- 维护端点错误结构化

**PM**:
- 模型回滚健康测试
- 后端重载失败测试

**目标**: 新增7个测试，覆盖率90%+

---

### Day 2 - Phase A收尾 + Phase B开始
**AM**:
- 降级迁移链统计
- 空结果拒绝计数

**PM**:
- 缓存调优端点
- 迁移维度差异直方图

**目标**: 2个新端点，2个新指标

---

### Day 3 - Phase B + Phase C-1
**AM**:
- Dashboard框架（6面板）
- Prometheus录制规则

**PM**:
- Opcode Audit模式
- 安全文档
- v4算法预研

**目标**: Dashboard可用，audit模式上线

---

### Day 4 - Phase C-2 + Phase D-1
**AM**:
- 接口校验扩展
- 回滚层级3
- v4 surface_count基础

**PM**:
- v4 shape_entropy平滑
- 性能对比测试

**目标**: v4基础版本，overhead <5%

---

### Day 5 - Phase D-2 + Phase E-1
**AM**:
- 迁移preview/trends端点
- Dashboard补全（+6面板）

**PM**:
- Alert规则完整版
- 文档全面更新

**目标**: Dashboard 100%，文档齐全

---

### Day 6 - Phase E-2 + Phase F
**AM**:
- Rules验证
- CI集成
- 性能基线对比

**PM**:
- 回归测试
- 缓冲任务
- 最终验证

**目标**: 测试100%通过，无状态耦合

---

## 🚨 关键风险

### 🔴 v4性能风险
- **现状**: +20.8% overhead（目标: <5%）
- **缓解**: Simple模式fallback，Day 4决定是否发布

### 🟡 时间进度风险
- **现状**: Day 1任务量大
- **缓解**: 已调整30%，Day 6缓冲

### 🟡 覆盖率风险
- **现状**: 82% → 87%需+5pp
- **缓解**: P0优先≥90%，分级要求

---

## 🎯 质量门槛

### P0 (必须达成)
- [ ] 所有Phase A测试通过
- [ ] 安全audit模式上线
- [ ] v4基础版本可用
- [ ] 核心文档更新

### P1 (强烈建议)
- [ ] Dashboard 12面板就绪
- [ ] 缓存调优端点可用
- [ ] Prometheus rules完整
- [ ] 性能基线验证通过

### P2 (时间允许)
- [ ] v4 advanced模式
- [ ] Drift export/import
- [ ] Backend reload安全token

---

## 🔧 常用代码片段

### Feature Flag检查
```python
from config.feature_flags import V4_ENABLED, OPCODE_MODE

if V4_ENABLED:
    features = extract_features_v4(doc)
else:
    features = extract_features_v3(doc)
```

### 指标记录
```python
from src.utils.analysis_metrics import (
    analysis_requests_total,
    analysis_stage_duration_seconds,
)

# Counter
analysis_requests_total.labels(status="success").inc()

# Histogram
with analysis_stage_duration_seconds.labels(stage="parse").time():
    parse_cad_file(data)
```

### 结构化错误
```python
from src.core.errors_extended import build_error, ErrorCode

return build_error(
    code=ErrorCode.INPUT_VALIDATION_FAILED,
    stage="batch_similarity",
    message="Batch size exceeds limit",
    batch_size=350,
    max_batch=200,
)
```

### 测试数据使用
```python
from tests.fixtures.v4_test_data import get_test_case

test_case = get_test_case("complex")
doc = test_case["doc"]
expected_entropy = test_case["expected_entropy_range"]

result = extract_v4_features(doc)
assert expected_entropy[0] <= result["entropy"] <= expected_entropy[1]
```

---

## 📞 紧急联系

### 阻塞问题上报
- **渠道**: Slack #cad-ml-dev
- **响应时间**: < 2小时
- **升级**: 超过4小时未解决自动升级

### 每日站会
- **时间**: 9:30 AM
- **时长**: 15分钟
- **地点**: Zoom / 会议室

### 检查点报告
- **时间**: 4:00 PM
- **提交**: 运行 `daily_checkpoint.sh`
- **审查**: 当晚8:00 PM前

---

## 💡 最佳实践

### 开发流程
1. ✅ 早上查看进度（track_progress.sh）
2. ✅ 创建feature分支（git checkout -b feature/xxx）
3. ✅ 实现功能 + 单测（TDD优先）
4. ✅ 提交前验证（check_metrics_consistency.py）
5. ✅ 下午检查点（daily_checkpoint.sh）
6. ✅ 代码审查（PR提交）

### 测试原则
- 🎯 新代码覆盖率 ≥90%
- 🎯 边界case必须覆盖
- 🎯 错误路径必须测试
- 🎯 性能测试标记 @pytest.mark.slow

### 提交规范
```bash
git commit -m "feat: 添加批量相似度空结果拒绝计数

- 新增 analysis_rejections_total{reason=batch_empty_results} 指标
- 扩展 batch_similarity 端点检测逻辑
- 添加 4 个测试用例覆盖边界场景

Closes #123"
```

---

## 📊 预期成果

### Day 6目标
```
Tests: 461 → 520+ (+13%)
Coverage: 82% → 87%+ (+5pp)
Metrics: 70 → 78+ (+11%)
Endpoints: 50 → 59 (+18%)
Dashboard Panels: 0 → 12
Alert Rules: 0 → 8+
```

### 新增功能
- ✅ Feature flags系统
- 🔄 缓存调优算法
- 🔄 v4特征提取
- 🔄 Opcode审计模式
- 🔄 迁移预览/趋势
- 🔄 3级模型回滚
- 🔄 Grafana Dashboard
- 🔄 Prometheus告警

---

## 🔗 快速链接

### 文档
- [实施TODO](./IMPLEMENTATION_TODO.md)
- [执行成果](./IMPLEMENTATION_RESULTS.md)
- [开发摘要](./DEVELOPMENT_SUMMARY.md)

### 配置
- [Feature Flags](./config/feature_flags.py)
- [测试数据](./tests/fixtures/v4_test_data.py)

### 脚本
- [检查点](./scripts/daily_checkpoint.sh)
- [进度](./scripts/track_progress.sh)
- [性能](./scripts/performance_baseline.py)
- [指标验证](./scripts/check_metrics_consistency.py)

---

**保持此文档在手边，随时参考！**

*Last Updated: 2025-11-24*
