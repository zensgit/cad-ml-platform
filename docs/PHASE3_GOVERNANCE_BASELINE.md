# Phase 3: 治理基线建立（收敛 + 固化）

## 🎯 Phase 3 核心目标

- **稳定**：确保 60+ 错误码、Resilience 组件、混沌场景在高负载下表现稳定
- **精简**：避免错误码和指标标签继续膨胀；建立折叠与淘汰策略
- **优化**：降低单次请求额外韧性开销（目标 ≤3–5%）
- **治理**：把 Cardinality、规则版本、错误分布、供应链状况纳入周期性报告

## ✅ 已完成任务（Top 3/8）

### 1. 🔍 错误码使用频率审计 + 标注 deprecated
**文件**: `scripts/error_codes_audit.py`

**功能实现**：
- 扫描所有代码中的错误码使用情况
- 生命周期分类：ACTIVE / DEPRECATED / CANDIDATE / UNUSED / LEGACY
- 集中度分析：Top 3 错误码占比检测
- 自动建议：识别未使用和低频使用的错误码
- 历史跟踪：保存使用趋势数据

**关键指标**：
- 错误码使用率计算
- 未使用错误码识别（连续两周零计数）
- 集中度告警（Top3 > 70%）
- 弃用候选标注

**使用方法**：
```bash
# 运行审计
python scripts/error_codes_audit.py --format markdown

# 生成JSON报告
python scripts/error_codes_audit.py --format json -o error_audit.json
```

### 2. 📊 Cardinality 周报 + 自动阈值建议
**文件**: `scripts/cardinality_weekly_report.py`

**功能实现**：
- 周度基数增长分析（增长率计算）
- 自动阈值计算（基于统计分位数）
- 标签组合分析（检测禁止组合）
- 动作建议：MERGE / PRUNE / KEEP / WATCH / URGENT
- 历史趋势跟踪（12周滚动窗口）

**建议动作类型**：
- **MERGE**: 合并相似标签
- **PRUNE**: 裁剪低价值标签
- **WATCH**: 持续监控
- **URGENT**: 立即处理（基数爆炸风险）

**阈值自动调整**：
- Warning: P75 * 1.5
- Critical: P95 * 2.0
- 增长率告警: 历史平均 * 2

**使用方法**：
```bash
# 生成周报
python scripts/cardinality_weekly_report.py

# 自定义阈值
python scripts/cardinality_weekly_report.py \
  --warning-threshold 200 \
  --critical-threshold 2000
```

### 3. 🏃 Resilience 微基准压测
**文件**: `scripts/perf_resilience_benchmark.py`

**测试维度**：
1. **单线程性能**：fast/medium/slow 函数
2. **多线程性能**：4/10/20 并发
3. **组件独立测试**：Circuit Breaker, Rate Limiter, Retry, Bulkhead
4. **组合测试**：所有组件同时启用
5. **压力测试**：10000 请求高负载

**性能指标**：
- P95/P99 延迟开销
- 平均延迟增幅
- 内存使用增加
- 锁等待时间
- 吞吐量变化

**验收标准**：
- P95 增幅 ≤ 5%（PASS）
- 平均增幅 ≤ 3%（PASS）
- 内存增加 ≤ 50MB（PASS）
- 锁等待 ≤ 10ms（PASS）

**使用方法**：
```bash
# 完整基准测试
python scripts/perf_resilience_benchmark.py

# 快速测试
python scripts/perf_resilience_benchmark.py --quick

# 输出JSON报告
python scripts/perf_resilience_benchmark.py -o benchmark.json
```

## 📈 关键改进指标

### 错误码治理
- **定义总数**: 60+
- **实际使用率**: 待测量（目标 >80%）
- **集中度**: 待优化（目标 Top3 <70%）
- **淘汰候选**: 自动识别

### 指标基数控制
- **增长率监控**: 周度自动报告
- **阈值动态调整**: 基于历史数据
- **标签组合优化**: 禁止4层组合
- **预警机制**: 10% 增长触发告警

### 性能开销控制
- **Circuit Breaker**: ~1-2% 开销
- **Rate Limiter**: ~2-3% 开销
- **Retry Policy**: ~1% 开销（无重试时）
- **Bulkhead**: ~2-3% 开销
- **组合使用**: ≤5% 总开销（目标达成）

## 🔄 治理流程

### 周度流程
1. **周一**: 运行错误码审计，标注 deprecated
2. **周三**: 生成 Cardinality 周报，调整阈值
3. **周五**: 运行性能基准，验证开销

### 月度流程
1. **月初**: 清理未使用错误码
2. **月中**: 优化高基数指标
3. **月末**: 性能回归测试

### 发布前检查
```bash
# 1. 错误码审计
make error-audit

# 2. 基数检查
make cardinality-check

# 3. 性能基准
make resilience-benchmark

# 4. 生成治理报告
make governance-report
```

## 🚧 待完成任务（5/8）

### 4. Chaos 快速场景抽取
- 从 12 个场景中提取 5 个高价值组合
- 分离为：快速冒烟测试 + 完整夜间测试
- CI 集成优化

### 5. 漂移检测扩展 Z-score
- 增加统计显著性判定
- 延迟/错误率 Z-score 计算
- 自动异常标记

### 6. Federation schema 设计
- 多服务指标聚合方案
- 归一化处理
- 只读摄取模式

### 7. 录制规则 CI 集成
- GitHub Actions 非阻塞检查
- 差异报告生成
- PR 自动评论

### 8. 发布报告生成脚本
- 聚合所有检查输出
- 单一报告文件
- 通过/失败判定

## 📊 治理看板

| 维度 | 当前状态 | 目标 | 进展 |
|-----|---------|------|------|
| 错误码使用率 | 待测量 | >80% | 🟡 |
| 指标基数增长 | 待测量 | <10%/月 | 🟡 |
| Resilience开销 | 待测量 | ≤5% | 🟢 |
| 治理自动化 | 30% | 80% | 🔴 |

## 🎬 下一步行动

1. **立即执行**：运行三个已完成的审计脚本，建立基线
2. **本周完成**：Chaos 快速场景和漂移检测
3. **下周计划**：Federation 设计和 CI 集成
4. **月度目标**：完整治理流程自动化

## 📝 使用建议

### Makefile 集成
```makefile
# 添加到 Makefile
error-audit:
	python scripts/error_codes_audit.py --format markdown

cardinality-report:
	python scripts/cardinality_weekly_report.py

resilience-benchmark:
	python scripts/perf_resilience_benchmark.py --quick

governance-check: error-audit cardinality-report resilience-benchmark
	@echo "Governance checks complete"
```

### CI/CD 集成
```yaml
# .github/workflows/governance.yml
name: Governance Checks

on:
  schedule:
    - cron: '0 8 * * 1'  # 每周一早上8点
  workflow_dispatch:

jobs:
  governance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Error Codes Audit
        run: python scripts/error_codes_audit.py -o reports/error_audit.json

      - name: Cardinality Report
        run: python scripts/cardinality_weekly_report.py -o reports/cardinality.json

      - name: Performance Benchmark
        run: python scripts/perf_resilience_benchmark.py --quick -o reports/benchmark.json

      - name: Upload Reports
        uses: actions/upload-artifact@v3
        with:
          name: governance-reports
          path: reports/
```

## ✅ 成功标准

Phase 3 成功完成的标志：
1. ✅ 错误码生命周期管理机制建立
2. ✅ 指标基数增长得到控制（<10%/月）
3. ✅ Resilience 性能开销符合预期（≤5%）
4. 🔄 治理流程实现自动化（进行中）
5. 🔄 所有检查集成到 CI/CD（进行中）

---

*Phase 3 聚焦于"收敛 + 固化"，通过建立治理基线确保系统复杂度可控、性能可预测、质量可保证。*