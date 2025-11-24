# Drift Monitoring Runbook

## Purpose
监控材料分布与分类预测分布的稳定性，及时发现数据源变化、模型退化或输入异常。

## Key Metrics
- `material_distribution_drift_score` PSI近似 (0-1)
- `classification_prediction_drift_score` PSI近似 (0-1)
- Recording Rules: `MaterialDriftHigh`, `ClassificationDriftHigh` alerts trigger when 15m平均 >0.3。

## Baseline Logic
1. 收集连续样本直到达到 `DRIFT_BASELINE_MIN_COUNT` (默认100)。
2. 首批样本冻结为基线并持久化到 Redis。
3. 后续每次分析更新当前分布并计算漂移分数。

## Common Causes
- 数据源材料偏移（新项目集中使用单一材料）。
- 模型版本更新导致分类映射变化。
- 批量重复文件或测试流量注入。
- 解析失败导致默认材料/类型占比异常增大。

## Investigation Steps
1. 在 Grafana 查看材料/分类堆叠条形图是否单一类别激增。
2. 查看最近模型重载记录 `model_reload_total` 对应版本是否变更。
3. 检查解析错误或输入拒绝指标是否突增 (`analysis_errors_total`, `analysis_rejections_total`)。
4. 采样高频材料文件，确认其元数据是否真实变化。

## Remediation
- 若真实业务变化：执行 `/api/v1/analyze/drift/reset` 重置基线以适配新分布。
- 若异常文件造成：隔离特殊数据源或修复上游产生的格式问题后再重置基线。
- 模型退化：回滚到上一个稳定模型版本 (`/api/v1/analyze/model/reload` with previous path)。

## Reset Modes (未来增强)
- soft: 重置漂移分数但保留当前观察数据。
- hard: 清空所有观察与基线，重新开始采样。

## Alert Tuning
默认阈值 0.3 适合中低类别数分布；若材料或分类种类非常多，可降低至 0.2 防止漏报，或提高至 0.4 避免频繁告警。

## Prevention
- 维持多样化测试集，避免批量单类文件集中推送。
- 在模型重载前执行小流量影子测试验证预测分布差异。
- 设置材料与分类字段输入校验，减少缺失值自动回退为默认值。

