# Drift Baseline Stale Runbook

## Summary
当告警 `DriftBaselineStale` 触发时，表示材料或分类预测漂移基线已经超过允许的最大年龄 (`DRIFT_BASELINE_MAX_AGE_SECONDS`)。过期基线可能导致漂移得分失真，无法及时反映当前数据分布变化。

## Detection
Prometheus 规则通过 `baseline_material_age_seconds` 与 `baseline_prediction_age_seconds` 对比最大年龄阈值触发告警。

## Impact
- 漂移监控不准确，可能漏报或误报分布变化。
- 新模型上线或数据渠道变更后无法及时更新基线。

## Immediate Actions
1. 在 Grafana 查看当前材料与预测分布趋势。
2. 评估是否仍需旧基线：
   - 若数据分布已经明显改变，执行基线重置。
3. 调用 API `POST /api/v1/drift/reset` 重置基线（确保样本数达到 `DRIFT_BASELINE_MIN_COUNT` 后会自动重新建立）。
4. 若长时间未达到最小样本数，考虑降低 `DRIFT_BASELINE_MIN_COUNT` 或加速数据回放。

## Preventive Actions
- 定期观察基线年龄面板，避免接近过期才处理。
- 部署流程中（模型/规则更新）自动调用基线重置。
- 根据数据增长速度调整 `DRIFT_BASELINE_MAX_AGE_SECONDS`。

## Validation After Reset
1. 使用 `GET /api/v1/drift` 检查 baseline_* 字段是否已清空并重新积累。
2. 达到最小样本后确认 `status=ok` 且漂移得分正常（常见 <0.2）。

## Configuration Reference
- `DRIFT_BASELINE_MIN_COUNT`: 建立基线所需最小样本数。
- `DRIFT_BASELINE_MAX_AGE_SECONDS`: 基线最大允许年龄。

## Related Metrics
- `baseline_material_age_seconds`
- `baseline_prediction_age_seconds`
- `material_distribution_drift_score`
- `classification_prediction_drift_score`

## Escalation
若多次重置后仍快速过期或无法建立基线，联系平台工程团队排查数据流是否中断。

