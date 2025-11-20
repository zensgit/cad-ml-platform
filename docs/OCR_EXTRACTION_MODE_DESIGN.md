## OCR Extraction Mode Design (v1)

### 背景
不同 Provider 与解析链（LLM JSON / 正则回填）会产生多种组合路径，需要在结果中暴露 *extraction_mode* 以便：
- 指标分层分析（JSON 成功率 / 正则补全贡献 / 纯正则独立路径）
- 动态阈值调优（仅在 JSON 不完整时才依赖 regex）
- 成本评估（JSON 失败频率与回退触发次数）

### 模式定义
| 模式 | 描述 | 典型场景 | 成本 | 监控重点 |
|------|------|---------|------|----------|
| provider_native | 基础引擎原生输出（Paddle 未做增强） | Paddle 正常返回 | 低 | 召回基线/延迟 |
| json_only | LLM 严格 JSON 返回结构完整 | DeepSeek 结构化成功 | 中 | JSON 合规率 |
| json+regex_merge | JSON 部分字段缺失，正则补齐 | DeepSeek 缺螺纹/粗糙度字段 | 中 | 补齐增益率、假阳性率 |
| regex_only | 无有效 JSON (fallback) 纯正则抽取 | LLM 崩溃/格式错误 | 低-中 | 解析覆盖率、误差分布 |

### 路径逻辑
1. Provider执行（Paddle 或 DeepSeek）
2. 若 DeepSeek：尝试 JSON → 若解析成功且字段齐全 → `json_only`
3. 若 JSON 成功但字段不全 → 正则补齐 → `json+regex_merge`
4. 若 JSON 失败（三级降级最终纯文本）→ 正则抽取 → `regex_only`
5. Paddle 原生直接输出结构（后续可加入正则增强）→ `provider_native`；若补齐 → `regex_only`。

### 质量指标关联
- extraction_mode=provider_native: 作为 CPU 基线；监控与 fallback 触发比率。
- extraction_mode=json+regex_merge: 计算 merge 增益 `(merged_fields - json_fields) / json_fields`。
- extraction_mode=regex_only: 关注 *假阳性率* 与 *关键字段召回*；若低于阈值触发重新请求或策略调整。

### 动态阈值策略挂钩
调整 `confidence_fallback` 时参考：
- 最近 N 次中 `json_only` 占比下降且 `regex_only` 假阳性上升 → 降低阈值避免不必要 DeepSeek 调用。
- `json+regex_merge` 增益率明显（>30%） → 提升阈值以更多使用 DeepSeek。

### 日志字段建议
```json
{
  "trace_id": "...",
  "provider": "deepseek_hf",
  "extraction_mode": "json+regex_merge",
  "confidence": 0.83,
  "calibrated_confidence": 0.86,
  "completeness": 0.75,
  "dimensions_count": 5,
  "symbols_count": 2,
  "fallback_level": "confidence_fallback"
}
```

### 后续扩展 (v2)
- 引入 `structure_coverage`: JSON 中的已填字段 / 期望字段
- 将正则补齐字段标记来源：`source=json|regex` 用于误差分析。
- 语义修正规则：正则识别后调用轻量模型进行实体类型校准。

### 评测建议
在 Golden 集上分别统计四种模式的：
- Dimension Recall / Symbol Recall
- Edge-F1
- Brier Score (置信度校准)
- 平均延迟 / P95

### 风险与缓解
- 正则扩充导致误匹配：加入假阳性计数（与真实标注比对），超过阈值回滚或降级。
- 模式爆炸：保持枚举有限，避免过多细粒度导致监控复杂化。
- JSON 模式频繁降级：通过仪表盘对 `json_only` → `regex_only` 转化率预警。

### 结论
extraction_mode 字段建立了解析路径的可观测性基础，后续可驱动动态成本优化与精度提升策略，是 DeepSeek-OCR 进入生产前的关键工程指标之一。

