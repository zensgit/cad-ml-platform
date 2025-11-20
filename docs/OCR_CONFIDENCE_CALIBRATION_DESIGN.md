## Confidence Calibration Multi-Evidence Design (v1)

### 背景
当前置信度仅来源于 Provider 原始分值 + 简单 completeness 融合。提升校准质量可降低不必要 fallback 并稳定指标。

### 目标
构建多证据校准：`raw_confidence`, `completeness`, `structure_coverage`, `fallback_history`, `parse_error_rate` → 生成校准后概率接近真实正确率。

### 证据定义
| 证据 | 描述 | 范围 | 采集方式 |
|------|------|------|----------|
| raw_confidence | Provider原始打分 | 0-1 | 模型输出或内部置信度 | 
| completeness | 关键 token 覆盖度 | 0-1 | 解析后统计 | 
| structure_coverage | JSON字段填充比例 | 0-1 | 期望字段计数 | 
| fallback_history | 最近 N 次是否 fallback | 0-1 | Rolling 窗口 | 
| parse_error_rate | regex/JSON parse 错误率 | 0-1 | 错误计数/总解析 | 

### 融合公式 (初始)
加权线性模型:
`calibrated = w1*raw + w2*completeness + w3*coverage + w4*(1-parse_error_rate) - w5*fallback_bias`
参数建议: w1=0.5, w2=0.2, w3=0.15, w4=0.1, w5=0.05

### 动态权重微调
使用黄金样本：每次评测后计算 Brier Score，若某证据残差贡献高则提升其权重 (限制单次变化 ≤0.05)。

### 结构覆盖计算
期望字段集合: diameter|radius|thread|surface_roughness
`coverage = filled / expected_found_tokens`

### 数据结构
`CalibrationEvidence`:
```python
class CalibrationEvidence(BaseModel):
    raw_confidence: float
    completeness: float
    coverage: float
    parse_error_rate: float
    fallback_recent: float  # fraction
```

校准器接口:
```python
class MultiEvidenceCalibrator:
    def calibrate(e: CalibrationEvidence) -> float:
        ...
```

### 评测指标
主要关注：Brier Score, Reliability diagram (分桶正确率 vs 置信度)。

### 风险与缓解
- 过度拟合：控制权重调整频率 (每日一次)。
- 缺少证据：若解析失败则降级到 raw_confidence。
- coverage 与 completeness 重叠：保持定义独立性 (coverage 基于字段填充数)。

### v2 拓展
- 使用轻量回归模型 (Platt缩放 + 额外特征)。
- 引入时序特征 (最近 5 分钟 fallback 速率)。
- 置信度不确定性区间 (基于 Beta 分布估计)。

