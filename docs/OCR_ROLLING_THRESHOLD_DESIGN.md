## Rolling Confidence Fallback Threshold Design (v1)

### 问题
固定 `confidence_fallback=0.85` 可能在不同样本分布下过高或过低，导致：
- 过高：频繁调用高成本 DeepSeek，成本上升
- 过低：关键字段缺失不触发增强，解析质量下降

### 目标
动态调节阈值：基于最近窗口内 (N=200) 的 calibrated_confidence 与增强增益。

### 数据收集
对每次请求记录:
```json
{
  "trace_id": "...",
  "raw_confidence": 0.81,
  "calibrated_confidence": 0.84,
  "fallback_triggered": true,
  "added_dimensions": 2,
  "extraction_mode": "json+regex_merge"
}
```

### 增益计算
`gain = added_dimensions + added_symbols`
`gain_rate = gain / (json_dimensions + 1)` 防止除零。

### 调节策略
每分钟评估：
1. 统计 fallback 成功比率 `fallback_success = merges_with_gain / total_fallbacks`
2. 若 `fallback_success > 0.5` 且平均 `calibrated_confidence` 接近当前阈值 (±0.02)，提升阈值 +0.01 (上限 0.90)。
3. 若 `fallback_success < 0.2` 且 fallback 触发率 > 30%，降低阈值 -0.01 (下限 0.75)。
4. 使用 EMA 平滑：`new = alpha*computed + (1-alpha)*old`，alpha=0.6。

### 数据结构
`RollingStats`:
```python
class RollingStats:
    def __init__(self, maxlen=200): self.items=deque(maxlen=maxlen)
    def add(event): self.items.append(event)
    def compute(): ... returns suggested_threshold
```

Manager 在定时任务或首次调用间隔>60s 时更新内部 `self.confidence_fallback`。

### 安全保障
- 阈值变动写入日志与 metrics: `ocr_threshold_changes_total{old,new}`。
- 若 Redis 可用，将当前阈值存入 `ocr:threshold` 便于多实例一致。

### 风险与缓解
- 窗口不足：若 <50 样本不调整。
- 剧烈波动：EMA 平滑 + 上下限约束。
- 多实例不同步：Redis广播或周期性对比差异>0.02 时同步最小值。

### v2 拓展
- 引入置信度校准误差 (Brier) → 若误差变大则降低阈值。
- 基于成本模型 (GPU 秒级占用) 做成本-质量平衡函数。

