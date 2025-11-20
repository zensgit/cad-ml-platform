# Batch D 动态阈值（RollingStats/EMA）设计总结

## 目标
让置信度回退阈值依据近期观测到的（校准后）置信度自适应，避免固定阈值在不同图纸分布下触发率过高或过低。

## 实现方案
- RollingStats 实现：`src/core/ocr/rolling_stats.py`，使用指数移动平均（EMA），alpha=0.2，首值直接赋值。
- 指标：
  - `ocr_confidence_ema`（Gauge）：当前 EMA 值。
  - `ocr_confidence_fallback_threshold`（Gauge）：当前动态阈值。
- OcrManager 集成：
  - 在每次请求后更新 EMA，优先使用 `calibrated_confidence`，否则使用原始 `confidence`。
  - 动态阈值公式：`threshold = clamp(ema - 0.05, 0.6, 0.95)`。
  - 回退判断使用 `calibrated_confidence` 与动态阈值比较。

## 代码变更
- 新增：`src/core/ocr/rolling_stats.py`
- 更新：`src/core/ocr/manager.py`（EMA更新、阈值更新、比较项改为 calibrated_confidence）
- 指标：`src/utils/metrics.py` 新增 Gauge，并在 Dummy 实现中补充 `set` 方法。

## 测试
文件：`tests/ocr/test_dynamic_threshold.py`
- `test_threshold_adapts_downward`：EMA 下降时阈值不应上升。
- `test_threshold_bounds`：极低置信度场景下，阈值仍被限制在 [0.6, 0.95]。
测试使用 Dummy Provider 返回预设置信度序列。

## 局限与后续
- 目前仅基于置信度单一信号，未引入 completeness、最近 fallback 比例、解析错误率等多证据融合。
- 未持久化 EMA 状态；进程重启后需要重新收敛。
- 未设置最小样本数门槛（可加入 warmup N）

## 影响与风险
- 阈值变化将影响 DeepSeek 回退触发率，需要在 `/metrics` 中观察，并在 CHANGELOG 记录阈值调整策略。

---
Batch D 完成；下一步建议进入 Batch E（分布式限流 + 熔断器）。

