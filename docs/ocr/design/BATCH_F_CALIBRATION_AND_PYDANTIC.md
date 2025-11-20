# Batch F 多证据置信度校准与 Pydantic v2 清理 设计总结

## 目标
- 用多证据（原始 provider 置信度 + 解析 completeness）生成更可靠的 `calibrated_confidence`。
- 清理 Pydantic v2 警告：默认可变字段使用 `default_factory`。

## 实现
- `src/core/ocr/calibration.py` 新增 `MultiEvidenceCalibrator`（轻量版）：
  - 输入：`raw_confidence` 与 `completeness`；
  - 输出：加权平均（默认 `w_raw=0.6, w_completeness=0.4`）。
- `OcrManager` 集成：统一在主路径与回退路径计算 `calibrated_confidence`。
- Pydantic v2 清理：
  - `OcrResult` 使用 `Field(default_factory=...)` 修正列表/字典默认值。

## 测试
文件：`tests/ocr/test_calibration.py`
- 权重均衡与缺失输入处理。

## 后续规划
- 将 `confidence_calibrator.py` 的更完整方法（如 isotonic + DS 融合）纳入多证据框架。
- 在 Golden 评测中产出 Brier Score 与可靠性曲线，反馈权重。
- 将 `calibrated_confidence` 直方图纳入指标，观察分布与阈值互动。

---
Batch F 完成。下一步建议进入 Batch G（日志字段扩展 + 文档 Quickstart 更新），或回到 Week1 Gate 提升召回与解析精度。

