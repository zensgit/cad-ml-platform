# Batch G 日志增强与 Quickstart 文档 设计总结

## 目标
- 丰富关键路径日志，保持隐私安全（仅记录 image_hash 等非敏感信息）。
- 提供快速上手文档，统一本地验证与监控入口。

## 实现
- `OcrManager` 在完成一次请求后输出结构化日志：`ocr.manager.extract`，包含
  - `provider/latency_ms/fallback_level/image_hash/stage`。
- 新增 `docs/OCR_GUIDE.md`：
  - 环境验证、服务启动、请求示例、指标说明与评测入口。

## 后续
- 扩展日志字段：`extraction_mode/dimensions_count/symbols_count/stages_latency_ms`。
- 在 API 层统一使用 `utils/logging.setup_logging()` 并确保 /metrics 暴露。

---
Batch G 完成。建议继续围绕 Week1 Gate 提升召回（解析器微调、Paddle 预处理）或进入 Week2 评测与鲁棒性工作。

