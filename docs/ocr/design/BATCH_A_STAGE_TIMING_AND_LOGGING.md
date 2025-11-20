## Batch A: 阶段计时与结构化日志设计总结 (v1)

### 1. 背景与动机
现有OCR仅提供整体处理耗时与基础confidence。缺乏细粒度阶段数据，难以定位性能瓶颈，也无法支撑后续动态阈值、熔断策略与多证据置信度融合。Batch A目标：为后续优化建立可观测基线。

### 2. 目标
- 采集 preprocess / infer / parse / postprocess 四阶段耗时
- 将阶段耗时写入Prometheus指标 `ocr_stage_duration_seconds{provider,stage}`
- 将阶段耗时集成到 `OcrResult.stages_latency_ms` 以便缓存与离线分析
- 增强日志结构，便于追踪 fallback 与错误

### 3. 非目标
- 不进行推理性能优化
- 不引入动态阈值或熔断逻辑（后续批次实现）

### 4. 设计概览
新增 `StageTimer` 简单类 (毫秒精度)；在每个 provider 的 `extract()` 流程中包裹阶段 start/end；阶段完成后写指标+返回时装入 `OcrResult`。

### 5. 数据结构与字段
`OcrResult.stages_latency_ms: Dict[str,int]` eg: `{"infer": 1834, "parse": 12}`

### 6. 核心实现
文件: `src/core/ocr/stage_timer.py`
使用: provider中创建 `timer = StageTimer()` → `timer.start(stage)` / `timer.end(stage)` → 获得 `durations_ms()`。

### 7. Metrics
- 新增/使用: `ocr_stage_duration_seconds` Histogram；buckets: `[0.005,0.01,0.05,0.1,0.5,1.0,2.0]`
- 记录 infer / parse / preprocess / postprocess 四阶段
- 保留原 `ocr_processing_duration_seconds` 作端到端统计

### 8. Logging 规划 (v1)
当前 `utils/logging.py` 支持基础字段；后续 Batch D/E 将补充 extraction_mode、dimensions_count 等；本批次不强制在所有调用点添加info日志避免噪声。

### 9. 修改的文件与摘要
- `src/core/ocr/stage_timer.py` 新增计时工具
- `src/core/ocr/providers/paddle.py` 插入阶段计时+指标+写入result
- `src/core/ocr/providers/deepseek_hf.py` 插入阶段计时+指标+写入result

### 10. 风险与回退策略
风险: 阶段计时逻辑异常导致结果为空或指标标签爆炸。缓解：阶段名称固定不从外部输入；异常时不写该阶段结束时间。
回退: 若出现高错误率，可临时禁用阶段指标（注释observe调用）。

### 11. 验收标准
- Provider返回的 `OcrResult.stages_latency_ms` 包含至少 infer / parse 两阶段
- Prometheus端点暴露新指标名称，示例标签: `provider="paddle", stage="infer"`
- 不影响现有测试全部通过

### 12. 测试用例（规划）
- 单元测试: 构造fake provider调用后验证 `stages_latency_ms` 键集合
- 现阶段：依赖端到端 smoke 测试 + 手动检查 `/metrics` 是否含有 `ocr_stage_duration_seconds`

### 13. 后续扩展点
- v2: 增加 merge 阶段 (LLM JSON + regex 合并)
- 增加 token 数 / 输出长度关联指标 (infer阶段) 支撑吞吐量分析
- 阶段异常超时监控 (infer阶段 > 阈值自动打点错误)

