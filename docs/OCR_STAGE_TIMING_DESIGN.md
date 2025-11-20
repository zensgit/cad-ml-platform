## OCR Stage Timing Design (v1)

### 目标
精细采集各阶段耗时: preprocess / infer / parse / merge / postprocess, 支撑:
- 性能瓶颈定位 (GPU vs 正则解析 vs JSON解析)
- 动态阈值调优时的成本权衡 (infer 加权)
- 熔断器与超时策略制定

### 方案
Provider 内注入计时器:
```python
class StageTimer:
    def __init__(self): self._t = {}
    def start(self, name): self._t[name] = [time.time(), None]
    def end(self, name): self._t[name][1] = time.time()
    def durations(self):
        return {k: int((v[1]-v[0])*1000) for k,v in self._t.items() if v[1]}
```

DeepSeek 流程映射:
1. preprocess (图像 bytes 简单校验)  
2. infer (LLM 生成/或 stub)  
3. parse (JSON / Markdown / regex)  
4. merge (JSON + regex 合并)  
5. postprocess (校准/构建 OcrResult)

指标: `ocr_stage_duration_seconds{provider,stage}` 在 `end()` 处 `observe`. 总耗时仍用现有 histogram。

### 数据结构
`OcrResult.stages_latency_ms = {stage: ms}` 持久化到缓存用于后续离线分析。

### 边界与降级
- 若某阶段异常未结束 -> 不写入该阶段时长。
- regex-only 模式不写 merge 阶段。

### 迭代展望 (v2)
- 添加 token 数统计 (LLM 输出长度) 与 infer 阶段关联。
- 指标与输入分辨率/图像大小关联标签 (避免标签爆炸, 通过 bucket 化)。

