# 📋 DeepSeek OCR Integration TODO List (Upgraded)

> 3周实施 + 1周 Buffer | 团队: 1.5~2 开发 + 0.5 测试 | 预计总工时 ~118h (合并与优化后)
> 原始计划在此基础上升级: 指标公式 / 安全合规 / 数据集规范 / 错误体系 / 监控强化 / 时间再分配

---
## 🧪 指标定义与公式 (Metrics & Formulas)

维度召回率 (Dimension Recall)
```
dimension_recall = matched_dimensions / ground_truth_dimensions

匹配条件:
abs(value_pred - value_gt) <= max(0.05 * value_gt, tolerance_gt_if_present)
单位统一为 mm 后比较；直径/半径文字前缀忽略；螺纹 Mx*y 中 x 为公称直径用于比较。
```

符号召回率 (Symbol Recall)
```
symbol_recall = matched_symbols / ground_truth_symbols
匹配以 normalized_form (例如 ⟂ → perpendicular) 对齐。
```

标题栏字段准确率 (Title Block Accuracy)
```
title_block_accuracy = correctly_extracted_fields / required_fields
required_fields = {drawing_number, material, part_name}
```

Edge-F1 (复杂/极端样本边缘框质量)
```
precision = TP / (TP + FP)
recall = TP / (TP + FN)
edge_f1 = 2 * precision * recall / (precision + recall)
TP 条件: IoU(pred, gt) >= 0.5 且文本相似度 >= 0.8
```

延迟指标 (Latency)
```
冷启动: 第一请求 DeepSeek 模型加载时间 (target < 60s)
热启动: 模型已加载后的推理延迟; 统计 P50 / P95 / P99
目标 Week1: Paddle P95 < 2s, DeepSeek P95 < 5s, P50 < 2s
```

吞吐指标 (Week3 可选)
```
tokens_per_second_vllm >= 2 * tokens_per_second_hf
```

缓存命中率
```
cache_hit_rate = cache_hits / (cache_hits + cache_misses)
目标 Week1 ≥ 40%, Week2 ≥ 60%
```

置信度校准质量 (Brier Score)
```
brier = mean( (p_i - y_i)^2 ) 目标 Week1 < 0.20, Week2 < 0.15
```

---
## 🔐 安全与合规 (新增要求)

- MIME 白名单: image/png, image/jpeg, application/pdf
- 最大分辨率: 任一边 > 2048px 时触发裁剪或拒绝 (避免 OOM)
- PDF 安全检查: 禁止嵌入脚本 / XFA / 加密；拒绝含 JS 的对象
- 文件大小限制: 50MB (413 返回)
- 页数限制: 20 页 (PDF) 超出返回 422
- PII 日志策略: 日志仅存储 image_hash 与统计指标；材料/图号等通过结构化字段输出，不写入原始内容行级日志
- 许可证审查: DeepSeek (MIT), PaddleOCR (Apache 2.0)；权重文件不混入非再分发许可
- 输入校验: 仅接受二进制文件字段 `file`；拒绝多文件上传 (后续批处理单独端点)
- 超时: 全流程硬超时 30s (`OCR_TIMEOUT_MS`)

---
## 📂 数据集规范 (tests/ocr/golden/metadata.yaml)
```yaml
dataset:
  version: "v1.0"
  categories:
    easy: {count: 10, description: "清晰图纸"}
    medium: {count: 10, description: "一般质量"}
    hard: {count: 5, description: "模糊/倾斜"}
    edge: {count: 5, description: "极端案例 (低对比度/复杂装配)"}
  annotation_schema:
    dimensions: {type, value, tolerance, unit, bbox}
    symbols: {type, value, normalized_form, bbox}
    title_block: {drawing_number, material, part_name}
  evaluation_rules:
    dimension_match: "abs(pred - gt) <= max(0.05*gt, tolerance_gt_if_present)"
    unit_normalization: "全部换算为 mm"
    symbol_normalization: "映射到 canonical form"
```

Golden 集版本化: 新增字段 `dataset.version`；变更需更新 `CHANGELOG` 与缓存失效策略 (prompt_version + dataset_version 纳入 key)。

---
## 🧱 错误与异常体系

统一异常类:
```python
class OcrError(Exception):
    def __init__(self, code: str, message: str, provider: str, stage: str):
        super().__init__(message)
        self.code = code        # OCR_001 - OCR_999
        self.provider = provider  # paddle|deepseek_hf|deepseek_vllm
        self.stage = stage        # load|preprocess|inference|parse|normalize|route
```

错误分类枚举:
```
timeout | parse_error | oom | provider_down | invalid_input | degraded | schema_violation
```

返回结构 (API):
```json
{
  "status": "failed",
  "error": {"code": "OCR_013", "type": "parse_error", "message": "JSON schema mismatch"}
}
```

---
## 🧩 缓存键设计 (含 Prompt 版本)
```
key = f"ocr:{sha256(image_bytes)}:{provider}:{prompt_version}:{crop_cfg_hash}:{dataset_version}"  # dataset_version 用于评测回溯
TTL 默认 3600s (MVP)，Week2 评估热度调整 (LRU/分级 TTL)
```

---
## 🖥️ 硬件基线 Profiles
```yaml
environment_profiles:
  development:
    cpu: 4 cores
    memory: 8GB
    gpu: null
    expected_qps: 2
  gpu_workstation:
    cpu: 8 cores
    memory: 16GB
    gpu: T4|RTX3060 (>=6GB VRAM)
    expected_qps: 5-10
  production_gpu:
    cpu: 16 cores
    memory: 64GB
    gpu: A10|L40 (>=24GB)
    expected_qps: 15-25 (混合 provider)
```

监控 VRAM: `torch.cuda.mem_get_info()` → Gauge `ocr_gpu_memory_mb`

---
## 🛠️ 开发周计划 (调整后)

### Week 1 — MVP & 安全基线

Day 1 脚手架 + 安全初始化 (6h)
- [ ] 目录结构 + base 抽象 + OcrResult Schema (2.5h)
- [ ] Pydantic 模型 (DimensionInfo / SymbolInfo / TitleBlock / OcrResult) (2h)
- [ ] 环境验证脚本 `scripts/verify_environment.py` (1h)
- [ ] 安全检查模块 `security/input_validator.py` (0.5h)
- 质量门控: mypy / flake8 / env 脚本成功 / MIME 拒绝无效类型

Day 2 Paddle Provider + 基础预处理 + 测试 (9h)
- [ ] 初始化封装 + detect/recognize (5h)
- [ ] 预处理 (resize / denoise / binarize 可选) (2h)
- [ ] bbox→结构映射 (1h)
- [ ] 单元测试: 3 样本 + bbox 映射测试 (1h)
- 质量门控: 3 样本延迟 P95 < 2s, 内存 <1.5GB

Day 3 DeepSeek-HF Provider + 降级策略 + 测试 (9h)
- [ ] 懒加载 + asyncio.Lock() (2h)
- [ ] 超时封装 (1h)
- [ ] 三级降级: JSON → Markdown fenced → 原始文本 + regex (3h)
- [ ] Prompt 模板 & 版本 (2h)
- [ ] 单元测试: JSON 失败 → Markdown fallback → 文本模式 (1h)
- 质量门控: 冷启动 <60s, 热 P95 <5s, Fallback 正常

Day 4 结构化解析 + 标准化 (10h)
- [ ] 尺寸解析器 Φ/R/M/±t (3h)
- [ ] 符号解析器 Ra / ⟂ / ∥ / GD&T 基础 (3h)
- [ ] 单位标准化 mm (2h)
- [ ] 解析置信度加权 + BBox IoU 校正 (1h)
- [ ] 单元测试: 尺寸归一化 + 符号映射 (1h)
- 质量门控: 关键字段召回 ≥70%, 延迟 <100ms/页

Day 5 路由策略 + API 接口 + 幂等 (9h)
- [ ] `OcrManager` auto/fallback (3h)
- [ ] `POST /api/v1/ocr/extract` (3h)
- [ ] Idempotency-Key 支持 (2h)
- [ ] 健康检查 provider 状态 (1h)
- 质量门控: cURL 成功 / provider 状态可见 / 幂等通过

Day 6 缓存 + Metrics + 置信度门控 + 测试 (9h)
- [ ] Redis 缓存键实现 (2h)
- [ ] Prometheus 基础指标 (requests/cache/latency) (3h)
- [ ] 置信度 fallback 阈值 (2h)
- [ ] 单元测试: 缓存键一致性 (1h)
- [ ] GPU/CPU 内存监控 hooks (1h)
- 质量门控: cache 命中 >40%, fallback 比率 <20%

Day 7 文档 + Demo + 冒烟测试 (8h)
- [ ] `docs/OCR_GUIDE.md` Quickstart (2h)
- [ ] API 使用示例 (1h)
- [ ] `examples/ocr_demo.ipynb` (2h)
- [ ] 冒烟测试 `tests/test_ocr_smoke.py` (2h)
- [ ] CI 集成基础工作流 (1h)
- 质量门控: Quickstart 完整运行 / CI 通过

里程碑 (Week1): Recall ≥70% (清晰) / Paddle P95 <2s / DeepSeek P95 <5s / 缓存命中 ≥40%

### Week 2 — 鲁棒性 & 评测体系

Day 8-9 智能裁剪/合并 + PDF 流式处理 (14h)
- [ ] 重叠裁剪算法 + 相邻文本合并 (7h)
- [ ] PDF 异步分页处理 (4h)
- [ ] OOM 保护 + max_crops 限制 (2h)
- [ ] 测试: 大文件/多页/裁剪数量 (1h)
- 质量门控: 无 OOM / 准确率不下降 (>Week1 recall -2%)

Day 10 质量控制器 + Schema 严格验证 + 错误分类 (8h)
- [ ] Schema 验证器 (3h)
- [ ] 关键字段检查 + 降级触发器 (3h)
- [ ] 质量报告生成 (1h)
- [ ] 错误分类 (集成 metrics 标签) (1h)
- 质量门控: 无效 JSON 自动降级 / 缺失字段列出 / 验证延迟 <50ms

Day 11 Golden Cases + CI 扩展 (10h)
- [ ] Golden 样本分类与元数据 (3h)
- [ ] 评测脚本 + 报告 (3h)
- [ ] CI 集成 (2h)
- [ ] 性能基准记录 (2h)
- 质量门控: 清晰 recall ≥80% / Edge-F1 ≥0.75 / CI 成功

Day 12 Analyze 接口集成 (8h)
- [ ] AnalysisOptions 扩展 enable_ocr / ocr_provider (2h)
- [ ] 向后兼容逻辑 (2h)
- [ ] OCR结果整合 + 延迟统计 (3h)
- [ ] 响应时间监控 (1h)
- 质量门控: 响应增量 <30% / 原有调用不破坏

Day 13 可观测性优化 (8h)
- [ ] Metrics 标签规范化 (2h)
- [ ] Grafana 面板 JSON (3h)
- [ ] 告警规则 (1h)
- [ ] 错误类型对齐 (2h)
- 质量门控: 仪表盘完整 / 错误类型精确

Day 14 加固与限流 (8h)
- [ ] 文件大小/页数限制 (2h)
- [ ] Rate limiting hooks (2h)
- [ ] 优雅关闭 (2h)
- [ ] 负载测试 (2h)
- 质量门控: QPS ≥5 稳定 / 正确 429/413 / 无 OOM

里程碑 (Week2): Recall ≥80% / Edge-F1 ≥0.75 / Fallback <20% / 缓存命中 ≥60% / Golden CI 通过

### Week 3 — 可选高级特性 (按需)

Day 15-16 vLLM Provider (14h 可选)
- [ ] vLLM 服务端部署 + 客户端 (8h)
- [ ] 批处理逻辑 (4h)
- [ ] 性能对比测试 (2h)
- 质量门控: 吞吐 ≥2x HF / 精度持平 (dimension_recall 差异 <1%)

Day 17 标题栏/表格解析 (10h 可选)
- [ ] 关键字映射 (4h)
- [ ] 位置关系推理 (3h)
- [ ] 表格结构识别 (3h)
- 质量门控: 标题栏字段准确率 ≥85%

Day 18-19 几何对齐 (16h 可选)
- [ ] R-tree 空间索引 (6h)
- [ ] 空间匹配算法 (6h)
- [ ] 容差匹配逻辑 (4h)
- 质量门控: 对齐成功率 ≥60% (±5% 容差)

Day 20-21 成本优化器 + 分层路由 (10h 可选)
- [ ] 成本计算器 (3h)
- [ ] 分层策略 (3h)
- [ ] A/B 测试钩子 (2h)
- [ ] 成本报告 (2h)
- 质量门控: 成本降低且 recall ≥95% 基线

里程碑 (Week3 可选): vLLM / 标题栏 / 几何对齐 / 成本优化 达成即进入 Week4 Buffer

### Week 4 — Buffer & 生产准备
- Bug 修复 / 性能调优 / 文档完善 / SLA 验证 / 技术债清理 / 灰度发布策略

---
## 📊 监控指标 (Prometheus)

Counters:
- `ocr_requests_total{provider,status}` status=success|degraded|failed
- `ocr_errors_total{type}` type=timeout|parse_error|oom|provider_down|invalid_input|schema_violation
- `ocr_cache_hits_total{provider}` / `ocr_cache_misses_total{provider}`

Gauges:
- `ocr_inflight_requests` 当前并发
- `ocr_gpu_memory_mb{device}` VRAM 使用
- `ocr_model_loaded{provider}` 0|1
- `ocr_field_recall{field_type}` dimension|symbol|title_block (评测后更新)

Histograms:
- `ocr_processing_duration_seconds{provider,stage}` stage=preprocess|inference|parse|normalize|route
- `ocr_prompt_length_chars{provider}`
- `ocr_confidence_score` buckets=[0.5,0.6,0.7,0.8,0.9,0.95,0.99]

Derived (Grafana):
- P50/P95/P99 延迟
- fallback_rate = rate(ocr_requests_total{status="degraded"}[5m]) / rate(ocr_requests_total[5m])

---
## 🧪 测试矩阵 (Unit & Integration)

Week1 单元测试:
- Paddle 基础: 文本 + bbox → OcrResult
- DeepSeek 降级: 人为破坏 JSON 验证 fallback 触发
- 解析器: Φ20±0.02 / M10×1.25 / R5 / Ra3.2 / ⟂A
- 缓存键一致性: 相同输入+provider+prompt_version → 相同 key

Week2 集成测试:
- 多页 PDF 处理
- 裁剪合并精度 (无尺寸丢失)
- 质量门控缺失字段报告
- Golden cases 批量评测脚本

Week3 可选性能测试:
- vLLM vs HF 吞吐
- 几何对齐成功率

---
## ⚠️ 风险管理 (更新)

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| DeepSeek 模型加载失败 | 中 | 高 | 环境验证 + CPU fallback | 
| JSON 结构频繁异常 | 高 | 中 | 三级降级 + schema 验证器 | 
| GPU OOM (大 PDF) | 中 | 高 | 裁剪流 / max_crops / 分辨率限制 | 
| 性能未达标 | 中 | 中 | Week3 vLLM 优化路径 | 
| 缓存膨胀 | 中 | 中 | TTL 调整 + 大文件不缓存 (>10MB) | 
| 错误分类不一致 | 低 | 中 | 统一 OcrError + metrics 标签 | 

---
## ✅ 验收标准总览

Week1 MVP:
- Recall ≥70% (清晰)
- DeepSeek 热 P95 <5s / Paddle P95 <2s
- 冷启动加载 <60s
- 缓存命中 ≥40%

Week2 Robustness:
- Recall ≥80% (清晰) / Edge-F1 ≥0.75
- Fallback <20%
- 缓存命中 ≥60%
- Brier <0.15

Week3 Optional:
- vLLM 吞吐 ≥2x HF
- 几何对齐成功率 ≥60%
- 标题栏准确率 ≥85%

Week4 Production:
- QPS ≥5 稳定无 OOM
- 完整监控 + 告警
- SLA 达标 (错误率 <1%, 95% 延迟可控)

---
## 🧩 依赖与环境

环境变量 (新增 prompt version / dataset version):
```bash
OCR_PROVIDER=auto
DEEPSEEK_ENABLED=true
CONFIDENCE_FALLBACK=0.85
OCR_TIMEOUT_MS=30000
OCR_MAX_CONCURRENT=10
PROMPT_VERSION=v1
DATASET_VERSION=v1.0
REDIS_URL=redis://localhost:6379/0
OCR_CACHE_TTL=3600
MAX_FILE_SIZE_MB=50
MAX_PDF_PAGES=20
RATE_LIMIT_QPS=10
```

环境验证脚本输出示例:
```
✅ PaddleOCR: OK
✅ CUDA: Available (optional)
✅ Redis: Connected
✅ Disk Space: 25.6 GB available
✅ Environment: Ready for OCR integration
```

---
## 👥 团队分工 (优化后)

| 角色 | 负责模块 | 工时 |
|------|----------|------|
| Developer 1 | Provider 实现 / 性能 / 缓存 / DeepSeek / vLLM(可选) | 58h |
| Developer 2 | API / 解析器 / 测试 / 文档 / 质量门控 | 40h |
| DevOps | 监控 / 部署 / CI / Grafana | 20h |
| QA | Golden cases / 评测脚本 / 报告 | 10h |

---
## 🔄 变更记录 (Changelog Snippet)

| 日期 | 版本 | 变更 |
|------|------|------|
| 2025-11-14 | v1.1 | 指标公式 / 安全规范 / 缓存键扩展 / 错误体系加入 |

后续新增需更新: `PROMPT_VERSION` 或 `DATASET_VERSION` → 触发缓存失效策略。

---
## 📝 备注

1. Week3 特性按吞吐需求与反馈动态决定。
2. GPU 非必需：保持 CPU first 路径完整。
3. 每日结束前运行基础评测脚本快速回归指标。
4. 错误码分配从 OCR_001 起：加载失败 OCR_001，超时 OCR_002，解析失败 OCR_010，JSON schema OCR_011。
5. 安全策略只记录图像 hash，不持久化原始图像内容于日志。

---
## ✅ 快速核对清单 (Daily Checklist)

- 模型加载锁 (asyncio.Lock) 生效
- 监控指标暴露 `/metrics`
- 安全限制 (大小/页数/MIME) 验证
- 缓存命中率走势 (Grafana)
- 回归脚本 `run_golden_evaluation.py` 绿灯
- 错误分类与计数 (无 unknown 类型)

---
（本文件为升级版 TODO，落实时请在 PR 描述引用对应章节号，保证追踪与验收一致。）

