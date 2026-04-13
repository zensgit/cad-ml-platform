# 2026 Q2 详细开发计划

**起始日期**: 2026-04-14  
**预计周期**: 10-12 周  
**基线状态**: main@fedf50c8，代码与 GitHub 完全同步  

---

## 现状诊断

在制定计划前，必须正视当前系统的真实状态：

| 维度 | 数据 | 评估 |
|------|------|------|
| 分类准确率 | 99.67% | 高度依赖文件名（权重 0.7），Graph2D 独立仅 ~27% |
| 训练数据量 | ~1,023 DXF (training_v8) | 60+ 类别，严重不足 |
| 标签质量 | 文件名弱标签提取 | 无大规模人工标注 |
| 代码规模 | 604K LOC / 1,754 .py | 存在巨型文件和原型代码 |
| LLM 集成 | 5 Provider（Claude/OpenAI/Qwen/Ollama/Offline） | vLLM 仅有 benchmark 脚本，未集成 |
| 部署 | Docker + K8s Helm | 无 GPU 服务定义 |
| CI/CD | 38 个 workflow | vLLM benchmark 用 `--dry-run`，未真正测试 |
| Feature Flags | `hybrid_classifier_enabled: 0%` | 混合分类器在生产未启用 |

---

## 阶段 A：工程治理与代码瘦身（Week 1-3）

> 目标：降低维护成本，为后续开发扫清障碍

### A1. 拆分巨型文件（Week 1）

**最高优先级**：`src/core/materials/classifier.py`（15,763 LOC）

| 新文件 | 内容 | 预估行数 |
|--------|------|----------|
| `materials/data_models.py` | MaterialCategory, MaterialSubCategory, MaterialGroup 枚举 + MaterialProperties, ProcessRecommendation, MaterialInfo 数据类 | ~250 |
| `materials/classify.py` | `classify_material_detailed()`, `classify_material_simple()`, `search_materials()` | ~500 |
| `materials/properties.py` | `get_material_info()`, `search_by_properties()` | ~800 |
| `materials/processing.py` | `get_process_recommendations()`, `get_material_recommendations()` | ~900 |
| `materials/equivalence.py` | `get_material_equivalence()`, `find_equivalent_material()`, `list_material_standards()` | ~600 |
| `materials/cost.py` | `get_material_cost()`, `compare_material_costs()`, `search_by_cost()`, `get_cost_tier_info()` | ~700 |
| `materials/compatibility.py` | `check_weld_compatibility()`, `check_galvanic_corrosion()`, `check_heat_treatment_compatibility()` | ~600 |
| `materials/export.py` | `export_materials_csv()`, `export_equivalence_csv()` | ~300 |
| `materials/__init__.py` | 统一导出，保持外部 API 不变 | ~50 |

**操作步骤**：
1. 先写测试保护现有 API：确认 `classify_material_detailed()` 等函数的输入输出
2. 逐模块提取，每提取一个跑一次全量测试
3. 旧 `classifier.py` 最终只保留 re-export（过渡期），后续删除

**次优先级**：`src/api/v1/analyze.py`（3,922 LOC）
- 按 API 路由组拆分为独立 Router：`analyze_routes.py`, `classify_routes.py`, `similarity_routes.py`
- 使用 FastAPI 的 `APIRouter` + `include_router` 模式

### A2. 清理原型代码（Week 2）

**审计 `src/core/vision/` 的 106 个文件（共 80K LOC）**：

按以下分类处理：

| 类别 | 处理方式 | 示例模块 |
|------|----------|----------|
| 生产用 | 保留，补测试 | `providers/openai.py`, `ocr/manager.py` |
| 实验性但有价值 | 移入 `src/experimental/` | `chaos_engineering.py`, `hot_reload.py` |
| 框架桩/未使用 | 归档到 `archive/` 或删除 | 未被任何 import 引用的模块 |

**具体操作**：
```bash
# 第 1 步：找出未被引用的 vision 模块
# 对每个 vision/*.py，grep 其被 import 的次数
# import 数 = 0 且不在 __init__.py 中 → 候选删除
```

**预期收益**：减少 20-30K LOC 维护面积

### A3. 补齐测试缺口（Week 3）

**当前状态**：725 test 文件 / 189K LOC，test-to-code ratio 0.63

**优先补充领域**：

| 模块 | 当前覆盖 | 目标 | 具体任务 |
|------|----------|------|----------|
| `materials/classifier.py` | 低 | 90%+ | 拆分后对每个子模块写 pytest |
| `hybrid_classifier.py` | 中 | 85%+ | 补充边界 case：所有分支器都低置信、部分分支器不可用 |
| `fusion.py` | 中 | 90%+ | 测试 4 种融合策略 + auto-selection 逻辑 |
| `calibration.py` | 低 | 80%+ | 测试 5 种标定方法 + 降级回退路径 |
| `llm_providers.py` | 低 | 70%+ | Mock 测试每个 Provider 的错误处理 |

**交付物**：
- [ ] `materials/` 拆分完成，所有 re-export 正常
- [ ] `analyze.py` 拆分为 3+ 个 Router 文件
- [ ] 清理 15+ 个未使用的 vision 模块
- [ ] 测试覆盖率从 ~63% → 75%+
- [ ] CI 全绿

---

## 阶段 B：ML 核心能力提升（Week 4-7）

> 目标：让分类系统在无文件名线索时仍然可用

### B1. 训练数据扩充与标注（Week 4-5）

**问题**：当前 ~1,023 个 DXF 样本覆盖 60+ 类别，平均每类仅 17 个样本

**数据策略**：

```
目标数据量：≥ 3,000 个已标注 DXF
目标类别数：聚焦 20-30 个高频类别（覆盖 80%+ 真实场景）
```

#### Week 4：数据收集与标注基础设施

| 任务 | 具体内容 | 文件 |
|------|----------|------|
| 4.1 标签体系精简 | 将 94 类 → 20-30 核心类（合并低频类为"其他"） | `data/knowledge/label_synonyms_template.json` |
| 4.2 弱标签置信度提升 | 改进文件名提取正则，支持更多命名模式 | `src/ml/filename_classifier.py` |
| 4.3 标注工具 | 编写简易标注脚本：展示 DXF 预览 + 文件名推测 → 人工确认/修正 | `scripts/label_annotation_tool.py`（新建） |
| 4.4 数据增强管线 | 对现有 DXF 做几何变换增强（旋转、缩放、镜像） | `src/ml/augmentation/` |

#### Week 5：标注执行与数据验证

| 任务 | 具体内容 |
|------|----------|
| 5.1 人工标注 | 使用标注工具处理 500+ 新样本 |
| 5.2 交叉验证标签 | 文件名标签 vs 人工标签一致性检查 |
| 5.3 类别分布平衡 | 确保每个核心类别 ≥ 50 个样本 |
| 5.4 构建标准评测集 | 200 个精标样本作为 golden test set，不参与训练 |

### B2. Graph2D 模型强化（Week 5-6）

**当前瓶颈分析**：

```
架构：EdgeGraphSAGE (2 层, hidden=128)
节点特征：19 维 (几何+文本提示)
边特征：7 维
训练：focal loss + balanced sampler + 知识蒸馏
问题：
  1. 数据不足 → 严重过拟合
  2. 类别太多 → 决策边界模糊
  3. 图结构稀疏 → GNN 信息传播受限
  4. 节点采样上限 200 → 大图信息丢失
```

**改进计划**：

| 任务 | 修改文件 | 具体变更 |
|------|----------|----------|
| 6.1 增加 GNN 深度 | `src/ml/vision_2d.py` | 2 层 → 3-4 层，加 residual connection |
| 6.2 扩大节点上限 | `src/ml/vision_2d.py` | `DXF_MAX_NODES: 200 → 500` |
| 6.3 增加节点特征 | `src/ml/vision_2d.py` | 19 维 → 25+ 维：加 `area_norm`, `perimeter_norm`, `aspect_ratio`, `curvature_mean`, `is_hatching`, `is_centerline` |
| 6.4 图增强 | `src/ml/augmentation/` | 节点 dropout (10%)、边 perturbation、子图采样 |
| 6.5 对比学习预训练 | `scripts/pretrain_graph2d_contrastive.py`（新建） | 在无标签数据上用 GraphCL/SimCLR 做预训练 |
| 6.6 分阶段训练 | `scripts/run_graph2d_pipeline_local.py` | 先训练 5 类粗分类 → 再训练细分类（hierarchical classification） |

**训练配置调整**：
```yaml
# 更新 config/graph2d_profile_strict_node23_edgesage_v1.yaml
model_type: edge_sage
node_dim: 25          # 19 → 25
hidden_dim: 256       # 128 → 256
num_layers: 3         # 2 → 3
epochs: 30            # 10 → 30
lr: 0.0005            # 0.001 → 0.0005
weight_decay: 0.0001  # 新增
max_nodes: 500        # 200 → 500
dropout: 0.3          # 0.2 → 0.3
```

**目标**：Graph2D 独立准确率从 ~27% → **55%+**（在 20-30 类体系下）

### B3. TitleBlock 分类器激活（Week 6-7）

**当前状态**：`titleblock` 分支默认禁用，权重 0.2，min_confidence 0.75

| 任务 | 具体内容 | 文件 |
|------|----------|------|
| 7.1 OCR 提取优化 | 确保 PaddleOCR 对标题栏区域的识别率 > 90% | `src/core/ocr/providers/paddle.py` |
| 7.2 标题栏定位 | 基于 DXF 图框（border_hint）自动裁剪标题栏区域 | `src/ml/titleblock_extractor.py` |
| 7.3 文本分类器 | 标题栏文本 → 零件类别映射（TF-IDF + 关键词匹配） | `src/ml/titleblock_classifier.py` |
| 7.4 集成测试 | 在 golden test set 上验证 titleblock 分支的独立准确率 | `tests/integration/test_titleblock_pipeline.py` |

### B4. 混合分类器权重调优（Week 7）

**当前权重 vs 目标权重**：

```
Branch          当前权重    目标权重    变更原因
─────────────────────────────────────────────────
filename         0.70       0.50      降低，避免过度依赖文件名
graph2d          0.30       0.30      保持，随准确率提升自然增值
titleblock       0.20       0.25      提升，激活后作为第三信号源
process          0.15       0.15      保持
history          0.20       0.10      降低，功能不完善
─────────────────────────────────────────────────
rejection gate   0.35       0.40      提高门槛，宁可拒绝不可误判
```

**验证方法**：
1. 在 golden test set 上用 grid search 找最优权重组合
2. 对比有/无文件名时的准确率（模拟真实 worst case）
3. 更新 `config/hybrid_classifier.yaml`
4. 更新 `config/hybrid_superpass_targets.yaml` 中的 min_hybrid_accuracy

**交付物**：
- [ ] 训练数据量 1,023 → 3,000+
- [ ] 标签体系从 94 类精简为 20-30 核心类
- [ ] Golden test set 200 个精标样本
- [ ] Graph2D 独立准确率 27% → 55%+
- [ ] TitleBlock 分支启用并通过集成测试
- [ ] 混合分类器在无文件名时准确率 > 60%
- [ ] Superpass gate 全绿

---

## 阶段 C：vLLM 本地推理落地（Week 8-10）

> 目标：LLM 推理延迟从 ~800ms → <100ms，消除外部 API 依赖

### C1. 基础设施准备（Week 8）

| 任务 | 具体内容 | 文件 |
|------|----------|------|
| 8.1 依赖管理 | 新增 `requirements-vllm.txt`：vllm>=0.4.0, torch>=2.0 | `requirements-vllm.txt`（新建） |
| 8.2 VLLMProvider 实现 | 继承 `BaseLLMProvider`，对接 vLLM 的 OpenAI 兼容 API | `src/core/assistant/llm_providers.py` |
| 8.3 Docker 服务定义 | 新增 vllm service，GPU 资源分配 | `docker-compose.yml` |
| 8.4 Feature Flag | 新增 `vllm_enabled` 开关 | `config/feature_flags.json` |
| 8.5 Provider 注册 | 在 ProviderRegistry 注册 "llm/vllm" | `src/core/providers/bootstrap.py` |

**VLLMProvider 设计**：
```python
# src/core/assistant/llm_providers.py 新增

class VLLMProvider(BaseLLMProvider):
    """通过 OpenAI 兼容 API 对接本地 vLLM 服务"""
    
    def __init__(self):
        self.endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8001/v1")
        self.model_name = os.getenv("VLLM_MODEL", "deepseek-coder-6.7b-instruct")
    
    async def process(self, messages, **kwargs):
        # POST /v1/chat/completions
        # 支持 streaming
        # Token 计数
        # 超时 & 重试
        ...
```

**docker-compose.yml 新增**：
```yaml
vllm:
  image: vllm/vllm-openai:latest
  runtime: nvidia
  environment:
    - MODEL_NAME=deepseek-ai/deepseek-coder-6.7b-instruct
    - QUANTIZATION=awq
    - MAX_MODEL_LEN=4096
    - GPU_MEMORY_UTILIZATION=0.85
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  ports:
    - "8001:8000"
  volumes:
    - vllm-cache:/root/.cache/huggingface
```

### C2. 模型选型与量化（Week 8-9）

**候选模型对比**：

| 模型 | 参数量 | VRAM (AWQ) | 延迟 (预估) | CAD 适用性 |
|------|--------|------------|-------------|-----------|
| DeepSeek-Coder-6.7B | 6.7B | ~4GB | ~30ms | 高（代码/结构理解） |
| Qwen2-7B-Instruct | 7B | ~4.5GB | ~35ms | 高（中文理解） |
| Llama-3-8B-Instruct | 8B | ~5GB | ~40ms | 中（通用能力） |
| DeepSeek-V2-Lite | 16B MoE | ~6GB | ~50ms | 高（MoE 效率） |

**推荐**：DeepSeek-Coder-6.7B (AWQ)
- 理由：参数效率高、中英文好、代码/结构理解强、VRAM 需求可控

**量化验证**：
```bash
# 使用现有 benchmark 脚本
python scripts/benchmark_vllm_quantization.py \
  --model deepseek-ai/deepseek-coder-6.7b-instruct \
  --quantization awq \
  --concurrency 1,10,50 \
  --output results/vllm_benchmark.json
```

**验收标准**：
- P95 延迟 < 100ms（单请求）
- 吞吐量 > 20 req/s（10 并发）
- CAD 领域 prompt 质量不低于 API 调用

### C3. 集成与切换（Week 9-10）

| 任务 | 具体内容 | 文件 |
|------|----------|------|
| 9.1 Provider 选择逻辑 | vLLM 可用时优先本地，降级到云端 API | `src/core/assistant/assistant.py` |
| 9.2 OCR 管线切换 | DeepSeek stub → 真实 VLLMProvider | `src/core/vision/providers/deepseek_stub.py` → 实际实现 |
| 9.3 Prompt 适配 | 为本地模型调整 system prompt 和 few-shot 示例 | `src/core/assistant/prompts/` |
| 9.4 监控指标 | vLLM 延迟、吞吐、GPU 利用率接入 Prometheus | `config/prometheus.yml` |
| 9.5 CI 集成 | 移除 benchmark `--dry-run`，加真实 vLLM 测试 | `.github/workflows/enterprise_release.yml` |

### C4. 灰度发布（Week 10）

```
Week 10 Day 1-2: 内部测试环境全量切换到 vLLM
Week 10 Day 3:   对比 API vs vLLM 在相同 query 上的结果质量
Week 10 Day 4:   Feature flag 开启 10% 流量到 vLLM
Week 10 Day 5:   监控指标正常 → 50% → 100%
```

**交付物**：
- [ ] VLLMProvider 实现并注册
- [ ] Docker Compose 含 vLLM 服务（GPU）
- [ ] 量化模型 benchmark 报告
- [ ] P95 延迟 < 100ms 验证通过
- [ ] DeepSeek stub 替换为真实实现
- [ ] CI/CD 含 vLLM 集成测试
- [ ] Feature flag 灰度完成

---

## 阶段 D：稳固与展望（Week 11-12）

### D1. 端到端回归测试（Week 11）

| 测试类型 | 范围 | 工具 |
|----------|------|------|
| API 回归 | 全部 v1 端点 | pytest + httpx |
| 分类回归 | Golden test set 200 样本 | superpass workflow |
| 性能回归 | P95 延迟、吞吐量 | locust / k6 |
| LLM 质量 | 50 个 CAD 领域问答 | AI eval script |
| 内存/GPU | 长时间运行稳定性 | Prometheus + Grafana |

### D2. 文档更新（Week 11）

| 文档 | 内容 |
|------|------|
| API 文档 | 更新 OpenAPI spec，新增 vLLM 相关端点说明 |
| 运维手册 | vLLM 服务运维：模型更新、GPU 监控、降级方案 |
| 架构图 | 更新系统架构图，包含 vLLM 服务 |
| CHANGELOG | 总结 Q2 所有变更 |

### D3. Q3 规划（Week 12）

基于 Q2 成果评估下一步方向：

| 方向 | 前置条件 | 优先级 |
|------|----------|--------|
| PointNet++ 3D 分类 | 需要 3D 点云训练数据 | 若有数据→高 |
| 多模型 Ensemble | Graph2D 准确率 >55% | 中 |
| 联邦学习 | 多租户客户需求明确 | 低 |
| Streaming API | 用户量增长需要 | 中 |
| Active Learning Loop | 生产数据积累足够 | 高 |

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| GPU 硬件不足 | 中 | 阶段 C 受阻 | 备选：Ollama + CPU 量化方案 |
| 标注数据不足 | 高 | 阶段 B 效果打折 | 优先精简类别数，用数据增强补充 |
| vLLM 版本兼容性 | 低 | 集成延迟 | 锁定 vLLM 版本，容器化隔离 |
| materials 拆分引入 bug | 中 | 阶段 A 延期 | 先写测试再拆分，逐模块验证 |
| Graph2D 改进不达预期 | 中 | 系统仍依赖文件名 | 同时推进 TitleBlock，多信号源互补 |

---

## 里程碑总览

```
Week  1-3  ┃ 阶段 A ┃ 工程治理    ┃ materials 拆分 → 原型清理 → 补测试
Week  4-5  ┃ 阶段 B1┃ 数据建设    ┃ 标签精简 → 标注工具 → 扩充数据
Week  5-6  ┃ 阶段 B2┃ 模型强化    ┃ Graph2D 架构升级 → 重新训练
Week  6-7  ┃ 阶段 B3┃ 分支激活    ┃ TitleBlock 启用 → 权重调优
Week  8-10 ┃ 阶段 C ┃ vLLM 落地   ┃ Provider → 量化 → 集成 → 灰度
Week 11-12 ┃ 阶段 D ┃ 稳固 & 规划 ┃ 回归测试 → 文档 → Q3 方向
```

---

*文档版本*: 1.0  
*创建日期*: 2026-04-13  
*维护者*: Platform Team
