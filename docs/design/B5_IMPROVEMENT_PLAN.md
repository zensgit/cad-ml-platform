# B5 提升计划：无文件名 80%+ 与生产就绪

**日期**: 2026-04-14  
**基线**: B4.5 — 无文件名 90.5%，有文件名 99-100%，24 类 Graph2D  
**目标**: B5 系列完成后，系统达到生产就绪状态

---

## 0. 当前基线总结

| 指标 | B4.5 达成值 | B5 目标 |
|------|------------|---------|
| 无文件名 acc | **90.5%** | **≥ 93%** |
| 有文件名 acc | **99-100%** | **≥ 99%（保持）** |
| 轴承座 recall | 60% | **≥ 85%** |
| 阀门 recall | 33%（仅3样本） | **≥ 70%** |
| 推理延迟 | 未基准测试 | **< 100ms/文件** |
| 覆盖率 | 4,574 样本 | 7,000+ 样本 |

---

## 1. B5.0 — 数据增强与弱类专项强化

**目标**: 轴承座 ≥ 85%, 阀门 ≥ 70%，整体 acc ≥ 92%  
**预期工期**: 2-3 天

### 1.1 轴承座专项增强

**问题根因**：
- 训练样本仅 74 个（所有类别中第 4 低）
- 形态多样（截面轴承座、法兰轴承座、分离式轴承座）与箱体/罐体存在视觉混淆
- 当前 recall=60%，precision=55%（存在双向混淆）

**增强方案**：

```bash
# 1. 收集更多轴承座 DXF（目标：150+ 样本）
# 2. 数据增强（几何变换）
python scripts/augment_dxf_graphs.py \
    --class 轴承座 \
    --methods "rotate,mirror,scale" \
    --factor 5 \           # 每个样本生成5个变体
    --output data/graph_cache_augmented/
```

**图级数据增强操作**：
| 操作 | 实现 | 效果 |
|------|------|------|
| 节点坐标旋转 | `x[:, :2] = R @ x[:, :2]` | 旋转不变性 |
| 镜像翻转 | `x[:, 0] = -x[:, 0]` | 对称结构泛化 |
| 缩放归一化扰动 | `x[:, :2] *= scale` | 尺寸无关性 |
| 边权重随机掩码 | 随机 drop 10% 边 | 鲁棒性 |

**实现文件**：`scripts/augment_dxf_graphs.py`（待创建）

### 1.2 阀门专项增强

**问题根因**：
- 验证集仅 3 个样本（不可靠评估）
- 训练集 29 个样本，结构上与传动件有重叠
- 优先动作：收集更多样本，而不是调参

**行动方案**：
1. 从原始数据集收集更多阀门样本（目标 60+）
2. 重新训练时采用 FocalLoss(gamma=1.5) 针对阀门类
3. 考虑阀门的子类别（球阀/蝶阀/闸阀）→ 统一归为阀门

### 1.3 数据集扩展

```python
# 目标：将缓存从 4,574 扩展到 7,000+ 样本
# 来源：
#   1. 现有未解析的 DXF（843 个解析失败的文件）
#   2. 新数据收集（工厂图纸扫描）
#   3. 合成数据增强（图级变换）

python3 scripts/preprocess_dxf_to_graphs.py \
    --dxf-dir data/raw_dxf/ \
    --cache-dir data/graph_cache_v2/ \
    --manifest data/graph_cache_v2/cache_manifest.csv \
    --skip-existing
```

---

## 2. B5.1 — vLLM 文字识别融合

**目标**: 无文件名场景 ≥ 93%（利用图纸内文字内容）  
**预期工期**: 3-5 天

### 2.1 架构设计

```
输入: DXF 字节流（无文件名）
  │
  ├─ Graph2DClassifier (g2d_w=0.50) → 形状特征 → P(class | graph)
  ├─ TitleBlockClassifier (tb_w=0.10) → 标题栏文字 → P(class | titleblock)
  └─ TextContentClassifier (txt_w=0.20) [新增 B5.1]
       │
       ├─ 从 DXF entities 提取所有 TEXT/MTEXT
       ├─ 关键词匹配（零件号、材料、技术要求）
       └─ vLLM/LLM 语义分类（可选）
```

**TextContentClassifier 关键字特征**：

| 文字来源 | 特征 | 示例 |
|---------|------|------|
| 标注文字 | 关键零件名 | "法兰盘", "轴承" |
| 技术要求 | 材料/工艺 | "45钢", "淬火" |
| 零件号 | 编号前缀规律 | "FL-", "ZC-" |
| 尺寸标注 | 特征尺寸 | Φ100, M12×1.5 |

### 2.2 实现步骤

**Step 1**: 提取 DXF 文字内容

```python
# src/ml/text_extractor.py
class DXFTextExtractor:
    def extract(self, dxf_path: str) -> dict:
        doc = ezdxf.readfile(dxf_path)
        texts = []
        for entity in doc.modelspace():
            if entity.dxftype() in ("TEXT", "MTEXT", "ATTDEF", "ATTRIB"):
                texts.append(entity.dxf.text)
        return {
            "all_text": " ".join(texts),
            "text_count": len(texts),
        }
```

**Step 2**: 关键词分类器

```python
# src/ml/text_classifier.py
class TextContentClassifier:
    KEYWORDS = {
        "法兰": ["法兰", "法兰盘", "flange", "连接盘"],
        "轴类": ["轴", "主轴", "转轴", "阶梯轴", "shaft"],
        "箱体": ["箱", "壳体", "机壳", "housing"],
        "轴承座": ["轴承座", "bearing housing", "轴承支座"],
        # ... 24 类完整列表
    }
    
    def predict_probs(self, text: str) -> dict[str, float]:
        scores = {cls: 0.0 for cls in self.KEYWORDS}
        for cls, kws in self.KEYWORDS.items():
            for kw in kws:
                if kw in text:
                    scores[cls] += 1.0 / len(kws)
        # Softmax 归一化
        return softmax(scores)
```

**Step 3**: 融合权重调整

```python
# 更新 hybrid_config.py
class TextContentConfig:
    enabled: bool = True
    fusion_weight: float = 0.20

class Graph2DConfig:
    fusion_weight: float = 0.45  # 略降（让文字分类参与）

class TitleBlockConfig:
    enabled: bool = True
    fusion_weight: float = 0.10
```

---

## 3. B5.2 — 推理性能优化

**目标**: 推理延迟 < 100ms/文件，内存 < 500MB  
**预期工期**: 1-2 天

### 3.1 延迟分解（预期）

| 步骤 | 预计耗时 | 优化方向 |
|------|---------|---------|
| DXF 文件读取 | ~30ms | 异步 I/O |
| DXF→Graph 转换 | ~80ms | 缓存化（已实现） |
| Graph2D 推理 | ~15ms | 模型量化（INT8） |
| 文件名分类 | ~1ms | — |
| 标题栏解析 | ~10ms | 并行化 |
| **总计（无缓存）** | **~136ms** | **目标 < 100ms** |

### 3.2 优化方案

**Graph 缓存命中**（已实现，B4.1）：
```python
# 已存在：CachedGraphDataset
# 命中率预期：70%+（重复文件）
# 优化后延迟：~50ms
```

**模型量化（新增）**：
```python
# scripts/quantize_graph2d_model.py
import torch
model = load_model("models/graph2d_finetuned_24class_v2.pth")
quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# 预期：延迟 -40%，内存 -60%，精度损失 < 1pp
```

**并行推理**：
```python
# src/ml/hybrid_classifier.py
async def predict_async(self, dxf_path: str) -> dict:
    results = await asyncio.gather(
        self.filename_clf.predict(dxf_path),
        self.graph2d_clf.predict(dxf_path),
        self.titleblock_clf.predict(dxf_path),
        self.text_clf.predict(dxf_path),
    )
    return self.fusion_engine.fuse(results)
```

---

## 4. B5.3 — 生产监控与反馈回路

**目标**: 自动检测精度漂移，触发重训练  
**预期工期**: 2-3 天

### 4.1 置信度监控

```python
# src/ml/monitoring.py
class PredictionMonitor:
    LOW_CONFIDENCE_THRESHOLD = 0.6
    
    def log_prediction(self, pred: dict):
        if pred["confidence"] < self.LOW_CONFIDENCE_THRESHOLD:
            self.low_conf_queue.append(pred)
        
        # 每日统计
        self.daily_stats["count"] += 1
        self.daily_stats["low_conf_rate"] = (
            len(self.low_conf_queue) / self.daily_stats["count"]
        )
```

### 4.2 漂移检测与自动重训练触发

```python
DRIFT_THRESHOLD = 0.05  # 低置信度比例超过 5% 触发告警

if monitor.daily_stats["low_conf_rate"] > DRIFT_THRESHOLD:
    notify_team("Graph2D 置信度下降，建议检查数据分布")
    # 可选：自动触发增量重训练
```

---

## 5. B5 整体验收标准

| 阶段 | 验收指标 | 判定方式 |
|------|---------|---------|
| **B5.0** | 轴承座 ≥ 85%，阀门 ≥ 70%，整体 ≥ 92% | `evaluate_graph2d_v2.py` |
| **B5.1** | 无文件名 ≥ 93%（TextContent 融合后） | 场景测试集 200 样本 |
| **B5.2** | 推理 < 100ms（无缓存），< 50ms（缓存命中） | `benchmark_inference.py` |
| **B5.3** | 监控上线，低置信告警生效 | 生产日志 |
| **最终** | 有名 ≥ 99%，无名 ≥ 93%，延迟 < 100ms | 端到端集成测试 |

---

## 6. 时间线

```
B5.0 数据增强     [2026-04-15 ~ 04-17]  约 3 天
B5.1 vLLM 文字   [2026-04-18 ~ 04-22]  约 5 天
B5.2 性能优化    [2026-04-23 ~ 04-24]  约 2 天
B5.3 监控上线    [2026-04-25 ~ 04-27]  约 3 天
────────────────────────────────────────────────
总计                                    约 13 天
```

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解方案 |
|------|------|------|---------|
| 轴承座数据收集困难 | 中 | 高 | 图级数据增强补充不足 |
| vLLM 文字提取噪声大 | 高 | 中 | 使用置信度阈值过滤，仅高置信文字参与融合 |
| 量化后精度下降 > 2pp | 低 | 中 | 回退 FP32，探索 INT16 量化 |
| DXF 解析库 ezdxf 文字提取不完整 | 中 | 低 | 多字段尝试：TEXT + MTEXT + ATTDEF |
| 数据增强导致过拟合 | 低 | 中 | 使用独立 held-out 测试集验证 |

---

## 8. 关键文件规划

| 文件 | 状态 | 描述 |
|------|------|------|
| `scripts/augment_dxf_graphs.py` | 待创建 | 图级数据增强 |
| `src/ml/text_extractor.py` | 待创建 | DXF 文字提取 |
| `src/ml/text_classifier.py` | 待创建 | 关键词文字分类 |
| `scripts/quantize_graph2d_model.py` | 待创建 | 模型量化 |
| `scripts/benchmark_inference.py` | 待创建 | 延迟基准测试 |
| `src/ml/monitoring.py` | 待创建 | 预测监控 |
| `src/ml/hybrid_config.py` | 已更新 | fn=0.45, g2d=0.50 |

---

## 9. 里程碑总结（B4 系列回顾）

| 版本 | 关键突破 | 无名 acc | 有名 acc |
|------|---------|---------|---------|
| B4.1 | 对比预训练 + 图缓存（247× 加速） | ~27% | ~60% |
| B4.2 | Focal Loss + 5 类模型 | ~25% | ~70% |
| B4.3 | 24 类细粒度标签 | ~30% | ~82% |
| B4.4 | GraphEncoderV2（Attention Pool） | ~65% | ~90% |
| **B4.5** | **Graph2D 激活 + 权重调优** | **90.5%** | **99-100%** |
| B5.0 目标 | 数据增强 | **92%** | **99%** |
| B5.1 目标 | 文字融合 | **93%** | **99%** |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
