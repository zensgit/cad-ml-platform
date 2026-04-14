# B6.0 提升计划：下一代模型架构 + 多模态融合 + 95%+ 突破

**日期**: 2026-04-14  
**基线**: B5.9 — avg=94.8%，v4=91.9%，P50=53ms，430KB  
**目标**: 突破 95% avg（需更深模型或异构特征）

---

## 1. B5 总结与瓶颈

### 1.1 B5 系列达成

| 指标 | 起点 (B4.5) | B5 最终 | 提升 |
|------|-----------|---------|------|
| Graph2D acc | 90.5% | **91.9%** | +1.4pp |
| 综合 avg | — | **94.8%** | — |
| 文字精度 | 0% | **70.7%** | +70.7pp |
| 推理延迟 | ~133ms | **34-53ms** | -55~75% |
| 模型体积 | 1623KB | **430KB** | -74% |

### 1.2 为什么 94.8% 是当前上限

1. **法兰文字天然极限**：DXF 中不含"法兰"词汇（命中率 5%），关键词方法无法突破
2. **三主类几何相似**：法兰/轴类/箱体均为圆形/圆柱形件，3 层 GNN 难以区分微妙拓扑差异
3. **集成无多样性**：v2/v3/v4 同架构同数据分布，集成无法超越 v4 单模型
4. **数据量限制**：4574 原始样本，部分类仅 1-5 样本

---

## 2. B6.0a：更深 GNN 架构

### 2.1 GraphEncoderV3（提案）

| 改进 | 说明 | 预期收益 |
|------|------|---------|
| 层数 3→5 | 更大感受野，捕捉远距离拓扑关系 | +0.5-1pp |
| 注意力头数 1→4 | Multi-head attention pooling | +0.3pp |
| 跳跃连接（JK-Net） | 融合不同层的表示 | +0.3pp |
| DropEdge 正则 | 训练时随机丢边（防过拟合） | 稳定性 |

```python
class GraphEncoderV3WithHead(torch.nn.Module):
    def __init__(self, node_dim=19, edge_dim=7, hidden_dim=256, num_layers=5, heads=4):
        self.layers = ModuleList([
            SAGEConv(in_ch, hidden_dim, ...) for i in range(num_layers)
        ])
        self.jk = JumpingKnowledge('cat')  # 融合所有层
        self.pool = MultiHeadAttentionPooling(hidden_dim * num_layers, heads=heads)
        self.classifier = Linear(hidden_dim * num_layers, num_classes)
```

### 2.2 训练要求

- 数据量：需 ≥ 10,000 样本（当前 4574+增强 6760）
- GPU 加速：5 层 GNN 在 CPU 上训练过慢（建议 CUDA）
- 训练时间：预计 2-4 小时（vs v4 的 ~30 分钟）

---

## 3. B6.0b：文字 Embedding 融合

### 3.1 方案

替代关键词匹配，使用轻量文字 embedding：

```
DXF 文字 → TF-IDF + SVD(50d) → MLP(50→24) → text_probs
```

**优势**：
- 不依赖人工关键词词典
- 能捕捉词频模式（法兰图纸中"公称压力"出现频率显著高于其他类）
- 训练数据：用现有 manifest 中的文字内容自动构建

### 3.2 实施步骤

```bash
# 步骤 1：构建文字语料库
python scripts/build_text_corpus.py \
    --manifest data/manifests/unified_manifest_v2.csv \
    --output data/text_corpus.csv

# 步骤 2：训练 TF-IDF + MLP 文字分类器
python scripts/train_text_classifier_ml.py \
    --corpus data/text_corpus.csv \
    --output models/text_classifier_tfidf.pkl

# 步骤 3：集成到 TextContentClassifier（fallback 模式）
# 关键词匹配优先，无命中时 fallback 到 TF-IDF
```

---

## 4. B6.0c：异构集成

### 4.1 方案

训练不同架构的模型，形成真正有多样性的集成：

| 成员 | 架构 | 特征 | 预期 |
|------|------|------|------|
| GNN (v4) | GraphEncoderV2 | 图拓扑+节点特征 | 91.9% |
| MLP | 3-layer MLP | 统计特征（节点数/边数/度分布） | ~85% |
| Text MLP | TF-IDF + MLP | 文字特征 | ~60%（覆盖有限） |
| **集成** | **软投票** | **三种互补特征** | **预期 93%+** |

### 4.2 统计特征 MLP（无 GNN 依赖）

```python
# 简单统计特征（无需 GNN）
features = [
    num_nodes, num_edges, 
    mean_degree, max_degree, std_degree,
    num_circles, num_lines, num_arcs,
    bbox_width, bbox_height, aspect_ratio,
    # ... 约 30 维
]
# 训练 MLP(30→64→64→24)
```

这种 MLP 虽然精度低（~85%），但**错误模式与 GNN 不同**，可在集成中提供互补信息。

---

## 5. B6.0d：数据扩充策略

### 5.1 外部数据源

| 来源 | 预计增量 | 难度 |
|------|---------|------|
| 生产低置信度队列（B5.3 已建立） | 100-500/月 | 低（自动积累） |
| 公开 DXF 数据集 | 数百-数千 | 中（需标注） |
| CAD 软件生成合成数据 | 不限 | 高（需 CAD API） |

### 5.2 半监督学习

利用大量未标注 DXF 文件：

```
有标注数据（4574）→ 训练初始模型
未标注数据 → 模型预测 → confidence > 0.9 的样本加入训练集
重新训练 → 迭代
```

---

## 6. 实施优先级

```
Phase 1 (1-2 周): B6.0d 数据扩充
  → 继续通过 low_conf_queue 积累审核样本
  → 达到 200+ 条后触发 v5 增量训练
  → 目标：v5 acc ≥ 92.5%

Phase 2 (2-4 周): B6.0b 文字 Embedding
  → 构建 TF-IDF 语料库
  → 训练文字 MLP
  → 集成到 TextContentClassifier fallback
  → 目标：文字精度 80%+（vs 70.7%）

Phase 3 (4-6 周): B6.0a 更深 GNN
  → 实现 GraphEncoderV3（JK-Net + Multi-head）
  → 需要 GPU 训练环境
  → 目标：Graph2D acc ≥ 93%

Phase 4 (6-8 周): B6.0c 异构集成
  → 统计特征 MLP + 文字 MLP + GNN
  → 目标：集成 acc ≥ 94%，综合 avg ≥ 96%
```

---

## 7. 验收标准

| 阶段 | 指标 | 目标 |
|------|------|------|
| B6.0d 数据扩充 | v5 acc | ≥ 92.5% |
| B6.0b 文字 Embedding | 文字精度 | ≥ 80% |
| B6.0a 更深 GNN | Graph2D acc | ≥ 93% |
| B6.0c 异构集成 | 综合 avg | **≥ 96%** |

---

## 8. B5→B6 交接清单

### 生产环境已就绪

```bash
# 当前推荐配置
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v4_int8.pth
export FILENAME_FUSION_WEIGHT=0.50
export GRAPH2D_FUSION_WEIGHT=0.40
export TEXT_CONTENT_FUSION_WEIGHT=0.10
export TEXT_CONTENT_ENABLED=true
export MONITOR_WINDOW_SIZE=1000
export LOW_CONF_QUEUE_THRESHOLD=0.50
```

### 监控系统活跃

- `PredictionMonitor`：自动漂移检测（low_conf > 10% → WARN）
- `LowConfidenceQueue`：自动积累审核样本
- `text_hit_rate` 实时追踪文字信号有效性

### 持续学习闭环就绪

```
生产推理 → monitor + queue → 人工审核 → append_manifest → 增量训练 → 部署
```

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
