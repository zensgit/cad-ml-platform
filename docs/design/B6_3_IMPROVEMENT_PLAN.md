# B6.3 提升计划：97%+ 路线 + GraphEncoderV3 + 半监督学习

**日期**: 2026-04-14  
**基线**: B6.2 — 异构集成 95.8%，蒸馏 v5 训练中  
**长期目标**: 单模型 ≥ 95%，集成 ≥ 97%

---

## 1. 当前状态与长期路线图

### 1.1 已达成里程碑

```
B4.5  90.5% (GNN 基线)
  ↓ +0.5pp 数据增强
B5.0  91.0% (v3)
  ↓ +3.8pp avg (三路融合)
B5.8  94.8% avg (v4 + 关键词)
  ↓ +1.0pp (异构集成)
B6.1  95.8% (GNN + StatMLP + TextMLP)
```

### 1.2 路线图

```
当前: 95.8% (异构集成) / 91.9% (GNN 单模型)
  ↓ 蒸馏 (B6.2)
近期: ~93% (蒸馏 v5) → 量化 → 单模型部署
  ↓ GraphEncoderV3 (B6.3)
中期: ~94% (5层 GNN) + 集成 → ~97%
  ↓ 半监督 + 更多数据 (B6.4)
长期: ~96% (单模型) → ~98% (集成)
```

---

## 2. B6.3a：GraphEncoderV3（5 层 + JK-Net）

### 2.1 架构

| 参数 | V2 (当前) | V3 (提案) |
|------|----------|---------|
| 层数 | 3 | **5** |
| 隐藏维度 | 256 | 256 |
| Pooling | 单头 Attention | **4头 Multi-head Attention** |
| 跳跃连接 | 无 | **JumpingKnowledge (cat)** |
| 正则化 | Dropout | **Dropout + DropEdge** |
| 预期精度 | 91.9% | **93-94%** |

### 2.2 实施要点

```python
class GraphEncoderV3WithHead(nn.Module):
    def __init__(self, node_dim=19, edge_dim=7, hidden_dim=256,
                 num_layers=5, jk_mode='cat', num_classes=24):
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # JumpingKnowledge: 融合所有层的节点表示
        jk_dim = hidden_dim * num_layers
        self.jk = JumpingKnowledge(jk_mode, hidden_dim, num_layers)
        
        # Multi-head attention pooling
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(nn.Linear(jk_dim, 128), nn.Tanh(), nn.Linear(128, 1)),
            nn=nn.Linear(jk_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        layer_outputs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            layer_outputs.append(x)
        
        x = self.jk(layer_outputs)  # [N, hidden_dim * num_layers]
        x = self.pool(x, batch)     # [B, hidden_dim]
        return self.classifier(x)
```

### 2.3 训练需求

- **GPU 必需**：5 层 × 6760 样本 × 50 epochs ≈ 2-4 小时（GPU）
- CPU 不可行：预计 20+ 小时
- 建议环境：CUDA 11.8+，RTX 3060 以上

---

## 3. B6.3b：半监督学习

### 3.1 自训练（Self-Training）

```
有标签数据 (4574) → 训练初始模型
未标签数据 (收集中) → 模型预测 → confidence > 0.95 的样本加入训练
重新训练 → 迭代 3 轮
```

### 3.2 数据来源

- 生产推理积累的 DXF 文件（通过 monitor 记录）
- confidence > 0.95 的预测作为 pseudo-label
- 低置信度样本通过审核队列获得人工标签

---

## 4. B6.3c：StatMLP 特征扩展

当前 59 维 → 扩展到 ~120 维：

| 新增特征类 | 维度 | 说明 |
|-----------|------|------|
| 连通分量 | 3 | 分量数、最大分量比例、孤立分量数 |
| 聚类系数 | 3 | 均值、最大值、标准差 |
| 图谱特征 | 10 | 邻接矩阵前10特征值 |
| 节点类型分布 | 19 | 每种节点特征 > 0 的比例 |
| 边属性分位数 | 21 | 每维 Q25/Q50/Q75 |

预期：StatMLP 94.4% → **95%+**（更丰富特征 + 与 GNN 更大互补性）

---

## 5. 全项目交付总清单

### 代码文件（18 新建 + 5 修改）

| # | 文件 | 功能 | 阶段 |
|---|------|------|------|
| 1 | `src/ml/text_extractor.py` | DXF 文字提取 | B5.1 |
| 2 | `src/ml/text_classifier.py` | 关键词+共现+margin 分类 | B5.1-B5.7 |
| 3 | `src/ml/monitoring/prediction_monitor.py` | 漂移监控 | B5.3 |
| 4 | `src/ml/low_conf_queue.py` | 主动学习队列 | B5.3 |
| 5 | `scripts/augment_dxf_graphs.py` | 图增强 | B5.0 |
| 6 | `scripts/finetune_graph2d_v2_augmented.py` | FocalLoss 训练 | B5.0 |
| 7 | `scripts/quantize_graph2d_model.py` | INT8 量化 | B5.2 |
| 8 | `scripts/benchmark_inference.py` | 推理基准 | B5.5 |
| 9 | `scripts/audit_text_coverage.py` | 关键词审计 | B5.1 |
| 10 | `scripts/search_hybrid_weights_v2.py` | 权重搜索 | B5.1 |
| 11 | `scripts/append_reviewed_to_manifest.py` | 审核追加 | B5.4 |
| 12 | `scripts/train_stat_mlp.py` | 统计特征 MLP | B6.0 |
| 13 | `scripts/train_text_classifier_ml.py` | TF-IDF 文字 MLP | B6.0 |
| 14 | `scripts/evaluate_hetero_ensemble.py` | 异构集成评估 | B6.0 |
| 15 | `scripts/auto_retrain.sh` | 数据飞轮 | B6.1 |
| 16 | `scripts/distill_ensemble.py` | 知识蒸馏 | B6.2 |
| 17 | `tests/unit/test_monitoring.py` | 28 个测试 | B5.3 |
| 18 | `tests/unit/test_low_conf_queue.py` | 26 个测试 | B5.3 |

### 模型文件（7 个）

| 文件 | 精度 | 大小 | 阶段 |
|------|------|------|------|
| v3 | 91.0% | 1.6MB | B5.0 |
| v3_int8 | ~91.0% | 430KB | B5.2 |
| v4 | **91.9%** | 1.6MB | B5.7 |
| v4_int8 | ~91.9% | 430KB | B5.8 |
| stat_mlp | **94.4%** | ~100KB | B6.0 |
| text_tfidf | **73.7%** | ~500KB | B6.0 |
| v5 (蒸馏) | 训练中 | 1.6MB | B6.2 |

### 设计文档（26 个）

B5.0-B5.9 + B6.0-B6.3（训练报告 + 提升计划各一对）

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
