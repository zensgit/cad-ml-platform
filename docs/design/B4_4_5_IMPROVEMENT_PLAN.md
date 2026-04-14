# B4.4-B4.5 详细提升计划与验证方案

**日期**: 2026-04-14  
**基线**: B4.3 — 24 类 val_acc=79.65%，每 epoch 1.7s  
**目标**: B4.4 → 83%+（架构升级），B4.5 → Hybrid 无文件名 > 70%

---

## 当前里程碑回顾

```
B3   对比预训练+微调   5类  56.2%  ✓ 完成
B4.1 充分训练         5类  56.9%  ✓ 完成
B4.2 Focal+增强数据   5类  61.2%  ✓ 完成
B4.3 缓存+24类        24类 79.65% ✓ 完成（大幅超标）
B4.4 架构升级         24类 83%+   ← 当前阶段
B4.5 Hybrid集成       —    无名>70% ← 后续
```

---

## Phase B4.4：模型架构升级

**原理**：当前 2 层 EdgeGraphSAGE (hidden=128) 捕获的图结构信息有限，3 层 + 残差 + Attention Pooling 可以建模更长距离的结构关系。

### 4.4.1 架构对比

| 参数 | B4.3 当前 | B4.4 目标 |
|------|-----------|----------|
| GNN 层数 | 2 | **3（+残差连接）** |
| Hidden dim | 128 | **256** |
| 图池化 | Mean pooling | **Attention Pooling** |
| Dropout | — | **0.3** |
| 节点特征维度 | 19 | 19（不变） |
| 参数量（估） | ~0.15M | **~0.8M** |

### 4.4.2 实现计划

**Step 1：新增 GraphEncoderV2**（`src/ml/train/model_2d.py`）

```python
class GraphEncoderV2(nn.Module):
    """3 层 EdgeGraphSAGE + 残差 + Attention Pooling."""

    def __init__(self, node_dim=19, edge_dim=7, hidden_dim=256, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import SAGEConv, GlobalAttention

        self.lin_in = nn.Linear(node_dim, hidden_dim)

        # 3 层 SAGEConv（带边特征投影）
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.sage3 = SAGEConv(hidden_dim, hidden_dim)

        # 残差映射（维度相同时恒等）
        self.res1 = nn.Identity()
        self.res2 = nn.Identity()

        # Attention Pooling（替换 scatter_mean）
        self.pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = F.relu(self.lin_in(x))

        # Layer 1
        h = F.relu(self.norm1(self.sage1(x, edge_index)))
        h = self.dropout(h) + self.res1(x)  # 残差

        # Layer 2
        h2 = F.relu(self.norm2(self.sage2(h, edge_index)))
        h2 = self.dropout(h2) + self.res2(h)  # 残差

        # Layer 3（无残差，维度变化）
        h3 = F.relu(self.norm3(self.sage3(h2, edge_index)))
        h3 = self.dropout(h3)

        # Attention pooling → 图级表示
        return self.pool(h3, batch)
```

**Step 2：训练命令**

```bash
python3 -u scripts/finetune_graph2d_from_pretrained.py \
    --manifest data/graph_cache/cache_manifest.csv \
    --use-cache \
    --tail-oversample-threshold 50 \
    --pretrained None \
    --hidden-dim 256 \
    --model-type edge_sage_v2 \
    --epochs 80 \
    --encoder-lr 0.0001 \
    --head-lr 0.001 \
    --patience 15 \
    --output models/graph2d_finetuned_24class_v2.pth
```

注：B4.4 从头训练（不加载 B4.3 预训练权重，因为架构变化导致权重不兼容）。

**Step 3：消融实验**

| 实验 | 变更 | 预期 |
|------|------|------|
| A | +第3层 only | +1-2% |
| B | +Attention Pooling only | +1-2% |
| C | +hidden=256 only | +1-2% |
| **D（完整）** | 全部 | **+3-5%（→83%）** |

### 4.4.3 验证标准

| 指标 | B4.3 基线 | B4.4 目标 |
|------|----------|----------|
| 24 类 Val Acc | 79.65% | **≥ 83%** |
| 主类(法兰/轴/箱) recall | 82-87% | **≥ 85%** |
| 难类(传动件/轴承座) recall | 0% | **≥ 10%** |
| 推理延迟 | <5ms/图 | **<10ms/图** |
| 模型参数量 | ~0.15M | **<5M** |

### 4.4.4 风险

| 风险 | 概率 | 应对 |
|------|------|------|
| Attention Pooling 过拟合 | 中 | Dropout 0.3 + 早停 patience=15 |
| hidden=256 内存翻倍 | 低 | CPU 足够，单图约 100KB 内存 |
| 从头训练收敛慢 | 中 | 先预训练 30ep，再微调 |
| 推理延迟超 10ms | 低 | 限制最大节点数 200（已有） |

---

## Phase B4.5：Hybrid 分类器集成

**原理**：将 Graph2D 从 5 类 61% 升级为 24 类 80%，重新平衡 HybridClassifier 权重，使无文件名场景从当前 ~30% 提升到 70%+。

### 4.5.1 当前 HybridClassifier 现状

```python
# 当前权重（估）
filename_weight   = 0.70   # 文件名匹配
graph2d_weight    = 0.30   # Graph2D（5类，61.2%）
titleblock_weight = 0.00   # 标题栏（未启用）
```

**问题**：
- Graph2D 输出 5 类概率，但 Hybrid 系统需要 24 类（不兼容）
- 无文件名时完全依赖 graph2d，但原模型只有 5 类

### 4.5.2 实现步骤

**Step 1：加载 24 类 Graph2D 模型**（`src/ml/vision_2d.py`）

```python
# 修改 Graph2DClassifier.predict()
def predict(self, dxf_path: str) -> ClassificationResult:
    graph = self._load_graph(dxf_path)
    logits = self.model(graph)  # [24] 输出
    probs = F.softmax(logits, dim=0)
    
    # label_map: {class_name: idx}，来自 checkpoint
    label_map = self.model_checkpoint['label_map']
    inv_map = {v: k for k, v in label_map.items()}
    
    top_k = torch.topk(probs, k=3)
    return ClassificationResult(
        predicted_class=inv_map[top_k.indices[0].item()],
        confidence=top_k.values[0].item(),
        top3=[(inv_map[i.item()], v.item()) for i, v in zip(top_k.indices, top_k.values)],
    )
```

**Step 2：统一 24 类标签空间**

HybridClassifier 中的文件名匹配也需要映射到 24 类：

```python
# 当前 5 类 → 24 类映射
LEGACY_5_TO_24 = {
    "法兰":  ["法兰"],
    "箱体":  ["箱体"],
    "轴类":  ["轴类", "轴承座"],
    "传动件": ["传动件", "旋转组件"],
    "其他":  ["分离器", "阀门", "弹簧", "支架", ...],
}
```

**Step 3：权重搜索**

```python
from itertools import product

best_acc, best_weights = 0, None
for fn_w, g2d_w, tb_w in product([0.3, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5], [0.1, 0.2]):
    if fn_w + g2d_w + tb_w > 1.0: continue
    acc = evaluate_hybrid(golden_set, fn_w, g2d_w, tb_w)
    if acc > best_acc:
        best_acc, best_weights = acc, (fn_w, g2d_w, tb_w)
```

**目标权重（估）**：
```yaml
filename:   0.45  # 降低（从 0.70）
graph2d:    0.40  # 提升（从 0.30，现在 24 类 80% 有实际意义）
titleblock: 0.15  # 启用（之前禁用）
```

### 4.5.3 验证标准

| 场景 | 当前 | B4.5 目标 |
|------|------|----------|
| **有文件名（正常）** | ≥ 95% | **≥ 95%（不能退步）** |
| **无文件名** | ~30% | **≥ 70%** |
| 文件名故意错误（对抗） | — | Graph2D 能纠正，acc ≥ 50% |
| Top-3 accuracy（全量） | — | **≥ 90%** |

### 4.5.4 关键测试清单

- [ ] Golden set（每类 20% held-out）上 overall acc ≥ 上一版本
- [ ] 无文件名场景：从 100 个无文件名 DXF 中至少 70 个正确
- [ ] 有文件名场景：从 100 个标准 DXF 中至少 95 个正确
- [ ] 推理延迟：< 100ms/文件（含 DXF 解析 + 图构建 + 推理）
- [ ] 内存：< 500MB（模型加载后）

---

## Phase B4.6（可选）：难类专项提升

针对 B4.3/B4.4 中 recall=0% 的 4 类（传动件、轴承座、搅拌器、阀门）：

### 专项策略

| 类别 | 当前样本 | 策略 |
|------|----------|------|
| 传动件 | 49 | 专项增强 × 20（旋转/缩放/镜像），生成 ~1000 样本 |
| 轴承座 | 74 | 专项增强 × 10，生成 ~700 样本 |
| 搅拌器 | 24 | 专项增强 × 20，生成 ~500 样本 |
| 阀门 | 29 | 专项增强 × 15，生成 ~500 样本 |

```bash
# 专项增强
python3 scripts/augment_dxf.py \
    --manifest data/manifests/unified_manifest_v2.csv \
    --classes 传动件 轴承座 搅拌器 阀门 \
    --augment-factor 20 \
    --output-dir data/augmented_hard_classes
```

**预期效果**：这 4 类 recall 从 0% → 30%+，overall val_acc +1-2%

---

## 整体时间线

```
B4.4 (2-3小时):
  ├── Step 1 (~30min): 实现 GraphEncoderV2
  ├── Step 2 (~1h):    80 epoch 训练（缓存，约 2.5 分钟/运行）
  └── Step 3 (~30min): 验证 + 报告

B4.5 (2-3小时):
  ├── Step 1 (~1h):    修改 vision_2d.py 加载 24 类模型
  ├── Step 2 (~1h):    权重搜索 + 测试
  └── Step 3 (~30min): 验证有名/无名/对抗场景

B4.6 (可选, ~3小时):
  ├── 专项增强 4 个难类
  └── 重训验证
```

---

## 成功标准总览

| 阶段 | 指标 | 目标 | 当前状态 |
|------|------|------|----------|
| B4.3 | 24类 val_acc | ≥ 55% | **79.65% ✓** |
| **B4.4** | 24类 val_acc | **≥ 83%** | 待执行 |
| **B4.5** | 无文件名 acc | **≥ 70%** | 待执行 |
| B4.6 | 难类 recall | ≥ 30% | 可选 |
| **最终** | **生产就绪** | **有名≥95%, 无名≥70%** | — |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
