# B4.4 架构升级训练计划与验证方案

**日期**: 2026-04-14  
**基线**: B4.3 — 24 类 val_acc=79.65%，GraphEncoder 2层 hidden=128，Mean Pooling  
**目标**: val_acc ≥ 83%（+3-5pp），难类 recall > 0%  
**核心策略**: GraphEncoderV2（3层+残差+Attention Pooling，hidden=256）

---

## 1. 架构对比

| 参数 | B4.3（当前） | B4.4（目标） | 变化 |
|------|------------|------------|------|
| GNN 层数 | 2 | **3** | +1 层 |
| Hidden dim | 128 | **256** | 2x |
| 残差连接 | ✗ | **✓**（层间残差） | 新增 |
| 归一化 | ✗ | **LayerNorm × 3** | 新增 |
| 图池化 | Mean Pooling | **Attention Pooling** | 替换 |
| Dropout | 0.2 | **0.3** | 提升 |
| 参数量 | ~0.15M | **~0.41M** | 2.7x |
| 嵌入维度 | 128 | **256** | 2x |

### GraphEncoderV2 结构

```
输入 [N, 19]
  → lin_in: Linear(19→256) + ReLU
  → sage1: EdgeSageLayer(256, 7, 256) + LayerNorm + Dropout + 残差
  → sage2: EdgeSageLayer(256, 7, 256) + LayerNorm + Dropout + 残差
  → sage3: EdgeSageLayer(256, 7, 256) + LayerNorm + Dropout
  → Attention Pooling: gate=Linear(256→1), per-graph softmax
  → 图嵌入 [B, 256]
  → 分类头: Linear(256→24)
```

### Attention Pooling vs Mean Pooling

```
Mean Pooling:  output = (1/N) * Σ h_i          # 等权重
Attn Pooling:  output = Σ softmax(gate(h_i)) * h_i  # 关键节点加权
```

关键节点（如标注、尺寸线）对类别判断贡献更大，Attention Pooling 能自动学习这种偏重。

---

## 2. 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据 | `data/graph_cache/cache_manifest.csv` | 4,574 缓存样本，24 类 |
| 权重初始化 | **部分加载预训练**（sage1/sage2 from B4.1） | sage3/lin_in/norm/gate 随机初始化 |
| Focal Loss gamma | 1.0 | 与 B4.3 相同 |
| 长尾过采样 | threshold=50 | 与 B4.3 相同 |
| Encoder LR | 0.0001（预训练层）/ 0.001（新增层） | 差异学习率 |
| Head LR | 0.001 | 分类头 |
| Batch size | 16 | |
| 调度器 | CosineAnnealingLR（T_max=80） | |
| Max epochs | 80 | 比 B4.3 多 20 epoch（更大模型需更长收敛） |
| Patience | 15 | |
| 输出模型 | `models/graph2d_finetuned_24class_v2.pth` | |

---

## 3. 消融实验设计

| 实验 | 变更 | 预期 vs B4.3 基线 |
|------|------|------------------|
| A（+第3层 only） | 3层 + hidden=128 + Mean Pool | +1-2% |
| B（+hidden=256 only） | 2层 + hidden=256 + Mean Pool | +1-2% |
| C（+Attn Pool only） | 2层 + hidden=128 + Attn Pool | +1-2% |
| **D（完整 B4.4）** | 3层 + hidden=256 + Attn Pool + 残差 + LN | **+3-5%（→83%）** |

**本次运行**：直接跑实验 D（完整 B4.4），如未达标再做消融。

---

## 4. 验证标准

### 主要指标

| 指标 | B4.3 基线 | B4.4 目标 | 判定 |
|------|----------|----------|------|
| 24 类 Val Acc | 79.65% | **≥ 83%** | 主指标 |
| 法兰 recall | 82.6% | **≥ 85%** | 主类不能退步 |
| 轴类 recall | 86.8% | **≥ 87%** | 主类不能退步 |
| 箱体 recall | 79.5% | **≥ 80%** | 主类不能退步 |
| 难类（传动件/轴承座）recall | 0% | **≥ 10%** | 激活标准 |
| 推理延迟 | <5ms | **< 15ms** | 更大模型可允许 |

### 回归检查

- [ ] 所有主类（法兰/轴类/箱体）recall 不低于 B4.3
- [ ] Overall val_acc 不低于 B4.3（79.65%）
- [ ] 每 epoch 时间 < 10s（缓存 + 更大模型）
- [ ] 模型文件 < 10MB

---

## 5. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| Attention Pooling 过拟合 | 中 | Dropout 0.3 + patience=15 |
| hidden=256 收敛更慢 | 中 | Max epochs 提升至 80 |
| 难类 recall 仍为 0% | 中 | B4.6 专项增强 |
| 主类 recall 退步 | 低 | 如发生，回退到 B4.3 模型 |
| index_reduce_ beta 警告 | 低 | 非错误，不影响结果 |

---

## 6. 成功标准

| 结果 | 判定 | 行动 |
|------|------|------|
| val_acc ≥ 83% | ✓ 达标 | 进入 B4.5 Hybrid 集成 |
| 80% ≤ val_acc < 83% | ⚠️ 部分达标 | 考虑 B4.6 难类增强后再集成 |
| val_acc < 80% | ✗ 退步 | 回退到 B4.3，检查 V2 实现 |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
