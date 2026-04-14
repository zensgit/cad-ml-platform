# B4 Graph2D 分类全面提升路线图

**日期**: 2026-04-14（持续更新）  
**项目**: CAD ML Platform — Graph2D 零件分类  
**当前最优**: B4.3 — 24 类 val_acc=**79.65%**（每 epoch 1.7s）

---

## 里程碑全览

| 阶段 | 日期 | 类别数 | Val Acc | 关键技术 | 状态 |
|------|------|--------|---------|----------|------|
| B3 基线 | 04-13 | 5 | 56.2% | 对比预训练 + 微调 | ✓ |
| B4.1 充分训练 | 04-13 | 5 | 56.9% | 50ep 预训练 | ✓ |
| B4.2a 激进均衡 | 04-13 | 5 | 29% | Focal γ=2 + 均衡采样 | ✓（失败） |
| B4.2b 温和均衡 | 04-14 | 5 | 61.2% | Focal γ=1 + 增强数据 | ✓ |
| B4.3 缓存+24类 | 04-14 | 24 | **79.65%** | DXF缓存 + 长尾采样 | ✓ |
| **B4.4 架构升级** | 04-14 | 24 | **目标83%** | GraphEncoderV2 | 进行中 |
| B4.5 Hybrid集成 | 待定 | 24 | 无名>70% | HybridClassifier | 待执行 |
| B4.6 难类增强 | 待定 | 24 | +1-2% | 专项数据增强 | 可选 |

---

## Phase B4.4：架构升级（当前）

### 核心变更
```
GraphEncoder（B4.3）         GraphEncoderV2（B4.4）
─────────────────           ──────────────────────
EdgeSageLayer ×2            EdgeSageLayer ×3
hidden=128                  hidden=256
Mean Pooling                Attention Pooling
无残差                       层间残差连接
无归一化                     LayerNorm ×3
0.15M 参数                  0.41M 参数
```

### 验证标准
- val_acc ≥ 83%
- 主类（法兰/轴类/箱体）recall 不退步
- 难类（传动件/轴承座）recall > 0%

---

## Phase B4.5：Hybrid 分类器集成

### 目标
将 Graph2D（24类，80%+）集成回 HybridClassifier，使无文件名场景从 ~30% 提升至 70%+。

### 关键任务

**1. 更新 Graph2D 加载逻辑**（`src/ml/vision_2d.py`）
```python
# 当前：加载 5 类模型
# 目标：加载 24 类 B4.4 模型
checkpoint = torch.load('models/graph2d_finetuned_24class_v2.pth')
label_map = checkpoint['label_map']  # 24 类映射
```

**2. 统一标签空间**

HybridClassifier 内部需要在文件名匹配（24类）和 Graph2D（24类）之间对齐，合并成统一 confidence 分布：
```python
# 两个预测器都输出 24 类 logits，直接加权融合
hybrid_logits = filename_w * fn_logits + graph2d_w * g2d_logits + tb_w * tb_logits
```

**3. 权重搜索**

在 golden test set 上 grid search：
```python
for fn_w in [0.3, 0.4, 0.5, 0.6]:
    for g2d_w in [0.3, 0.4, 0.5]:
        for tb_w in [0.0, 0.1, 0.2]:
            if fn_w + g2d_w + tb_w > 1.1: continue
            acc = evaluate_hybrid(golden_set, fn_w, g2d_w, tb_w)
```

**4. 场景验证**

| 场景 | 测试方法 | 目标 |
|------|----------|------|
| 有标准文件名 | 100 个标准 DXF | ≥ 95% |
| 无文件名（纯图） | 100 个重命名 DXF | **≥ 70%** |
| 文件名故意错误 | 30 个错标签 DXF | Graph2D 能纠正 |
| 混合场景 | 200 个随机 DXF | ≥ 85% |

### 验证标准
- [ ] 有文件名 acc ≥ 95%（不能退步）
- [ ] 无文件名 acc ≥ 70%（当前 ~30%）
- [ ] 推理延迟 < 100ms/文件
- [ ] Superpass gate 全绿

---

## Phase B4.6：难类专项提升（可选）

### 4 个 recall=0% 的类别

| 类别 | 总样本 | 验证样本 | 根因 |
|------|--------|----------|------|
| 传动件 | 49 | 8 | 极少，与轴类 DXF 相似 |
| 轴承座 | 74 | 10 | 形态多样，训练不足 |
| 搅拌器 | 24 | 4 | 数量太少 |
| 阀门 | 29 | 3 | 验证集过小，无法评估 |

### 专项方案

**方案 A：DXF 专项增强**
```bash
python3 scripts/augment_dxf.py \
    --classes 传动件 轴承座 搅拌器 阀门 \
    --augment-factor 20 \
    --output-dir data/augmented_hard_classes
# 生成 ~2,000 个额外样本
```

**方案 B：Few-shot 迁移**
- 将难类视为 few-shot 问题，使用 Prototypical Networks
- 类中心 = 所有样本嵌入的均值
- 预测 = 找最近类中心

**方案 C：标签合并**
- 将样本 < 20 的类合并到相似类（如阀门→传动件）
- 减少为 20 类，提升整体 recall

**推荐**：先执行方案 A，效果不足再考虑 B/C。

---

## 数据状态

| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| `data/training_v8/` | 1,023 | 原始训练（5类目录标签） |
| `data/augmented/` | 2,046 | 增强数据（B4.2使用） |
| `data/manifests/unified_manifest_v2.csv` | 5,417 | 全量（24类，含标注） |
| `data/graph_cache/` | 4,574 pt | 图缓存（B4.3+使用） |

---

## 模型版本历史

| 模型文件 | 类别 | Val Acc | 备注 |
|----------|------|---------|------|
| `graph2d_pretrained_contrastive.pth` | — | — | 10ep 对比预训练 |
| `graph2d_pretrained_v2_50ep.pth` | — | — | 50ep 对比预训练（B4.1） |
| `graph2d_finetuned_v2.pth` | 5 | 56.2% | B3 基线 |
| `graph2d_finetuned_v2_50ep.pth` | 5 | 56.9% | B4.1 |
| `graph2d_finetuned_b42_mild.pth` | 5 | 61.2% | B4.2b |
| `graph2d_finetuned_24class_v1.pth` | 24 | **79.65%** | B4.3（当前最优） |
| `graph2d_finetuned_24class_v2.pth` | 24 | 目标83% | B4.4（进行中） |

---

## 关键技术经验

| 发现 | 影响 |
|------|------|
| 细粒度标签（24类文件名）比粗粒度（5类目录名）质量高得多 | B4.3 从 61% 跳到 80% 的核心原因 |
| DXF 图缓存消除解析瓶颈 | epoch 从 7 分钟降至 1.7 秒 |
| 过度均衡补偿导致多数类崩溃 | B4.2a 教训（Focal γ=2 + 采样 → 整体 29%） |
| 增强数据（3x）> 更多 epoch（B4.1 教训） | 数据质量优于训练时长 |
| 24 类长尾过采样（threshold=50）有效 | 中型类（50-100 样本）recall 显著提升 |

---

*文档版本: 1.1*  
*最后更新: 2026-04-14*
