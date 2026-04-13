# B4 Graph2D 分类准确率提升计划与验证方案

**日期**: 2026-04-13  
**基线**: 56.2% val accuracy (5 类, 对比预训练+微调, 1,023 样本)  
**目标**: 75%+ val accuracy (24 类)  

---

## 1. 现状诊断

### 性能瓶颈分析

| 瓶颈 | 当前值 | 影响程度 | 可提升空间 |
|------|--------|----------|-----------|
| 预训练不足 | 10 epochs, loss 未收敛 | 高 | loss 1.73 → 预计 1.2-1.5 |
| 微调不足 | 15 epochs, loss 未收敛 | 高 | val_acc 56% → 预计 65%+ |
| 类别过少 | 5 类（粗粒度目录标签） | 中 | 扩展到 24 类 |
| 类别不均衡 | "其他"占 52.6% | 高 | 加权采样/Focal Loss |
| 数据量 | 1,023 训练样本 | 中 | 增强数据 +2,046，全量 5,417 |
| 模型容量 | 2 层 EdgeSAGE, hidden=128 | 低 | 可增加到 3-4 层 |

### 改进优先级排序
```
投入产出比从高到低:
1. [高] 增加训练 epochs（零成本，直接提升）
2. [高] 类别均衡处理（Focal Loss + Balanced Sampler）
3. [高] 使用增强数据（已生成 2,046 个）
4. [中] 扩展到 24 类完整分类
5. [中] 模型架构升级（3 层 + 残差）
6. [低] 超参数搜索（网格/贝叶斯）
```

---

## 2. 分阶段提升计划

### Phase B4.1: 充分训练（预计 56% → 65%）

**原理**: 当前预训练和微调都未收敛，loss 仍在下降。

#### 预训练强化
```bash
python3 scripts/pretrain_graph2d_contrastive.py \
  --dxf-dir data/training_v8 \
  --epochs 50 \
  --batch-size 8 \
  --lr 0.001 \
  --temperature 0.3 \
  --output models/graph2d_pretrained_v2_50ep.pth
```

**变更**:
- epochs: 10 → 50（5x）
- temperature: 0.5 → 0.3（更强的对比信号）

#### 微调强化
```bash
python3 scripts/finetune_graph2d_from_pretrained.py \
  --pretrained models/graph2d_pretrained_v2_50ep.pth \
  --manifest data/manifests/training_v8_manifest.csv \
  --dxf-dir data/training_v8 \
  --epochs 50 \
  --batch-size 8 \
  --encoder-lr 0.00005 \
  --head-lr 0.0005 \
  --patience 10 \
  --output models/graph2d_finetuned_v2_50ep.pth
```

**变更**:
- epochs: 15 → 50
- patience: 5 → 10
- 学习率降低 50%（避免过拟合）

#### 验证标准
- [ ] 预训练 loss < 1.5（当前 1.73）
- [ ] 微调 val_acc > 62%（当前 56.2%）
- [ ] 两次运行标准差 < 2%

---

### Phase B4.2: 类别均衡 + 增强数据（预计 65% → 72%）

**原理**: "其他"类占 52.6% 导致模型偏向多数类。

#### 修改 `scripts/finetune_graph2d_from_pretrained.py`

添加以下功能:

```python
# 1. Focal Loss (替换 CrossEntropyLoss)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 2. Balanced Sampler
class_counts = Counter(label for _, label in dataset.samples)
weights = [1.0 / class_counts[label] for _, label in dataset.samples]
sampler = WeightedRandomSampler(weights, len(weights))
```

#### 使用增强数据
```bash
# 将增强数据加入预训练
python3 scripts/pretrain_graph2d_contrastive.py \
  --dxf-dir data/training_v8 \
  --dxf-dir data/augmented \
  --epochs 50 \
  --output models/graph2d_pretrained_v2_aug.pth
```

注意: 需要修改脚本支持多个 `--dxf-dir` 或合并目录。

#### 验证标准
- [ ] 少数类（传动件）recall > 40%（预估当前 <20%）
- [ ] val_acc > 68%
- [ ] per-class accuracy 标准差 < 15%

---

### Phase B4.3: 扩展到 24 类（预计 72% → 60-65% on 24 classes）

**原理**: 从 5 类扩展到 24 类，准确率会下降，但系统能力大幅提升。

#### 步骤
1. 使用完整 `unified_manifest_v2.csv`（5,417 样本，24 类）
2. 优化 DXF 加载：预处理为 pickle 缓存，避免每次重新解析
3. 对样本 < 30 的类别使用过采样

```bash
# 预处理 DXF → 图缓存（一次性）
python3 scripts/preprocess_dxf_to_graphs.py \
  --manifest data/manifests/unified_manifest_v2.csv \
  --output data/graph_cache/

# 微调 24 类
python3 scripts/finetune_graph2d_from_pretrained.py \
  --pretrained models/graph2d_pretrained_v2_aug.pth \
  --manifest data/manifests/unified_manifest_v2.csv \
  --epochs 50 \
  --output models/graph2d_finetuned_24class.pth
```

#### 需要新建的脚本
- `scripts/preprocess_dxf_to_graphs.py` — DXF → PyTorch tensor pickle 缓存
  - 解决 ezdxf 加载瓶颈（4,743 文件 90 分钟 → 缓存后 <1 分钟）
  - 输出: `data/graph_cache/{hash}.pt` (节点特征 + 边 + 标签)

#### 验证标准
- [ ] 24 类 val_acc > 55%
- [ ] Top-3 accuracy > 75%
- [ ] 每类至少 10 个正确预测

---

### Phase B4.4: 模型架构升级（预计 +3-5%）

**原理**: 更深的 GNN 可以捕获更远距离的结构关系。

#### 架构变更

| 参数 | 当前 | 升级 |
|------|------|------|
| GNN 层数 | 2 | 3 (+残差连接) |
| Hidden dim | 128 | 256 |
| 节点特征 | 19 维 | 25 维 (+area, perimeter, aspect_ratio, curvature, is_hatching, is_centerline) |
| 最大节点数 | 200 | 500 |
| Dropout | 0.2 | 0.3 |
| 图池化 | mean | attention pooling |

#### 修改文件
- `src/ml/train/model_2d.py`: 添加 `GraphEncoderV2` 类
- `src/ml/vision_2d.py`: 增加 6 个新节点特征

#### 验证标准
- [ ] 同数据集 val_acc 提升 > 3%
- [ ] 推理延迟 < 50ms/图（不能太慢）
- [ ] 参数量 < 5M

---

### Phase B4.5: Hybrid 权重调优

**原理**: 将改进后的 Graph2D 集成回 HybridClassifier。

#### 当前 vs 目标权重

```yaml
# 当前权重
filename:   0.70  # 过度依赖
graph2d:    0.30
titleblock: 0.20 (disabled)

# 目标权重（Graph2D 达到 65%+ 后）
filename:   0.45  # 降低
graph2d:    0.35  # 提升
titleblock: 0.25  # 启用
process:    0.15
```

#### 权重搜索方法
```python
# Grid search on golden test set
best_acc, best_weights = 0, None
for fn_w in [0.3, 0.4, 0.5, 0.6]:
    for g2d_w in [0.2, 0.3, 0.4]:
        for tb_w in [0.1, 0.2, 0.3]:
            acc = evaluate_hybrid(golden_set, fn_w, g2d_w, tb_w)
            if acc > best_acc:
                best_acc, best_weights = acc, (fn_w, g2d_w, tb_w)
```

#### 关键测试
- [ ] **有文件名时**: 准确率 ≥ 95%（不能退步）
- [ ] **无文件名时**: 准确率 > 60%（当前 ~30%）
- [ ] **对抗测试**: 文件名故意错误时，Graph2D 能纠正

---

## 3. 里程碑与时间线

```
Week 1: B4.1 充分训练
  ├── 预训练 50 epochs (~2h CPU)
  ├── 微调 50 epochs (~1.5h CPU)
  └── 验证: val_acc > 62%

Week 2: B4.2 均衡 + 增强
  ├── 实现 Focal Loss + Balanced Sampler
  ├── 加入增强数据
  └── 验证: val_acc > 68%, 少数类 recall > 40%

Week 3: B4.3 扩展 24 类
  ├── 实现 DXF → graph 缓存
  ├── 全量训练 24 类
  └── 验证: 24-class val_acc > 55%, top-3 > 75%

Week 4: B4.4 架构升级
  ├── GraphEncoderV2 (3层+残差+attention pooling)
  ├── 新增 6 维节点特征
  └── 验证: +3-5% accuracy

Week 5: B4.5 Hybrid 集成
  ├── 替换 Graph2D 模型
  ├── Grid search 最优权重
  └── 验证: 有名 ≥95%, 无名 >60%
```

---

## 4. 验证框架

### 评估指标

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| Overall Accuracy | correct / total | > 65% (5类), > 55% (24类) |
| Per-class Accuracy | 每类 correct / class_total | 标准差 < 15% |
| Top-3 Accuracy | 前3预测含正确标签 | > 75% |
| Macro F1 | 类别平均 F1 | > 0.55 |
| Confusion Matrix | 5×5 或 24×24 | 对角线占优 |
| ECE (校准误差) | Expected Calibration Error | < 0.10 |

### Golden Test Set

从每类随机抽取 20% 不参与训练的样本作为 golden set：

```python
# 构建 golden test set
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, golden_idx = next(sss.split(X, y))
```

### 回归检查清单

每次模型更新后必须验证：
- [ ] Golden set accuracy ≥ 上一版本
- [ ] 无文件名场景 accuracy 不退步
- [ ] 推理延迟 < 100ms/文件
- [ ] 模型文件大小 < 50MB
- [ ] Superpass gate 全绿

### 自动化验证脚本

```bash
# 运行完整验证
python3 scripts/evaluate_graph2d_model.py \
  --model models/graph2d_finetuned_v2.pth \
  --manifest data/manifests/training_v8_manifest.csv \
  --golden-ratio 0.2 \
  --output docs/design/B4_EVAL_REPORT.md
```

---

## 5. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 更多 epochs 导致过拟合 | 中 | 监控 train/val gap，early stopping |
| 24 类准确率低于 50% | 中 | 先合并低频类，渐进扩展 |
| DXF 缓存磁盘空间不足 | 低 | 每个 .pt 文件约 10KB，5000 文件 ≈ 50MB |
| Focal Loss 效果不明显 | 低 | 回退到加权 CE Loss |
| 架构升级推理太慢 | 低 | 限制层数为 3，用 knowledge distillation 压缩 |

---

## 6. 成功标准

| 阶段 | 目标 | 判定标准 |
|------|------|----------|
| B4.1 | 充分训练 | 5 类 val_acc > 62% |
| B4.2 | 均衡+增强 | 5 类 val_acc > 68%, 少数类 recall > 40% |
| B4.3 | 24 类 | 24 类 val_acc > 55%, top-3 > 75% |
| B4.4 | 架构升级 | 同数据 +3-5% |
| B4.5 | Hybrid 集成 | 有名 ≥95%, 无名 >60% |
| **最终** | **生产就绪** | **24 类 val_acc > 65%, hybrid 无名 >60%** |

---

*文档版本: 1.0*  
*创建日期: 2026-04-13*
