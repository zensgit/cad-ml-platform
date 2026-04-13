# B4.2 类别均衡训练计划与验证方案

**日期**: 2026-04-13  
**基线**: 56.9% val_acc (5 类, B4.1 充分训练后)  
**目标**: 65%+ val_acc，少数类 recall > 40%  
**核心策略**: Focal Loss + Balanced Sampler + 增强数据  

---

## 1. 问题诊断

### 为什么 57% 是天花板？

当前类别分布：
```
其他:   538 (52.6%)  ████████████████████████████
法兰:   163 (15.9%)  ████████
箱体:   133 (13.0%)  ███████
轴类:   132 (12.9%)  ███████
传动件:  57 ( 5.6%)  ███
```

模型学到的策略近似于「大部分预测其他 + 偶尔预测法兰/箱体/轴类」：
- 预测"其他"的 recall 可能 >80%
- 预测"传动件"的 recall 可能 <10%
- 整体 accuracy 被多数类拉高，掩盖了少数类的糟糕表现

### 解决方案

| 方法 | 作用 | 预期效果 |
|------|------|----------|
| **Focal Loss** | 降低简单样本（多数类）权重，聚焦困难样本 | +3-5% macro F1 |
| **Balanced Sampler** | 每 epoch 让每类被采样次数相等 | 少数类见更多 |
| **增强数据** | 2,046 个增强 DXF 加入训练 | 数据量 3x |
| **Class Weight** | CE Loss 中按类频率倒数加权 | 惩罚多数类错误 |

---

## 2. 实现计划

### 2.1 Focal Loss 实现

```python
class FocalLoss(nn.Module):
    """Focal Loss: 降低易分类样本权重，聚焦困难样本"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        # alpha: 每类权重（None=均匀）
        # gamma: 聚焦参数（0=CE, 2=标准focal）
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### 2.2 Balanced Sampler 实现

```python
from torch.utils.data import WeightedRandomSampler
from collections import Counter

# 计算每类样本数
class_counts = Counter(label for _, label in dataset.samples)
# 每个样本的采样权重 = 1/类频率
sample_weights = [1.0 / class_counts[label] for _, label in dataset.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
# DataLoader 使用 sampler（不能同时用 shuffle=True）
train_loader = DataLoader(dataset, batch_size=8, sampler=sampler, collate_fn=collate_fn)
```

### 2.3 增强数据集成

合并 training_v8 + augmented 为统一数据源：
```bash
# augmented/ 目录结构与 training_v8/ 相同（传动件/其他/壳体类/轴类/连接件）
# 总样本: 1,023 + 2,046 = 3,069
```

### 2.4 消融实验设计

| 实验 | Focal | Balanced | 增强数据 | 预期 |
|------|-------|----------|----------|------|
| A (基线) | ✗ | ✗ | ✗ | 56.9% |
| B (+Focal) | ✓ | ✗ | ✗ | ~60% |
| C (+Balanced) | ✗ | ✓ | ✗ | ~59% |
| D (+Focal+Balanced) | ✓ | ✓ | ✗ | ~62% |
| **E (全部)** | ✓ | ✓ | ✓ | **~65%** |

---

## 3. 验证标准

### 主要指标

| 指标 | B4.1 基线 | B4.2 目标 | 验证方法 |
|------|----------|----------|----------|
| Overall Val Acc | 56.9% | **65%+** | 20% held-out |
| Macro F1 | ~0.35 (估) | **0.50+** | sklearn.metrics |
| 少数类(传动件) Recall | <10% (估) | **40%+** | 混淆矩阵 |
| 多数类(其他) Recall | >80% (估) | **70-80%** | 允许小幅下降 |
| Per-class Acc 标准差 | >20% (估) | **<15%** | 类间均衡度 |

### 验证流程

每次实验输出：
1. Overall accuracy + Macro F1
2. Per-class precision / recall / F1
3. 混淆矩阵 (5×5)
4. 训练曲线 (loss + acc per epoch)
5. Early stopping epoch

### 回归检查

- [ ] Overall acc 不低于基线 56.9%
- [ ] 无类别 recall 为 0%
- [ ] 训练不发散（loss 持续下降或稳定）

---

## 4. 时间线

```
Step 1 (30min): 实现 Focal Loss + Balanced Sampler
Step 2 (30min): 构建合并 manifest (training_v8 + augmented)
Step 3 (2.5h):  运行消融实验 D (Focal+Balanced, 1023样本)
Step 4 (2.5h):  运行实验 E (Focal+Balanced+增强, 3069样本)
Step 5 (30min): 生成评估报告
```

---

## 5. 风险

| 风险 | 应对 |
|------|------|
| Focal Loss gamma 过大导致欠拟合 | 从 gamma=1.0 开始，逐步调到 2.0 |
| Balanced Sampler 过采样少数类导致过拟合 | 配合增强数据使用 |
| 增强数据标签噪声 | 增强数据继承原始目录标签，噪声与原数据相同 |
| 整体 acc 下降（牺牲多数类） | 监控"其他"类 recall，不低于 65% |

---

*文档版本: 1.0*  
*创建日期: 2026-04-13*
