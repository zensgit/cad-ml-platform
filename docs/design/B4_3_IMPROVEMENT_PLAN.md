# B4.3 扩展 24 类分类计划与验证方案

**日期**: 2026-04-14  
**基线**: 61.2% val_acc（5 类，B4.2b）  
**目标**: 24 类 val_acc ≥ 55%，Top-3 accuracy ≥ 75%  
**核心策略**: DXF 图缓存（秒级加载）+ 24 类全量训练 + 长尾过采样

---

## 1. 问题诊断

### 当前瓶颈

| 问题 | 当前状态 | B4.3 方案 |
|------|----------|----------|
| 加载速度 | 每 epoch 7 分钟（3,069 文件实时解析） | 图缓存 → 每 epoch <30s |
| 类别数 | 5 类（目录粗粒度标签） | **24 类**（taxonomy_v2 细粒度） |
| 数据量 | 3,069 样本 | **5,417 样本**（全量） |
| 长尾问题 | 传动件 5.6% → recall=0% | 专项过采样，集中处理 <50 样本类 |
| 训练稳定性 | 进程在 epoch 16 崩溃（内存/解析异常） | 缓存消除解析风险 |

### 24 类数据分布

```
法兰    : 1817 (33.5%)  ██████████████████████
轴类    : 1551 (28.6%)  ███████████████████
箱体    : 1545 (28.5%)  ███████████████████
轴承座  :   74 ( 1.4%)  █
传动件  :   49 ( 0.9%)  ▌
人孔    :   44 ( 0.8%)  ▌
盖罩    :   34 ( 0.6%)  ▌
旋转组件 :   29 ( 0.5%)  ▌
阀门    :   29 ( 0.5%)  ▌
弹簧    :   27 ( 0.5%)  ▌
支架    :   24 ( 0.4%)  ▌
搅拌器  :   24 ( 0.4%)  ▌
... （其余 12 类各 <22 个）
```

前 3 类占 **90.6%**，后 21 类共 504 个样本（平均 24 个/类）

---

## 2. 实现计划

### 2.1 Step 1：DXF 图缓存预处理（新脚本）

**脚本**: `scripts/preprocess_dxf_to_graphs.py`（已实现）

```bash
# 处理全量 5,417 个 DXF → .pt 缓存
python3 scripts/preprocess_dxf_to_graphs.py \
    --manifest data/manifests/unified_manifest_v2.csv \
    --output-dir data/graph_cache \
    --node-dim 19 \
    --edge-dim 7
```

**输出**:
- `data/graph_cache/{md5}.pt`：每个 DXF 对应一个图张量文件
- `data/graph_cache/cache_manifest.csv`：缓存路径索引

**性能预期**:
- 首次运行：~90 分钟（CPU ezdxf 解析 5,417 文件）
- 后续训练：每 epoch <30 秒（纯 tensor 加载）
- 磁盘占用：~54MB（每个 .pt 约 10KB）

### 2.2 Step 2：CachedFinetuneDataset（修改 finetune 脚本）

在 `scripts/finetune_graph2d_from_pretrained.py` 中添加缓存数据集类：

```python
class CachedGraphDataset(Dataset):
    """从 .pt 缓存加载图张量，无需 ezdxf 实时解析。"""

    def __init__(self, cache_manifest_csv: str):
        self.samples = []  # [(cache_path, label_idx), ...]
        self.label_map = {}
        with open(cache_manifest_csv) as f:
            for row in csv.DictReader(f):
                label = row['taxonomy_v2_class']
                if label not in self.label_map:
                    self.label_map[label] = len(self.label_map)
                self.samples.append((row['cache_path'], self.label_map[label]))

    def __getitem__(self, idx):
        cache_path, label_idx = self.samples[idx]
        data = torch.load(cache_path, map_location='cpu')
        return {'x': data['x'], 'edge_index': data['edge_index'],
                'edge_attr': data['edge_attr']}, label_idx
```

### 2.3 Step 3：长尾过采样策略

对样本数 < 50 的类别应用 WeightedRandomSampler：

```python
from collections import Counter
from torch.utils.data import WeightedRandomSampler

class_counts = Counter(label for _, label in dataset.samples)
# 阈值：<50 样本的类别权重提升 min_weight 倍
min_count = 50
weights = []
for _, label in dataset.samples:
    cnt = class_counts[label]
    w = max(1.0, min_count / cnt)  # 最多提升 min_count/cnt 倍
    weights.append(w)

sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
```

### 2.4 Step 4：24 类微调训练

```bash
python3 scripts/finetune_graph2d_from_pretrained.py \
    --pretrained models/graph2d_pretrained_v2_50ep.pth \
    --manifest data/graph_cache/cache_manifest.csv \
    --use-cache \
    --epochs 60 \
    --batch-size 16 \
    --encoder-lr 0.00005 \
    --head-lr 0.0005 \
    --patience 12 \
    --output models/graph2d_finetuned_24class_v1.pth
```

**参数说明**:
- `--use-cache`：使用 CachedGraphDataset
- 更低学习率（微调预训练编码器）
- patience=12（24 类比 5 类更难收敛）

### 2.5 消融实验设计

| 实验 | 数据 | 采样 | Focal | 预期 |
|------|------|------|-------|------|
| A（基线） | 5,417，无缓存 | 自然 | CE | 难运行（太慢） |
| **B（B4.3 主实验）** | **5,417，缓存** | **长尾过采样** | **γ=1.0** | **~55%** |
| C（对比） | 5,417，缓存 | 自然 | CE | ~50% |

---

## 3. 验证标准

### 主要指标

| 指标 | B4.2b 基线（5类） | B4.3 目标（24类） | 验证方法 |
|------|-------------------|-------------------|----------|
| Overall Val Acc | 61.2% | **≥ 55%** | 20% held-out |
| Top-3 Accuracy | N/A | **≥ 75%** | 前3预测含正确标签 |
| Macro F1 | — | **≥ 0.40** | sklearn.metrics |
| 主类(法兰/轴类/箱体) Recall | — | **≥ 70%** | 混淆矩阵 |
| 长尾类(< 50样本) Recall | 0%（传动件） | **≥ 20%** | 至少有识别能力 |
| 每 epoch 训练时间 | 7 分钟 | **< 30 秒** | wall clock |

### 验证流程

每次实验输出：
1. Overall accuracy + Top-3 accuracy + Macro F1
2. Per-class precision / recall / F1（24 类）
3. 混淆矩阵（24×24）热图
4. 训练曲线（loss + acc per epoch）
5. Early stopping epoch

### 回归检查

- [ ] 主类（法兰/轴类/箱体）val_acc ≥ 60%（不能因长尾严重退步）
- [ ] 无类别 recall 为 0%（所有类至少有一次正确预测）
- [ ] 每 epoch 训练时间 < 60s（缓存生效）
- [ ] 模型文件 < 50MB

---

## 4. 时间线

```
Step 1 (~90min, 后台):  DXF → 图缓存（5,417 文件）
Step 2 (~30min):        实现 CachedGraphDataset + --use-cache 参数
Step 3 (~30min):        实现长尾过采样 + Top-3 评估
Step 4 (~60min, 缓存):  实验 B（主实验，缓存 5,417 样本）
Step 5 (~30min):        生成 B4.3 评估报告
```

**总预期**: 首次缓存 90min + 训练 60min = ~2.5 小时

---

## 5. 风险

| 风险 | 概率 | 应对 |
|------|------|------|
| 24 类 val_acc < 50% | 中 | 先合并稀少类（<20样本），退为 15 类 |
| 缓存磁盘不足 | 低 | 5,417 × 10KB ≈ 54MB，远低于阈值 |
| 长尾过采样导致过拟合 | 中 | 配合 Focal Loss + Dropout 0.3 |
| 预训练编码器不兼容 24 类 | 低 | 编码器是通用 GNN，输出 128-dim 与类别数无关 |
| 部分 DXF 缓存失败 | 低 | 记录 failed 列表，跳过异常文件 |

---

## 6. 成功标准

| 阶段 | 目标 | 判定标准 |
|------|------|----------|
| 缓存完成 | 速度验证 | epoch 时间 < 30s（vs 当前 7 分钟） |
| **B4.3 主实验** | **24 类分类** | **val_acc ≥ 55%，Top-3 ≥ 75%** |
| B4.4（后续） | 架构升级 | 同数据 +3-5% |
| B4.5（后续） | Hybrid 集成 | 无文件名 > 60% |

---

## 7. 与全局路线图对应

```
B4.1 充分训练     56.9%  ✓ 完成
B4.2 均衡+增强    61.2%  ✓ 完成（未达 65% 目标但可接受）
B4.3 24类+缓存    55%+   ← 当前阶段
B4.4 架构升级     +3-5%
B4.5 Hybrid集成   无名>60%
```

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
