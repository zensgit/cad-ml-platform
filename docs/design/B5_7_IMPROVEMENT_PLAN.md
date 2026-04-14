# B5.7 提升计划：Graph2D 模型进化 + 主动学习闭环 + avg ≥ 95%

**日期**: 2026-04-14  
**基线**: B5.6 — avg=94.1%，精度 68.2%，P50=34ms，单次解析优化  
**目标**: 综合 avg ≥ 95%（需场景 B ≥ 93%），v4 模型部署

---

## 1. 差距分析

### 1.1 精度分解

```
当前: avg = 0.25×100% + 0.50×91.0% + 0.15×91.0% + 0.10×100% = 94.15%
目标: avg = 95.0%
差距: +0.85pp

场景B需提升至:
  0.25×100% + 0.50×B + 0.15×91% + 0.10×100% = 95%
  0.50×B = 95% - 25% - 13.65% - 10% = 46.35%
  B = 92.7%（即场景B需从91.0%提升至≥92.7%）
```

### 1.2 场景B错误分析（91.0%，82/914错误）

场景B = 无文件名 + 有文字 = Graph2D + TextContent。当前 91.0% 已等于 Graph2D 基线（margin 放弃策略消除了文字负面影响）。

**提升路径**：
1. **提升 Graph2D 基线** → 直接提升场景 B/C
2. **提升文字命中精度** → 仅提升场景 B（文字命中时才有效）
3. **组合** → 最佳效果

### 1.3 Graph2D 91.0% 的错误来源

基于 B5.0 混淆矩阵分析：
- 轴承座 recall=100% 但 precision=43%（大量箱体/法兰被误判为轴承座）
- 部分小类样本量仍偏少（泵 5 样本、人孔 6 样本）
- 主类内部混淆（法兰↔轴类：圆形几何特征相似）

---

## 2. B5.7a：Graph2D v4 增量训练

### 2.1 训练数据策略

| 数据源 | 作用 | 优先级 |
|--------|------|--------|
| 原始 manifest v2（4,574） | 基础分布 | P0 |
| 增强样本（B5.0，+1,430） | 弱类平衡 | P0 |
| **错误样本分析**（~82 错误） | 定向修复 | P1 |
| 审核队列样本（积累中） | 主动学习 | P2 |

### 2.2 错误样本定向增强

```bash
# 步骤1：分析 v3 在 val set 上的错误分布
python3 scripts/evaluate_graph2d_v2.py \
    --model models/graph2d_finetuned_24class_v3.pth \
    --manifest data/graph_cache/cache_manifest.csv \
    --show-confusion

# 步骤2：对高错误率的类-对做针对性增强
# 例如：轴承座 FP 来源 → 增加更多箱体/法兰对比样本
python3 scripts/augment_dxf_graphs.py \
    --manifest data/manifests/unified_manifest_v2.csv \
    --classes 箱体,法兰 \
    --target-count 300 \
    --output-dir data/graph_cache_v4_aug/

# 步骤3：合并 manifest
# original + aug_v1 + targeted_aug → cache_manifest_v4.csv
```

### 2.3 训练策略

**关键调整（相比 B5.0）**：

| 参数 | B5.0 | B5.7 v4 | 原因 |
|------|------|---------|------|
| encoder-lr | 5e-5 | **2e-5** | 更保守，避免过拟合新增小样本 |
| head-lr | 5e-4 | **2e-4** | 同上 |
| focal-gamma | 1.5 | **2.0** | 更强聚焦难分类样本 |
| patience | 12 | **15** | 给更多探索空间 |
| epochs | 60 | **50** | 低 lr 下 plateau 更快 |

```bash
python scripts/finetune_graph2d_v2_augmented.py \
    --checkpoint models/graph2d_finetuned_24class_v3.pth \
    --manifest data/graph_cache_v4_aug/cache_manifest_v4.csv \
    --output models/graph2d_finetuned_24class_v4.pth \
    --epochs 50 --batch-size 32 \
    --encoder-lr 2e-5 --head-lr 2e-4 \
    --focal-gamma 2.0 --patience 15 --device cpu
```

### 2.4 v4 验收标准

| 指标 | v3 基线 | v4 目标 |
|------|---------|---------|
| Overall acc | 91.0% | **≥ 92.0%** |
| 轴承座 precision | 43% | **≥ 60%** |
| 轴承座 recall | 100% | **≥ 90%** |
| 法兰 recall | 100% | **≥ 95%** |
| 阀门 recall | 100% | **≥ 90%** |
| Top-3 acc | 99.1% | **≥ 99%** |

---

## 3. B5.7b：TextContent 条件关键词（共现匹配）

### 3.1 方案

当前关键词是独立匹配。条件关键词要求多个词同时出现才算命中：

```python
# 法兰条件关键词（2/3 共现才命中）
"法兰_条件组": {
    "requires_any_n": 2,
    "keywords": ["密封面粗糙度", "RF面", "螺栓孔圆", "突面", "PN"],
}
```

**实施位置**：`TextContentClassifier.predict_probs()` 中，在 softmax 之前增加条件分数计算。

### 3.2 预期效果

条件共现能将法兰的精度从 100%（命中率 5%）提升到更高命中率（当"密封面粗糙度"+"PN"同时出现时，几乎确定是法兰）。

### 3.3 风险

- 增加代码复杂度
- 需要足够的 DXF 文字样本验证条件词组合
- 不确定命中率提升是否显著（法兰 DXF 可能根本不含这些词）

**建议**：先收集更多法兰 DXF 文字样本，分析实际内容后再决定是否实施条件关键词。

---

## 4. B5.7c：模型集成（Ensemble）

### 4.1 方案

使用 v2 + v3 + v4 三模型软投票集成：

```python
# 已有 EnsembleGraph2DClassifier
# 将 v2, v3, v4 三个 checkpoint 作为集成成员
export GRAPH2D_ENSEMBLE_ENABLED=true
export GRAPH2D_ENSEMBLE_MODELS="models/graph2d_v2.pth,models/graph2d_v3.pth,models/graph2d_v4.pth"
```

**预期效果**：
- 集成通常比单模型提升 0.5-1.5pp
- v2 和 v3 已有不同训练策略（v2 无增强，v3 有增强），互补性好
- 延迟增加：约 ×3（但 P50=34ms×3 ≈ 102ms，仍接近目标）

### 4.2 替代：知识蒸馏

若集成延迟不可接受，可将集成作为 teacher 蒸馏到单模型 student：

```bash
# 教师：v2+v3+v4 集成
# 学生：新 GraphEncoderV2（从头训练，用教师 soft label）
python scripts/distill_graph2d.py \
    --teacher-models models/graph2d_v2.pth,models/graph2d_v3.pth,models/graph2d_v4.pth \
    --manifest data/graph_cache/cache_manifest.csv \
    --output models/graph2d_distilled_v5.pth \
    --temperature 3.0 --alpha 0.7
```

---

## 5. 验收标准（B5.7 总体）

| 指标 | 当前 | 目标 |
|------|------|------|
| Graph2D val acc | 91.0% | **≥ 92.0%** |
| 场景 B（无名有文字） | 91.0% | **≥ 92.7%** |
| 综合 avg | 94.1% | **≥ 95.0%** |
| 文字精度 | 68.2% | **≥ 70%** |
| P50 推理延迟 | 34ms | **< 100ms** ✓ |
| 轴承座 precision | 43% | **≥ 60%** |

---

## 6. 实施步骤

```
Week 1: 错误分析 + 定向增强
  → 分析 v3 混淆矩阵（识别 top-5 错误类对）
  → 对高错误率类-对做针对性数据增强
  → 合并 manifest v4

Week 2: v4 增量训练
  → fine-tune v3 → v4（50 epochs, encoder-lr=2e-5）
  → 评估 v4 精度（目标 ≥ 92%）
  → 量化 v4 → v4_int8

Week 3: 综合评估
  → 四场景权重搜索（含 v4）
  → 全链路基准测试
  → 文字审计（验证文字精度维持 ≥ 68%）

Week 4: 可选——集成/蒸馏
  → 若 v4 单模型未达 92%，考虑 v2+v3+v4 集成
  → 若集成达标但延迟过高，蒸馏到 v5
  → 生产部署
```

---

## 7. B5 全系列成就回顾

```
B5.0  → 数据增强 91.0%（轴承座/阀门 recall 100%）
B5.1  → 文字融合评估 avg=94.1%（三路权重搜索 64 组合）
B5.2  → INT8 量化 -74% 体积 + 文字缓存单次 I/O
B5.3  → PredictionMonitor + LowConfidenceQueue（54/54 测试）
B5.4  → TextContent 正式推理集成 + 关键词扩充 +28.4pp 精度
B5.5  → 综合验收 + qnnpack 修复 + 全链路测量
B5.6  → 单次解析 P50=34ms + Margin 放弃 精度 68.2%
B5.7  → v4 增量训练（待实施）→ 目标 avg ≥ 95%
```

**从 B5.0 到 B5.6 的累计变化**：
- Graph2D 精度：91.0%（维持，模型未变）
- 文字分类器精度：0% → **68.2%**（从无到有 + 词典扩充 + margin 放弃）
- 推理延迟：估计 ~133ms → **P50=34ms, P95=118ms**（单次解析 + 优化）
- 模型体积：1623KB → **430KB**（-74% INT8 量化）
- 监控：**全套**（PredictionMonitor + LowConfidenceQueue + 自动告警）

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
