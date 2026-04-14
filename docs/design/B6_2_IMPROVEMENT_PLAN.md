# B6.2 提升计划：性能优化 + 向 97%+ 迈进 + 长期路线图

**日期**: 2026-04-14  
**基线**: B6.1 — 场景 B=93.9%，全 manifest=95.8%，异构集成上线  
**目标**: 性能验证 + 长期路线图 + 97%+ 探索

---

## 1. 当前状态总结

### 1.1 已达成

| 目标 | 结果 | 状态 |
|------|------|------|
| Graph2D acc ≥ 92% | **91.9%** (v4) | ⚠️ 接近 |
| 场景 B ≥ 93% | **93.9%** (异构集成) | ✓ |
| 综合 avg ≥ 95% | **95.8%** (异构集成) | ✓ |
| P50 < 100ms | **34-53ms** | ✓ |
| 模型体积 < 500KB | **430KB** (INT8) | ✓ |

### 1.2 异构集成架构

```
DXF Input
  ├── GNN v4 (GraphEncoderV2)     weight=0.60  → 91.9% standalone
  ├── StatMLP (59维统计特征)       weight=0.25  → 94.4% standalone
  ├── TextMLP (TF-IDF 500维)      weight=0.15  → 73.7% standalone
  └── Keyword TextContent          priority     → 70.7% (精准但低覆盖)
       ↓
  加权融合 → 95.8% (全manifest) / 93.9% (场景B)
```

---

## 2. B6.2a：异构集成延迟优化

### 2.1 当前延迟估计

```
GNN v4 推理:     ~15ms（INT8 后约 1ms 合成图）
StatMLP 推理:    ~0.1ms（59维 MLP 极快）
TextMLP 推理:    ~0.2ms（500维 MLP 极快）
文字提取:        ~15ms（ezdxf，已通过共享 doc 优化）
统计特征提取:    ~5ms（Python 循环计算度数）

总增量: ~5ms（StatMLP 特征提取）+ ~0.3ms（两个 MLP 推理）
预期 P50: ~40-60ms（已在目标范围内）
```

### 2.2 统计特征提取加速

将 Python 循环计算度数改为向量化操作：

```python
# 当前（慢，O(E) 循环）
for j in range(E):
    src = ei[0, j].item()
    deg[src] += 1

# 优化（快，向量化）
deg = torch.zeros(N).scatter_add_(0, ei[0], torch.ones(E))
```

预期：5ms → 0.5ms

---

## 3. B6.2b：向 97%+ 的路线图

### 3.1 当前误差分析

95.8% 的 4.2% 错误（~192 样本错误）主要来自：
- 三主类互混（法兰↔轴类↔箱体）：~100 样本
- 轴承座假阳性：~50 样本
- 小类混淆：~42 样本

### 3.2 四个提升路径

| 路径 | 预期 | 实施周期 | 依赖 |
|------|------|---------|------|
| **A: 更多训练数据** | +0.5-1pp | 1-3 月 | 数据飞轮积累 |
| **B: GraphEncoderV3** | +1-2pp | 2-4 周 | GPU 环境 |
| **C: 知识蒸馏** | +0.5pp | 1-2 周 | 集成作为 teacher |
| **D: 特征工程** | +0.3-0.5pp | 1 周 | StatMLP 扩展 |

#### A: 更多训练数据（数据飞轮）

```
当前: 4574 原始 + 2186 增强 = 6760
目标: 10000+（通过 low_conf_queue 持续积累）

每 200 条审核样本触发一次 auto_retrain.sh
预期每月积累: 50-200 条（取决于推理量）
```

#### B: GraphEncoderV3（更深 GNN）

```python
# 5层 + JK-Net + Multi-head Attention
# 需 GPU: 预计 2-4 小时训练
# 预期: GNN standalone 93%+ → 集成 97%+
```

#### C: 知识蒸馏（集成 → 单模型）

将 GNN+StatMLP+TextMLP 集成的软标签蒸馏到单个 GNN：

```python
# Teacher: 异构集成（95.8%）
# Student: GraphEncoderV2（单模型）
# 蒸馏 loss = α × CE(student, hard_label) + (1-α) × KL(student, teacher_soft)
# 预期: student 达 93-94%（vs 无蒸馏 91.9%）
```

#### D: StatMLP 特征工程

当前 59 维可扩展到 ~120 维：
- 添加频谱特征（邻接矩阵特征值分布）
- 添加连通分量数
- 添加聚类系数

---

## 4. 验收标准

| 指标 | B6.1 当前 | B6.2 目标 |
|------|----------|---------|
| 集成延迟 P50 | ~50ms | **< 60ms**（含 StatMLP） |
| 场景 B | 93.9% | **维持 ≥ 93%** |
| 全 manifest | 95.8% | **维持 ≥ 95%** |
| 数据飞轮 | auto_retrain.sh | **可执行，精度门控** |

---

## 5. 全系列交付总清单

### 代码文件（17 个新建 + 5 个修改）

| 文件 | 模块 | 阶段 |
|------|------|------|
| `src/ml/text_extractor.py` | DXF 文字提取 | B5.1 |
| `src/ml/text_classifier.py` | 24类关键词+共现+margin | B5.1/5.4/5.6/5.7 |
| `src/ml/monitoring/prediction_monitor.py` | 滑动窗口漂移监控 | B5.3 |
| `src/ml/low_conf_queue.py` | 低置信度主动学习队列 | B5.3 |
| `scripts/augment_dxf_graphs.py` | 图级数据增强 | B5.0 |
| `scripts/finetune_graph2d_v2_augmented.py` | FocalLoss 增量训练 | B5.0 |
| `scripts/quantize_graph2d_model.py` | INT8 动态量化 | B5.2 |
| `scripts/benchmark_inference.py` | 全链路基准测试 | B5.5 |
| `scripts/audit_text_coverage.py` | 关键词审计 | B5.1 |
| `scripts/search_hybrid_weights_v2.py` | 三路权重搜索 | B5.1 |
| `scripts/append_reviewed_to_manifest.py` | 审核样本追加 | B5.4 |
| `scripts/train_stat_mlp.py` | 统计特征 MLP | B6.0 |
| `scripts/train_text_classifier_ml.py` | TF-IDF 文字 MLP | B6.0 |
| `scripts/evaluate_hetero_ensemble.py` | 异构集成评估 | B6.0 |
| `scripts/auto_retrain.sh` | 自动化重训管线 | B6.1 |
| `tests/unit/test_monitoring.py` | 监控测试 28 个 | B5.3 |
| `tests/unit/test_low_conf_queue.py` | 队列测试 26 个 | B5.3 |

### 修改文件

| 文件 | 变更摘要 |
|------|---------|
| `src/ml/hybrid_classifier.py` | TextContent+StatMLP+TF-IDF 推理+监控+单次解析 |
| `src/ml/hybrid_config.py` | TextContentConfig + 三路权重 |
| `src/ml/vision_2d.py` | predict_from_doc + edge_attr 修复 |
| `src/ml/monitoring/__init__.py` | 导出 PredictionMonitor |
| `scripts/preprocess_dxf_to_graphs.py` | 文字缓存 |

### 模型文件（6 个）

| 文件 | 精度 | 大小 | 阶段 |
|------|------|------|------|
| `graph2d_finetuned_24class_v3.pth` | 91.0% | 1623KB | B5.0 |
| `graph2d_finetuned_24class_v3_int8.pth` | ~91.0% | 430KB | B5.2 |
| `graph2d_finetuned_24class_v4.pth` | **91.9%** | 1623KB | B5.7 |
| `graph2d_finetuned_24class_v4_int8.pth` | **~91.9%** | 430KB | B5.8 |
| `stat_mlp_24class.pth` | **94.4%** | ~100KB | B6.0 |
| `text_classifier_tfidf.pth` | **73.7%** | ~500KB | B6.0 |

---

## 6. 里程碑总览

| 阶段 | 关键成果 | 精度 |
|------|---------|------|
| B4.5 | GraphEncoderV2 基线 | 90.5% |
| B5.0 | 数据增强 v3 | 91.0% |
| B5.1-5.4 | 文字融合三路架构 | avg=94.1% |
| B5.5-5.6 | 性能优化 + margin | P50=34ms |
| B5.7-5.8 | v4 定向增强 | avg=94.8% |
| B5.9 | 同构集成无效确认 | 94.0% |
| **B6.0** | **异构集成突破** | **95.8%** |
| **B6.1** | **场景 B=93.9%** | **avg≈95.4%** |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
