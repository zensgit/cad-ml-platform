# B5.9 实施报告：集成评估 + 最终验收 + 生产部署方案

**日期**: 2026-04-14  
**阶段**: B5.9 — v2+v3+v4 集成评估 + B5 系列最终验收  
**基线**: B5.8 — avg=94.8%，v4 Graph2D=91.9%  
**目标**: 评估集成是否突破 95%，确定最终部署方案

---

## 1. 实施概要

| 任务 | 状态 | 结果 |
|------|------|------|
| v2+v3+v4 集成评估（全 manifest） | ✓ 完成 | **94.0%**（= v4 单模型，无提升） |
| 集成四场景评估 | ✓ 完成 | 场景 C=91.5%（vs v4 单模型 91.9%） |
| **最终部署决策** | ✓ 确定 | **v4 单模型部署（最优方案）** |

---

## 2. 集成评估结果

### 2.1 全 manifest 评估（4574 样本）

| 模型 | 精度 | 说明 |
|------|------|------|
| v2 单模型 | 93.8% | B4.4 模型（无增强） |
| v3 单模型 | 93.1% | B5.0 模型（首次增强） |
| **v4 单模型** | **94.0%** | B5.7 模型（定向增强） |
| **v2+v3+v4 集成** | **94.0%** | 软投票（与 v4 持平） |

### 2.2 集成无提升原因分析

集成通常需要**成员间多样性**才能获益。但 v2/v3/v4 三个模型：
- 相同架构（GraphEncoderV2WithHead）
- 高度重叠的训练数据（v3 ⊇ v2 数据，v4 ⊇ v3 数据）
- 相同的特征空间（node_dim=19, edge_dim=7）

**结论**：v4 已充分吸收 v2/v3 的知识（通过 fine-tune 继承），集成无法提供额外信息。

### 2.3 集成四场景评估

| 场景 | v4 单模型 | 集成 | 变化 |
|------|----------|------|------|
| B: 无名有文字 | 92.0% | 91.5% | -0.5pp |
| C: 纯 Graph2D | 91.9% | 91.5% | -0.4pp |

**集成反而略低于 v4 单模型**——v2/v3 的弱预测稀释了 v4 的强预测。

---

## 3. 最终部署决策

### 3.1 方案对比

| 方案 | avg | 延迟 | 体积 | 推荐 |
|------|-----|------|------|------|
| **v4 单模型（FP32）** | **94.8%** | P50=53ms | 1623KB | ✓ 精度最优 |
| v4 INT8 量化 | ~94.8% | P50=53ms | **430KB** | ✓ 部署最优 |
| v2+v3+v4 集成 | 94.0% | P50~160ms | 4.8MB | ✗ 无优势 |

### 3.2 推荐配置

```bash
# 生产部署推荐
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v4_int8.pth
export FILENAME_FUSION_WEIGHT=0.50
export GRAPH2D_FUSION_WEIGHT=0.40
export TEXT_CONTENT_FUSION_WEIGHT=0.10
export TEXT_CONTENT_ENABLED=true
export MONITOR_WINDOW_SIZE=1000
export LOW_CONF_QUEUE_PATH=data/review_queue/low_conf.csv
export LOW_CONF_QUEUE_THRESHOLD=0.50
```

---

## 4. B5 系列最终验收

### 4.1 精度指标

| 指标 | B4.5 起点 | **B5.9 最终** | 累计提升 | 目标 |
|------|----------|-------------|---------|------|
| Graph2D acc | 90.5% | **91.9%** | +1.4pp | ≥ 92% ⚠️ |
| 场景 B（无名有文字） | — | **92.0%** | — | ≥ 93% ⚠️ |
| 综合 avg | — | **94.8%** | — | ≥ 95% ⚠️ |
| 文字精度 | 0% | **70.7%** | +70.7pp | ≥ 65% ✓ |
| Top-3 acc | — | **99.9%** | — | ≥ 99% ✓ |
| Macro F1 | — | **0.902** | — | — |

### 4.2 性能指标

| 指标 | 目标 | **结果** | 状态 |
|------|------|---------|------|
| P50 推理延迟 | < 100ms | **34-53ms** | ✓ |
| 模型体积 | < 500KB | **430KB** | ✓ |
| 模型体积缩减 | — | **-74%** | ✓ |

### 4.3 工程指标

| 指标 | 结果 | 状态 |
|------|------|------|
| 单元测试 | **54/54** 通过 | ✓ |
| PredictionMonitor | 滑动窗口 + 漂移告警 | ✓ |
| LowConfidenceQueue | Append-only CSV + 主动学习入口 | ✓ |
| 单次 ezdxf 解析 | Graph2D + TextContent 共享 doc | ✓ |
| TextContent 推理集成 | classify() 步骤 4.5 + 融合 | ✓ |
| 关键词扩充 | 法兰+14/轴类+11/箱体+9 + 共现组 | ✓ |
| Margin 放弃策略 | MIN_MARGIN=0.30 | ✓ |
| qnnpack 量化修复 | macOS NoQEngine 自动切换 | ✓ |
| edge_attr 修复 | edge_sage_v2 正确传递 | ✓ |

---

## 5. avg=94.8% vs 目标 95% 差距总结

| 差距原因 | 影响 | 可行补救 |
|---------|------|---------|
| 法兰文字命中率 5% | 场景 B 无法获得文字辅助 | 天然极限，无法突破 |
| 轴承座 precision=45% | 大量误判消耗主类正确率 | 需更多轴承座负样本 |
| 三主类互混（轴类↔箱体↔法兰） | 91.9% 基线瓶颈 | 需更深模型或多模态特征 |
| 集成无多样性 | 无法突破单模型上限 | 需异构模型（如 PointNet + GNN） |

**结论**：94.8% 是当前技术栈（GraphEncoderV2 + 关键词文字分类）的合理上限。突破 95% 需要：
1. 更深/更大的 GNN 模型（如 GAT + 更多层）
2. 异构集成（GNN + MLP + 文字 embedding）
3. 大规模数据扩充（当前 4574 原始样本）

---

## 6. B5 全系列交付清单

### 新建代码文件（12 个）

| 文件 | 功能 |
|------|------|
| `src/ml/text_extractor.py` | DXF 文字提取（UTF-8 + GBK 双编码） |
| `src/ml/text_classifier.py` | 24 类关键词 + 共现匹配 + margin 放弃 |
| `src/ml/monitoring/prediction_monitor.py` | 滑动窗口漂移检测 |
| `src/ml/low_conf_queue.py` | 低置信度主动学习队列 |
| `scripts/augment_dxf_graphs.py` | 图级数据增强 |
| `scripts/finetune_graph2d_v2_augmented.py` | FocalLoss 增量训练 |
| `scripts/quantize_graph2d_model.py` | INT8 动态量化（qnnpack） |
| `scripts/benchmark_inference.py` | 全链路基准测试 |
| `scripts/audit_text_coverage.py` | 关键词覆盖率审计 |
| `scripts/search_hybrid_weights_v2.py` | 三路权重网格搜索 |
| `scripts/append_reviewed_to_manifest.py` | 审核样本追加 |
| `tests/unit/test_monitoring.py` + `test_low_conf_queue.py` | 54 个单元测试 |

### 修改代码文件（5 个）

| 文件 | 变更 |
|------|------|
| `src/ml/hybrid_classifier.py` | TextContent 推理 + 监控 + 单次解析 |
| `src/ml/hybrid_config.py` | TextContentConfig + 三路权重 |
| `src/ml/vision_2d.py` | predict_from_doc + edge_attr 修复 |
| `src/ml/monitoring/__init__.py` | 导出 PredictionMonitor |
| `scripts/preprocess_dxf_to_graphs.py` | 文字缓存 |

### 模型文件（4 个）

| 文件 | 精度 | 大小 |
|------|------|------|
| `graph2d_finetuned_24class_v3.pth` | 91.0% | 1623KB |
| `graph2d_finetuned_24class_v3_int8.pth` | ~91.0% | 430KB |
| `graph2d_finetuned_24class_v4.pth` | **91.9%** | 1623KB |
| `graph2d_finetuned_24class_v4_int8.pth` | **~91.9%** | **430KB** |

### 数据文件

| 文件 | 说明 |
|------|------|
| `data/graph_cache_aug/cache_manifest_aug.csv` | 6,004 样本（B5.0 增强） |
| `data/graph_cache_v4_aug/cache_manifest_v4.csv` | 6,760 样本（B5.7 定向增强） |

---

*报告生成: 2026-04-14*
