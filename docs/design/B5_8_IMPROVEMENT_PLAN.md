# B5.8 提升计划：v4 评估 + 模型集成 + 生产部署

**日期**: 2026-04-14  
**基线**: B5.7 — v4 训练中，共现关键词完成，定向增强 6760 样本  
**目标**: 生产部署 v4（或集成），综合 avg ≥ 95%，P50 < 100ms

---

## 1. v4 训练完成后执行步骤

### 1.1 v4 精度评估

```bash
# 步骤 1：在原始 manifest 上评估（公平对比 v3）
python scripts/evaluate_graph2d_v2.py \
    --model models/graph2d_finetuned_24class_v4.pth \
    --manifest data/graph_cache/cache_manifest.csv \
    --show-confusion

# 步骤 2：四场景权重搜索（含共现关键词）
python scripts/search_hybrid_weights_v2.py \
    --model models/graph2d_finetuned_24class_v4.pth \
    --manifest data/graph_cache/cache_manifest.csv \
    --output docs/design/B5_8_WEIGHT_SEARCH.md

# 步骤 3：文字审计（共现关键词效果）
python scripts/audit_text_coverage.py \
    --manifest data/manifests/unified_manifest_v2.csv \
    --sample 20 \
    --output docs/design/B5_8_TEXT_AUDIT.md
```

### 1.2 v4 vs v3 对比决策

```
v4 val acc ≥ 92% AND 轴承座 prec ≥ 65%?
  ├── 是 → 部署 v4（量化后 v4_int8）
  └── 否 →
        v4 val acc ≥ 91%?
          ├── 是 → v2+v3+v4 集成（EnsembleGraph2DClassifier）
          └── 否 → 调参重训（提高 focal-gamma, 增加数据量）
```

---

## 2. B5.8a：INT8 量化 v4

```bash
# 量化
python scripts/quantize_graph2d_model.py \
    --model models/graph2d_finetuned_24class_v4.pth \
    --output models/graph2d_finetuned_24class_v4_int8.pth \
    --benchmark \
    --verify-manifest data/graph_cache/cache_manifest.csv \
    --verify-limit 500
```

### 验收标准

| 指标 | 目标 |
|------|------|
| 精度损失 | < 0.5 pp |
| 模型大小 | < 500 KB |
| GNN 推理延迟 | < 2ms (小图) |

---

## 3. B5.8b：模型集成（备选方案）

若 v4 单模型未达 92%，使用 v2+v3+v4 三模型软投票：

```bash
# 启用集成
export GRAPH2D_ENSEMBLE_ENABLED=true
export GRAPH2D_ENSEMBLE_MODELS="models/graph2d_finetuned_24class_v2.pth,models/graph2d_finetuned_24class_v3.pth,models/graph2d_finetuned_24class_v4.pth"
```

预期效果：
- 集成比单模型通常提升 0.5-1.5pp
- 延迟增加 ×3（P50 ≈ 34×3 ≈ 102ms，接近 100ms 目标）
- 可接受：若 v4=91.5%，集成可能达 92-93%

---

## 4. B5.8c：生产部署配置

### 4.1 环境变量（推荐）

```bash
# 模型选择（v4 单模型或集成）
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v4_int8.pth

# 融合权重（B5.1 搜索最优，持续验证）
export FILENAME_FUSION_WEIGHT=0.45
export GRAPH2D_FUSION_WEIGHT=0.35
export TEXT_CONTENT_FUSION_WEIGHT=0.10
export TEXT_CONTENT_ENABLED=true

# 监控
export MONITOR_WINDOW_SIZE=1000
export LOW_CONF_QUEUE_PATH=data/review_queue/low_conf.csv
export LOW_CONF_QUEUE_THRESHOLD=0.50
```

### 4.2 部署验证清单

| 检查项 | 命令 | 预期 |
|--------|------|------|
| 模型加载 | `python -c "from src.ml.vision_2d import get_2d_classifier; print(get_2d_classifier()._loaded)"` | True |
| 推理延迟 | `python scripts/benchmark_inference.py --n-files 50` | P50 < 100ms |
| 监控初始化 | `clf.monitor.n > 0` after 10 requests | True |
| 低置信度队列 | `clf.low_conf_queue.size()` | 0 (初始) |
| 文字命中率 | `clf.monitor.text_hit_rate` after 100 requests | > 0.05 |
| 漂移告警 | `clf.monitor.check_drift()` | False (正常) |

### 4.3 回滚方案

```bash
# 如果 v4 出现问题，切回 v3
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v3.pth

# 如果文字融合出问题，临时禁用
export TEXT_CONTENT_ENABLED=false
```

---

## 5. B5 全系列最终成就清单

### 代码交付物

| 模块 | 文件 | 功能 |
|------|------|------|
| 数据增强 | `scripts/augment_dxf_graphs.py` | 图级增强（边/节点/特征噪声） |
| 增量训练 | `scripts/finetune_graph2d_v2_augmented.py` | FocalLoss + WeightedSampler 微调 |
| 文字提取 | `src/ml/text_extractor.py` | DXF 文字提取（UTF-8 + GBK 双编码） |
| 文字分类 | `src/ml/text_classifier.py` | 24类关键词 + 共现匹配 + margin放弃 |
| 配置层 | `src/ml/hybrid_config.py` | TextContentConfig + 三路权重 |
| 推理集成 | `src/ml/hybrid_classifier.py` | 单次解析共享 + 文字融合路径 |
| 监控 | `src/ml/monitoring/prediction_monitor.py` | 滑动窗口漂移告警 |
| 队列 | `src/ml/low_conf_queue.py` | 低置信度主动学习入口 |
| 预处理 | `scripts/preprocess_dxf_to_graphs.py` | .pt 缓存含文字字段 |
| 量化 | `scripts/quantize_graph2d_model.py` | INT8 动态量化 + qnnpack |
| 基准测试 | `scripts/benchmark_inference.py` | 全链路延迟测量 |
| 审计 | `scripts/audit_text_coverage.py` | 关键词覆盖率审计 |
| 权重搜索 | `scripts/search_hybrid_weights_v2.py` | 三路64组合网格搜索 |
| 主动学习 | `scripts/append_reviewed_to_manifest.py` | 审核队列→训练manifest |

### 模型交付物

| 模型 | 文件 | 精度 | 大小 |
|------|------|------|------|
| v2 (B4.4) | `graph2d_finetuned_24class_v2.pth` | 90.5% | ~1.6MB |
| v3 (B5.0) | `graph2d_finetuned_24class_v3.pth` | 91.0% | 1623KB |
| v3 INT8 | `graph2d_finetuned_24class_v3_int8.pth` | ~91.0% | **430KB** |
| **v4 (B5.7)** | `graph2d_finetuned_24class_v4.pth` | **训练中** | ~1.6MB |

### 指标演进

| 阶段 | Graph2D | 文字精度 | avg | P50 |
|------|---------|---------|------|------|
| B4.5 | 90.5% | — | — | — |
| B5.0 | **91.0%** | — | — | — |
| B5.1 | 91.0% | 39.8% | 94.1% | — |
| B5.4 | 91.0% | 57.8% | 94.1% | — |
| B5.6 | 91.0% | **68.2%** | 94.1% | **34ms** |
| B5.7 | **训练中** | 68.2%+ | — | 34ms |

---

## 6. 里程碑追踪

| 里程碑 | 内容 | 结果 | 状态 |
|--------|------|------|------|
| B5.0 | 数据增强 + v3 | 91.0% | ✓ |
| B5.1 | 三路融合评估 | avg=94.1% | ✓ |
| B5.2 | 缓存+量化 | -74% 体积 | ✓ |
| B5.3 | 监控上线 | 54/54 测试 | ✓ |
| B5.4 | TextContent推理 | 精度+28pp | ✓ |
| B5.5 | 综合验收 | 94.1% avg | ✓ |
| B5.6 | 单次解析+margin | P50=34ms | ✓ |
| **B5.7** | **v4训练+共现** | **训练中** | **⏳** |
| B5.8a | v4 量化 | 待v4完成 | 待实施 |
| B5.8b | 集成（备选） | 待评估 | 待决策 |
| B5.8c | 生产部署 | — | 待实施 |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
