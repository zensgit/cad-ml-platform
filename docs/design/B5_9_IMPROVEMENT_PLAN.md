# B5.9 提升计划：突破 95% + 生产部署 + 持续学习闭环

**日期**: 2026-04-14  
**基线**: B5.8 — avg=94.8%，Graph2D v4=91.9%，精度 70.7%，P50=53ms  
**目标**: avg ≥ 95.0%（仅需场景 B +0.4pp），生产部署，持续学习

---

## 1. 差距分析

```
当前: avg = 0.25×100% + 0.50×92.0% + 0.15×91.9% + 0.10×100% = 94.8%
需要: B ≥ 92.4%（场景B仅需 +0.4pp）
```

### 1.1 三个突破路径

| 路径 | 预期收益 | 实施难度 | 推荐 |
|------|---------|---------|------|
| **v2+v3+v4 集成** | +0.5-1.5pp | 低（代码已有） | ✓ 首选 |
| 更多主动学习数据 + v5 训练 | +0.3-1.0pp | 高（需数据积累） | 中期 |
| 文字命中率突破（法兰 5%→20%） | +0.1-0.3pp | 中 | 辅助 |

---

## 2. B5.9a：模型集成（最快路径）

### 2.1 实施

```bash
# 启用三模型软投票集成
export GRAPH2D_ENSEMBLE_ENABLED=true
export GRAPH2D_ENSEMBLE_MODELS="models/graph2d_finetuned_24class_v2.pth,models/graph2d_finetuned_24class_v3.pth,models/graph2d_finetuned_24class_v4.pth"

# 验证集成精度
python scripts/evaluate_graph2d_v2.py \
    --model models/graph2d_finetuned_24class_v4.pth \
    --manifest data/graph_cache/cache_manifest.csv \
    --ensemble
```

### 2.2 预期

- v2(90.5%) + v3(91.0%) + v4(91.9%) 软投票 → 预期 **92.0-93.0%**
- 延迟：×3 → P50 ≈ 53×3 ≈ 160ms（仍可接受）
- 若达到 92.5%：avg = 25 + 46.25 + 13.88 + 10 = **95.1%** ✓

### 2.3 风险缓解

若集成延迟不可接受（>200ms P50），可用知识蒸馏将集成压缩回单模型。

---

## 3. B5.9b：生产部署

### 3.1 部署配置

```bash
# === v4 单模型部署（推荐，P50=53ms）===
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v4_int8.pth
export FILENAME_FUSION_WEIGHT=0.50
export GRAPH2D_FUSION_WEIGHT=0.40
export TEXT_CONTENT_FUSION_WEIGHT=0.10
export TEXT_CONTENT_ENABLED=true
export MONITOR_WINDOW_SIZE=1000
export LOW_CONF_QUEUE_PATH=data/review_queue/low_conf.csv
export LOW_CONF_QUEUE_THRESHOLD=0.50

# === 或集成部署（精度更高但延迟更长）===
# export GRAPH2D_ENSEMBLE_ENABLED=true
# export GRAPH2D_ENSEMBLE_MODELS=models/graph2d_v2.pth,models/graph2d_v3.pth,models/graph2d_v4.pth
```

### 3.2 上线验证清单

```
□ 模型加载成功（_loaded=True）
□ 100 次推理后 monitor.text_hit_rate > 0.05
□ 100 次推理后 monitor.avg_confidence > 0.5
□ monitor.check_drift() = False（无漂移）
□ low_conf_queue.size() 合理增长
□ P50 < 100ms（benchmark 确认）
□ 回滚测试：切回 v3 模型正常工作
```

### 3.3 回滚方案

```bash
# 紧急回滚到 v3
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v3.pth
export FILENAME_FUSION_WEIGHT=0.45
export GRAPH2D_FUSION_WEIGHT=0.35

# 禁用文字融合
export TEXT_CONTENT_ENABLED=false
```

---

## 4. B5.9c：持续学习闭环

### 4.1 闭环流程

```
生产推理
  → PredictionMonitor.record()
  → low_conf_queue.maybe_enqueue()（confidence < 0.50）
  → data/review_queue/low_conf.csv

人工审核（每周/每月）
  → 填写 reviewed_label 列
  → python scripts/append_reviewed_to_manifest.py

增量训练（积累 200+ 条后触发）
  → python scripts/finetune_graph2d_v2_augmented.py
  → 评估 → 量化 → 部署新版本

监控
  → drift_detected=True 时触发审核加速
  → low_conf_rate > 15% 时考虑紧急重训
```

### 4.2 自动化建议

| 阶段 | 手动 | 可自动化 |
|------|------|---------|
| 推理 + 入队 | — | ✓ 已自动化 |
| 审核 | 人工 | 部分（高置信度纠正可自动确认） |
| append_to_manifest | 手动触发 | ✓ 可 cron |
| 训练 | 手动触发 | ✓ 可 CI/CD |
| 评估 | 手动审核 | ✓ 可自动化（精度门槛） |
| 部署 | 手动切换 | 谨慎（建议人工确认） |

---

## 5. B5 全系列总结

### 5.1 B5 全部里程碑

| 里程碑 | 内容 | 结果 | 状态 |
|--------|------|------|------|
| B5.0 | 数据增强 + v3 | 91.0% | ✓ |
| B5.1 | 三路融合评估 | avg=94.1% | ✓ |
| B5.2 | 缓存+量化 | -74% 体积 | ✓ |
| B5.3 | 监控上线 | 54/54 测试 | ✓ |
| B5.4 | TextContent推理+扩充 | 精度+28pp | ✓ |
| B5.5 | 综合验收 | avg=94.1% | ✓ |
| B5.6 | 单次解析+margin | P50=34ms | ✓ |
| B5.7 | v4训练+共现关键词 | 91.9% | ✓ |
| **B5.8** | **v4量化+配置更新** | **avg=94.8%** | **✓** |
| B5.9a | 集成（备选） | — | 待需要 |
| B5.9b | 生产部署 | — | 待实施 |
| B5.9c | 持续学习闭环 | — | 框架已就绪 |

### 5.2 B5 全部交付清单

**新建代码文件（12 个）**：
- `src/ml/text_extractor.py` — DXF 文字提取（UTF-8 + GBK）
- `src/ml/text_classifier.py` — 24 类关键词 + 共现 + margin 放弃
- `src/ml/monitoring/prediction_monitor.py` — 滑动窗口漂移检测
- `src/ml/low_conf_queue.py` — 低置信度主动学习队列
- `scripts/augment_dxf_graphs.py` — 图级数据增强
- `scripts/finetune_graph2d_v2_augmented.py` — FocalLoss 增量训练
- `scripts/quantize_graph2d_model.py` — INT8 动态量化
- `scripts/benchmark_inference.py` — 全链路基准测试
- `scripts/audit_text_coverage.py` — 关键词覆盖率审计
- `scripts/search_hybrid_weights_v2.py` — 三路权重网格搜索
- `scripts/append_reviewed_to_manifest.py` — 审核样本追加工具
- `tests/unit/test_monitoring.py` + `tests/unit/test_low_conf_queue.py` — 54 个测试

**修改代码文件（4 个）**：
- `src/ml/hybrid_classifier.py` — TextContent 推理集成 + 监控 + 单次解析
- `src/ml/hybrid_config.py` — TextContentConfig + 三路权重
- `src/ml/vision_2d.py` — predict_from_doc + edge_attr 修复
- `src/ml/monitoring/__init__.py` — 导出 PredictionMonitor
- `scripts/preprocess_dxf_to_graphs.py` — 文字缓存

**模型文件（4 个）**：
- `models/graph2d_finetuned_24class_v3.pth` (91.0%, 1623KB)
- `models/graph2d_finetuned_24class_v3_int8.pth` (430KB)
- `models/graph2d_finetuned_24class_v4.pth` (**91.9%**, 1623KB)
- `models/graph2d_finetuned_24class_v4_int8.pth` (**430KB**)

**文档（18 个 MD）**：
- B5.0~B5.8 训练报告（9 个）
- B5.1~B5.9 提升计划（9 个）
- 审计/权重搜索报告（自动生成，6 个）

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
