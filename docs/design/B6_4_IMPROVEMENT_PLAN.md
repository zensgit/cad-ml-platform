# B6.4 提升计划：生产固化 + 持续优化 + 长期架构演进

**日期**: 2026-04-14  
**基线**: B6.3 — 异构集成 95.8%，蒸馏/StatMLP v2 训练中  
**目标**: 生产固化、持续学习闭环运营、97%+ 路线规划

---

## 1. 生产固化清单

### 1.1 部署配置（三种模式）

**模式 A — 最高精度（推荐）**
```bash
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v4_int8.pth
export STAT_MLP_ENABLED=true
export STAT_MLP_MODEL_PATH=models/stat_mlp_v2_24class.pth  # 或 stat_mlp_24class.pth
export TFIDF_TEXT_ENABLED=true
export TFIDF_TEXT_MODEL_PATH=models/text_classifier_tfidf.pth
export TEXT_CONTENT_ENABLED=true
export FILENAME_FUSION_WEIGHT=0.50
export GRAPH2D_FUSION_WEIGHT=0.40
export TEXT_CONTENT_FUSION_WEIGHT=0.10
# 预期精度：~95.8%  延迟：~50ms
```

**模式 B — 低延迟**
```bash
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v4_int8.pth
export STAT_MLP_ENABLED=false
export TFIDF_TEXT_ENABLED=false
export TEXT_CONTENT_ENABLED=true
# 预期精度：~94.8% avg  延迟：~34ms
```

**模式 C — 极简**
```bash
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v4_int8.pth
export TEXT_CONTENT_ENABLED=false
# 预期精度：~91.9%  延迟：~15ms  体积：430KB
```

### 1.2 监控运维

```bash
# 每日检查
python3 -c "
from src.ml.hybrid_classifier import HybridClassifier
clf = HybridClassifier()
s = clf.monitor.summary()
print(f'n={s[\"n\"]}  conf={s[\"avg_confidence\"]:.3f}  drift={s[\"drift_detected\"]}')
print(f'queue: {clf.low_conf_queue.size()} total, {clf.low_conf_queue.pending_review()} pending')
"

# 每月重训（数据飞轮）
bash scripts/auto_retrain.sh
```

---

## 2. 持续优化路线

### 2.1 短期（1-4 周）

| 任务 | 预期收益 | 依赖 |
|------|---------|------|
| v5b 蒸馏结果评估 | 单模型 ≥ 93% | 训练完成 |
| StatMLP v2 评估 | 集成 ≥ 96% | 训练完成 |
| 新异构集成评估（GNN+StatV2+Text） | ≥ 96% | StatMLP v2 |
| 蒸馏 v5b 量化 | 430KB 单模型 | v5b 完成 |

### 2.2 中期（1-3 月）

| 任务 | 预期收益 | 依赖 |
|------|---------|------|
| 数据飞轮积累 200+ 审核样本 | v6 模型 +0.5pp | 生产运行 |
| GraphEncoderV3（5 层 JK-Net） | GNN 93%+ | GPU |
| 半监督自训练 | +0.5-1pp | 未标注 DXF 数据 |

### 2.3 长期（3-6 月）

| 任务 | 预期收益 | 说明 |
|------|---------|------|
| 异构集成 v2（V3+StatV2+Text） | ≥ 97% | 所有组件升级 |
| 端到端多模态模型 | 单模型 95%+ | 图+文字联合训练 |
| 在线学习（无审核自适应） | 持续改善 | 高置信度伪标签 |

---

## 3. 全项目总结

### 精度里程碑

```
B4.5  90.5% ──────┐
                   ├─ +0.5pp 数据增强
B5.0  91.0% ──────┤
                   ├─ +3.8pp 三路融合
B5.8  94.8% avg ──┤
                   ├─ +1.0pp 异构集成
B6.1  95.8% ──────┘  ← 突破 95% 目标
```

### 技术栈演进

```
B4: GNN 单模型
  ↓
B5: GNN + 关键词文字 + 文件名（三路融合）
  ↓
B6: GNN + StatMLP + TextMLP（异构集成） + 蒸馏 + 数据飞轮
```

### 工程成果

| 指标 | 起点 | 最终 | 提升 |
|------|------|------|------|
| 精度 | 90.5% | **95.8%** | +5.3pp |
| 延迟 | ~133ms | **34-53ms** | -55~75% |
| 体积 | 1623KB | **430KB** | -74% |
| 监控 | 无 | **漂移检测+主动学习** | 全套 |
| 自动化 | 手动 | **auto_retrain.sh** | 端到端 |
| 代码 | — | 18 文件 | 生产就绪 |
| 测试 | — | 54/54 | 全通过 |
| 文档 | — | 28 MD | 完整 |

---

---

## 4. B6.4 最终验收状态

### 全部通过的检查项

```
端到端集成验证:     21/21 ✓
单元测试:          54/54 ✓
YAML配置对齐:       fn=0.50/g2d=0.40/txt=0.10 ✓
所有模型文件存在:     6/6 ✓
```

### 最终精度确认

| 评估方式 | 精度 |
|---------|------|
| 全 manifest（4574 样本） | **95.8%** |
| 场景 B val（914 样本） | **93.9%** |
| 综合 avg（四场景加权） | **~95.4%** |

### 项目状态：**生产就绪** ✓

---

*文档版本: 2.0（最终版）*  
*创建日期: 2026-04-14*
