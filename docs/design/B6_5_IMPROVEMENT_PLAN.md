# B6.5 提升计划：长期路线图 + 运维手册

**日期**: 2026-04-14  
**基线**: B6.5 — 异构集成 95.8%，P50=27ms，零错误，蒸馏实验完结  
**状态**: **项目完成，生产就绪**

---

## 1. 最终实验结论

### 1.1 蒸馏实验关闭

| 版本 | 策略 | 结果 | 结论 |
|------|------|------|------|
| v5 | warm-start, α=0.3, T=3 | 91.0% | soft label 过平滑 |
| v5b | warm-start, α=0.7, T=1.5 | 90.6% | warm-start 本身是问题 |
| v5c | 从头训练, α=0.5, T=2 | 89.9% | 60 epochs 不足以从零学习 |
| **v4** | **直接 fine-tune** | **91.9%** | **最优单模型** |

**蒸馏在当前条件下不可行**。需要 200+ epochs 或 GPU 加速才能从头蒸馏到 v4 水平。

### 1.2 确认最终方案

**异构集成（GNN v4 + StatMLP + TextMLP）= 95.8% 是不可替代的最优方案。**

---

## 2. 未来提升方向（按优先级）

### P0: 数据飞轮运营（持续）

```bash
# 每月执行
bash scripts/auto_retrain.sh
```

预期：每积累 200 条审核样本 → v(N+1) 精度 +0.3-0.5pp

### P1: GraphEncoderV3（需 GPU，2-4 周）

5 层 JK-Net + Multi-head Attention → GNN 单模型预期 93%+  
集成升级后预期：**97%+**

### P2: 端到端多模态（需 GPU，1-2 月）

图特征 + 文字 embedding 联合训练的单一模型 → 预期 95%+ 单模型

### P3: 在线学习（长期）

高置信度生产预测自动加入训练集（pseudo-label）→ 持续自适应

---

## 3. 运维手册

### 3.1 日常监控

```python
from src.ml.hybrid_classifier import HybridClassifier
clf = HybridClassifier()
# 在 100+ 次推理后
s = clf.monitor.summary()
# 检查: drift_detected=False, low_conf_rate < 0.10, avg_confidence > 0.7
```

### 3.2 告警处理

| 告警 | 条件 | 处理 |
|------|------|------|
| DRIFT ALERT | low_conf_rate > 10% | 检查输入数据分布，可能需要重训 |
| TEXT SIGNAL | text_hit_rate < 5% | 检查文字提取是否正常 |

### 3.3 模型更新流程

```
1. 审核 low_conf_queue (每周)
2. 积累 ≥ 200 条 → bash scripts/auto_retrain.sh
3. 精度门控 ≥ 91.5% → 自动量化
4. 手动确认 → 更新 GRAPH2D_MODEL_PATH
```

### 3.4 回滚

```bash
# 切回 v4 单模型
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v4_int8.pth
export STAT_MLP_ENABLED=false
export TFIDF_TEXT_ENABLED=false

# 禁用文字融合
export TEXT_CONTENT_ENABLED=false
```

---

*项目完成 — 2026-04-14*
