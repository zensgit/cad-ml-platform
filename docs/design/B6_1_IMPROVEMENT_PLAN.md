# B6.1 提升计划：异构集成生产化 + GraphEncoderV3 + 数据飞轮

**日期**: 2026-04-14  
**基线**: B6.0 — **异构集成 95.8%（已突破 95%）**，StatMLP 94.4%，TextMLP 73.7%  
**目标**: 生产部署异构集成，持续数据飞轮，向 97%+ 迈进

---

## 1. B6.1a：异构集成生产集成

### 1.1 TextMLP 接入 HybridClassifier

将 TF-IDF MLP 作为 TextContentClassifier 的 fallback：

```python
# src/ml/text_classifier.py 修改
class TextContentClassifier:
    def __init__(self, tfidf_model_path: str = "models/text_classifier_tfidf.pth"):
        # 加载 TF-IDF MLP（如果存在）
        self._tfidf_model = None
        self._vectorizer = None
        if Path(tfidf_model_path).exists():
            ckpt = torch.load(tfidf_model_path, map_location="cpu")
            self._tfidf_model = TextMLP(...)
            self._tfidf_model.load_state_dict(ckpt["model_state"])
            # 加载 vectorizer
    
    def predict_probs(self, text: str) -> dict[str, float]:
        # 1. 优先：关键词匹配（高精度，低覆盖）
        kw_probs = self._keyword_predict(text)
        if kw_probs:
            return kw_probs
        
        # 2. Fallback：TF-IDF MLP（中精度，高覆盖）
        if self._tfidf_model and len(text.strip()) >= self.MIN_TEXT_LEN:
            return self._tfidf_predict(text)
        
        return {}
```

### 1.2 预期效果

| 场景 | 关键词覆盖 | +TF-IDF 覆盖 | 说明 |
|------|-----------|------------|------|
| 有文字+关键词命中 | 20.1% | 20.1% | 关键词优先（精度更高） |
| 有文字+无关键词 | — | +59% | TF-IDF fallback |
| 无文字 | — | — | 均放弃 |
| **总覆盖** | **20.1%** | **~79%** | **+59pp** |

---

## 2. B6.1b：GraphEncoderV3（更深 GNN）

### 2.1 架构设计

```python
class GraphEncoderV3WithHead(nn.Module):
    """5-layer GNN with JumpingKnowledge + Multi-head Attention Pooling."""
    
    def __init__(self, node_dim=19, edge_dim=7, hidden_dim=256,
                 num_layers=5, heads=4, num_classes=24):
        # 5-layer EdgeSAGEConv with residual connections
        self.convs = ModuleList([
            SAGEConv(in_dim, hidden_dim) for i in range(num_layers)
        ])
        # JumpingKnowledge: concatenate all layer outputs
        self.jk = JumpingKnowledge('cat', hidden_dim, num_layers)
        # Multi-head attention pooling
        self.pool = GlobalAttention(
            gate_nn=Linear(hidden_dim * num_layers, heads),
            nn=Linear(hidden_dim * num_layers, hidden_dim)
        )
        self.classifier = Linear(hidden_dim, num_classes)
```

### 2.2 训练要求

| 需求 | 说明 |
|------|------|
| GPU | 5 层 GNN × 4574 样本 → 预计 2-4 小时（GPU），20+ 小时（CPU） |
| 数据 | 现有 6760 增强样本足够 |
| 内存 | hidden_dim=256 × 5 层 ≈ 模型 ~5MB |

### 2.3 预期精度

基于文献和类似任务，5 层 + JK-Net 通常比 3 层提升 1-2pp：
- v4 (3层): 91.9% → V3 (5层): **预期 93-94%**

---

## 3. B6.1c：数据飞轮

### 3.1 自动化管线

```
┌─────────────────────────────────────────────────┐
│                 生产推理                          │
│  classify() → monitor.record()                  │
│            → low_conf_queue.maybe_enqueue()     │
└──────────────────────┬──────────────────────────┘
                       │ confidence < 0.50
                       ▼
┌─────────────────────────────────────────────────┐
│            data/review_queue/low_conf.csv        │
│  自动积累（生产运行中）                            │
└──────────────────────┬──────────────────────────┘
                       │ 每周人工审核
                       ▼
┌─────────────────────────────────────────────────┐
│       scripts/append_reviewed_to_manifest.py     │
│  审核样本 → 训练 manifest                         │
└──────────────────────┬──────────────────────────┘
                       │ 积累 200+ 条
                       ▼
┌─────────────────────────────────────────────────┐
│    scripts/finetune_graph2d_v2_augmented.py      │
│  增量训练 v(N) → v(N+1)                          │
│  + scripts/quantize_graph2d_model.py             │
└──────────────────────┬──────────────────────────┘
                       │ 自动评估 acc ≥ 阈值
                       ▼
┌─────────────────────────────────────────────────┐
│               部署新模型                          │
│  GRAPH2D_MODEL_PATH=models/vN+1_int8.pth        │
└─────────────────────────────────────────────────┘
```

### 3.2 自动化脚本（建议创建）

```bash
# scripts/auto_retrain.sh
#!/bin/bash
# 1. Check if enough reviewed samples
REVIEWED=$(python3 -c "from src.ml.low_conf_queue import LowConfidenceQueue; print(LowConfidenceQueue().reviewed_entries().__len__())")
if [ "$REVIEWED" -lt 200 ]; then echo "Not enough data ($REVIEWED < 200)"; exit 0; fi

# 2. Append to manifest
python3 scripts/append_reviewed_to_manifest.py ...

# 3. Train
python3 scripts/finetune_graph2d_v2_augmented.py ...

# 4. Evaluate (gate: acc >= 91.5%)
ACC=$(python3 scripts/evaluate_graph2d_v2.py ... | grep "Overall" | awk '{print $4}')
if (( $(echo "$ACC >= 91.5" | bc -l) )); then
    echo "PASS: acc=$ACC >= 91.5%"
    # 5. Quantize
    python3 scripts/quantize_graph2d_model.py ...
    echo "Ready for deployment"
else
    echo "FAIL: acc=$ACC < 91.5%, not deploying"
fi
```

---

## 4. 验收标准

| 阶段 | 指标 | 目标 |
|------|------|------|
| B6.1a TF-IDF fallback 集成 | 文字覆盖率 | 20% → **~79%** |
| B6.1a 融合精度 | avg | **≥ 95.0%** |
| B6.1b GraphEncoderV3 | Graph2D acc | **≥ 93%** |
| B6.1c 数据飞轮 | 自动化管线就绪 | 脚本可执行 |

---

## 5. 实施步骤

```
Week 1: B6.1a — TF-IDF fallback 集成
  → TextContentClassifier 增加 TF-IDF fallback
  → 重跑四场景权重搜索
  → 验证 avg ≥ 95%

Week 2-3: B6.1b — GraphEncoderV3
  → 实现 V3 架构（5层 + JK-Net）
  → 训练（需 GPU 环境）
  → 评估 → 量化

Week 4: B6.1c — 数据飞轮
  → auto_retrain.sh 脚本
  → CI/CD 集成（可选）
  → 文档化运维流程
```

---

## 6. 里程碑追踪

| 里程碑 | 内容 | 结果 | 状态 |
|--------|------|------|------|
| B5.0-B5.9 | 基础优化全系列 | avg=94.8% | ✓ 完成 |
| B6.0b | TF-IDF 文字 MLP | **73.7%** | ✓ |
| B6.0c | 统计特征 MLP | 训练中 | ⏳ |
| **B6.1a** | **TF-IDF fallback** | **avg ≥ 95%** | 待实施 |
| **B6.1b** | **GraphEncoderV3** | **≥ 93%** | 待 GPU |
| **B6.1c** | **数据飞轮** | **自动化** | 待实施 |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
