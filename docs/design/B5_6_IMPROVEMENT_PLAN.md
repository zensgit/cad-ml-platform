# B5.6 提升计划：推理路径优化 + 主动学习闭环 + 场景 B 突破

**日期**: 2026-04-14  
**基线**: B5.5 — avg=94.1%，INT8 量化完成（-74%体积），精度 57.8%  
**目标**:
  1. 场景 B（无名有文字）≥ 92%
  2. 综合 avg ≥ 95%
  3. 全链路推理 P50 < 100ms

---

## 1. B5.5 瓶颈定位

### 1.1 综合精度构成

```
avg = 0.25×A + 0.50×B + 0.15×C + 0.10×D
    = 0.25×100% + 0.50×90.8% + 0.15×91.0% + 0.10×100%
    = 25.0% + 45.4% + 13.7% + 10.0%
    = 94.1%

要达到 95%:
    0.50×B + 59.7% = 95%
    B ≥ 70.6% / 0.50 = 不可能从 B 单独解决

实际需要 A/B/C/D 同步提升:
    若 B=92%: avg = 25 + 46 + 13.7 + 10 = 94.7%
    若 B=93%: avg = 25 + 46.5 + 13.7 + 10 = 95.2% ✓
```

**结论**：场景 B 需从 90.8% 提升至 **≥ 93%** 才能达到 avg ≥ 95%。

### 1.2 场景 B 错误分析

场景 B（无名有文字）= Graph2D + TextContent 融合。

- Graph2D 单独 = 91.0%
- 加入 TextContent 后 = 90.8%（略降）
- 文字有效命中仅 15.8%，其中部分命中错误（精度 57.8%）

**核心问题**：文字信号的有效覆盖率太低（仅 15.8%），且精度不够（57.8%），低于 Graph2D 基线。当文字分类错误时，实际拖低了 Graph2D 的正确预测。

---

## 2. B5.6a：推理路径优化（单次 ezdxf 读取）

### 2.1 问题

当前 `classify()` 中：
1. `graph2d_classifier.predict_from_bytes(file_bytes)` — 第一次 ezdxf 解析
2. `extract_text_from_bytes(file_bytes)` — **第二次 ezdxf 解析**（~15ms 浪费）

### 2.2 方案

修改 `classify()` 共享 doc 对象：

```python
# 步骤 2 修改：Graph2D 预测时同时提取文字
if file_bytes:
    import io, ezdxf
    doc = ezdxf.read(io.BytesIO(file_bytes))
    
    # Graph2D：从 doc 提取图特征
    if self._is_graph2d_enabled():
        graph2d_pred = self.graph2d_classifier.predict_from_doc(doc, filename)
    
    # TextContent：从同一 doc 提取文字（零额外 I/O）
    if self.text_content_enabled:
        from src.ml.text_extractor import _extract_from_doc
        _dxf_text = _extract_from_doc(doc)
```

**预期收益**：-15ms/次（消除重复 ezdxf 解析）

### 2.3 实施要求

需要为 `graph2d_classifier` 添加 `predict_from_doc(doc)` 方法：

```python
# src/ml/vision_2d.py 新增
def predict_from_doc(self, doc, filename: str) -> dict:
    """Predict from pre-parsed ezdxf Document (avoids re-reading DXF)."""
    msp = doc.modelspace()
    x, edge_index, edge_attr = self._dxf_to_graph(msp, ...)
    ...
```

---

## 3. B5.6b：文字分类器改进策略

### 3.1 精度优化：条件放弃策略

当前策略：任何关键词命中 → 返回 softmax 概率。
改进策略：当 top1 - top2 margin < 0.3 时放弃（避免"稍微命中"导致错误融合）。

```python
def predict_probs(self, text: str) -> dict[str, float]:
    ...
    probs = softmax(raw_scores)
    
    # B5.6b: 低 margin 放弃（避免模棱两可的预测拖低 Graph2D）
    sorted_probs = sorted(probs.values(), reverse=True)
    if len(sorted_probs) >= 2 and sorted_probs[0] - sorted_probs[1] < 0.3:
        return {}  # 放弃：信号不够清晰
    
    return probs
```

**预期效果**：
- 减少 text 错误预测参与融合的比例
- 场景 B 从 90.8% → ~91.0%（不再被错误文字信号拖低）

### 3.2 法兰命中率提升

法兰命中率 5% 是最大瓶颈。法兰 DXF 的文字内容分析：

```
典型法兰 DXF 中的实际文字：
  - "技术要求" "表面粗糙度" "未注尺寸公差" ← 通用标注（所有类都有）
  - 尺寸标注："200" "150" "M16" ← 数值，无语义
  - 偶尔："NB/T 47010" ← 已覆盖
  - 极少出现 "法兰" 二字
```

**法兰的根本问题**：法兰图纸中几乎不包含"法兰"相关词汇，关键词方法存在天然极限。

**备选方案**：
1. 添加**尺寸比例特征**：法兰的 D/d 比值（外径/内径）是区分特征 → 但属于几何特征，不适合文字分类器
2. 添加**技术要求上下文**：检测"RF面"+"密封面粗糙度"共现模式 → 条件关键词（两词同时出现才命中）
3. **放弃法兰文字分类**：接受法兰文字命中率 5%，依赖 Graph2D + 文件名

### 3.3 推荐：保守路线

鉴于法兰文字方法的天然极限，推荐：
- 不强行追求法兰文字命中率
- 重心放在提升 Graph2D 基线精度（目标 91% → 93%）
- 通过主动学习（v4 增量训练）改善 Graph2D 的弱点类别

---

## 4. B5.6c：主动学习闭环执行

### 4.1 数据收集策略

```bash
# 方式 1：生产推理自动收集（confidence < 0.50 入队）
# 自然积累，但速度依赖推理量

# 方式 2：批量跑 val set 中错误预测的样本（主动构建训练数据）
python3 -c "
import csv, torch
from scripts.evaluate_graph2d_v2 import load_model
from scripts.finetune_graph2d_from_pretrained import CachedGraphDataset, collate_finetune
from torch.utils.data import DataLoader

model, label_map = load_model('models/graph2d_finetuned_24class_v3.pth')
inv_map = {v: k for k, v in label_map.items()}
model.eval()

rows = list(csv.DictReader(open('data/graph_cache/cache_manifest.csv')))
dataset = CachedGraphDataset([(r['cache_path'], r['taxonomy_v2_class']) for r in rows], label_map)
loader = DataLoader(dataset, batch_size=64, collate_fn=collate_finetune)

errors = []
idx = 0
with torch.no_grad():
    for batch in loader:
        x, ei, ea, b, labels = batch
        logits = model(x, ei, ea, b)
        preds = logits.argmax(dim=1)
        for p, l in zip(preds, labels):
            if p.item() != l.item():
                errors.append({
                    'file_path': rows[idx]['file_path'],
                    'true_label': inv_map[l.item()],
                    'pred_label': inv_map[p.item()],
                })
            idx += 1

print(f'Errors: {len(errors)}/{len(rows)} = {len(errors)/len(rows)*100:.1f}%')
for e in errors[:20]:
    print(f'  {e[\"true_label\"]:12s} → {e[\"pred_label\"]:12s}  {e[\"file_path\"]}')
"
```

### 4.2 增量训练计划

| 数据源 | 样本量 | 说明 |
|--------|--------|------|
| 原始 manifest v2 | 4,574 | 基础训练集 |
| 增强样本（B5.0） | +1,430 | 轴承座/阀门/21 类弱类 |
| 审核样本（B5.4b） | +100~200 | 低置信度纠错（目标） |
| **总计** | ~6,200 | v4 训练集 |

### 4.3 v4 训练命令

```bash
# 追加审核样本
python scripts/append_reviewed_to_manifest.py \
    --queue data/review_queue/low_conf.csv \
    --manifest data/graph_cache_aug/cache_manifest_aug.csv \
    --output data/graph_cache_aug/cache_manifest_v4.csv \
    --corrections-only --dxf-roots /data/dxf/

# 增量训练
python scripts/finetune_graph2d_v2_augmented.py \
    --checkpoint models/graph2d_finetuned_24class_v3.pth \
    --manifest data/graph_cache_aug/cache_manifest_v4.csv \
    --output models/graph2d_finetuned_24class_v4.pth \
    --epochs 40 --batch-size 32 \
    --encoder-lr 2e-5 --head-lr 2e-4 \
    --focal-gamma 1.5 --patience 10
```

---

## 5. 验收标准

| 指标 | B5.5 结果 | B5.6 目标 | 测量方式 |
|------|----------|---------|---------|
| 综合 avg | 94.1% | **≥ 95%** | search_hybrid_weights_v2.py |
| 场景 B | 90.8% | **≥ 92%** | search_hybrid_weights_v2.py |
| Graph2D 基线 | 91.0% | **≥ 92%** | evaluate_graph2d_v2.py |
| 分类器精度 | 57.8% | **≥ 65%** | audit_text_coverage.py |
| P50 推理延迟 | 待测 | **< 100ms** | benchmark_inference.py |
| 轴承座 precision | 43% | **≥ 60%** | evaluate_graph2d_v2.py |
| v4 val acc | — | **≥ 91.0%** | evaluate_graph2d_v2.py |

---

## 6. 实施步骤

```
Week 1 (B5.6a): 推理路径优化
  → predict_from_doc() 方法（graph2d_classifier）
  → classify() 共享 doc 对象
  → 全链路基准测试（benchmark_inference.py）

Week 2 (B5.6b): 文字分类器 margin 放弃策略
  → TextContentClassifier.predict_probs() 添加 margin < 0.3 放弃
  → 重跑 audit + weight search 验证场景 B 不再被拖低

Week 3-4 (B5.6c): 主动学习 v4
  → 收集/审核低置信度样本
  → append_reviewed_to_manifest.py
  → fine-tune v3 → v4

Week 5: 综合验收
  → 四场景精度
  → INT8 量化 v4
  → 生产部署
```

---

## 7. 里程碑追踪

| 里程碑 | 内容 | 结果 | 状态 |
|--------|------|------|------|
| B5.0 | 数据增强 + v3 | 91.0% | ✓ |
| B5.1 | 文字融合评估 | 94.1% avg | ✓ |
| B5.2 | 缓存+量化 | -74% 体积 | ✓ |
| B5.3 | 监控上线 | 54/54 测试 | ✓ |
| B5.4 | TextContent推理+扩充 | 精度+18pp | ✓ |
| B5.5 | 综合验收 | avg=94.1% | ✓ |
| **B5.6a** | **推理路径优化** | **P50<100ms** | 待实施 |
| **B5.6b** | **文字分类器改进** | **精度≥65%** | 待实施 |
| **B5.6c** | **v4 增量训练** | **acc≥92%** | 待数据积累 |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
