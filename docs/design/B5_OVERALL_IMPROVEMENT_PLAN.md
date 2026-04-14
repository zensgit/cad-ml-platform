# B5+ 系统级提升计划与验证方案

**日期**: 2026-04-14  
**前提**: B4 完成后（Graph2D 24类 ≥ 83%，Hybrid 无文件名 ≥ 70%）  
**目标**: 系统整体生产就绪，推理延迟 < 200ms，准确率 ≥ 90%（有文件名）

---

## 全局视图：从 B4 到 B5

```
B4（当前）       B5（规划）
─────────────   ─────────────────────────────
Graph2D 24类    → vLLM 语义增强 + 多模态融合
Hybrid 分类器   → 自动权重自适应（运行时学习）
5类→24类标签    → 动态标签扩展（新类自动发现）
静态模型        → 在线学习（用户反馈闭环）
```

---

## Phase B5.1：vLLM 语义增强分类器

**原理**：Graph2D 只看图结构，无法理解文字标注。vLLM（已集成）可以解析 DXF 中的文字内容（标题栏、标注、技术要求）并给出语义分类。

### 5.1.1 设计方案

```
DXF 文件
  ├── 图结构 → Graph2D（当前）→ P(class | graph)
  ├── 文字内容 → vLLM OCR+分类  → P(class | text)
  └── 文件名  → 关键词匹配      → P(class | filename)

HybridClassifier v2:
  final = w1 * P_graph + w2 * P_text + w3 * P_filename
```

### 5.1.2 文字提取策略

```python
def extract_dxf_text(dxf_path: str) -> dict:
    """从 DXF 提取所有文字内容."""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    texts = {
        'title_block': [],   # 标题栏文字
        'dimensions':  [],   # 尺寸标注
        'notes':       [],   # 技术要求
        'labels':      [],   # 零件标签
    }
    for e in msp:
        if e.dxftype() in ('TEXT', 'MTEXT'):
            texts['labels'].append(e.dxf.text)
        elif e.dxftype() == 'DIMENSION':
            texts['dimensions'].append(str(e.dxf.measurement))
    return texts
```

### 5.1.3 vLLM 分类 Prompt

```python
CLASSIFICATION_PROMPT = """
你是一个 CAD 零件分类专家。根据以下 DXF 图纸信息，判断零件类型。

文字内容：{text_content}
文件名：{filename}

可选类别：法兰、轴类、箱体、传动件、轴承座、阀门、弹簧、支架...（24类）

要求：
1. 输出最可能的类别（1个）
2. 输出置信度（0-1）
3. 简短说明理由

输出格式：{{"class": "法兰", "confidence": 0.92, "reason": "包含法兰盘尺寸标注"}}
"""
```

### 5.1.4 验证标准

| 指标 | B4.5 基线 | B5.1 目标 |
|------|----------|----------|
| 无文件名 acc | 70% | **≥ 80%** |
| 有文件名 acc | 95% | **≥ 97%** |
| 新类发现（未见类） | 0% | **> 0%**（open-set） |
| 延迟（vLLM 本地） | — | < 500ms |

---

## Phase B5.2：自适应权重调整

**原理**：当前 Hybrid 权重固定（手动调参），应该根据当前 DXF 特征动态调整。

### 设计

```python
class AdaptiveHybridClassifier:
    def predict(self, dxf_path: str) -> ClassificationResult:
        # 特征检测
        has_filename_signal = self._has_meaningful_filename(dxf_path)
        has_text_content   = self._extract_text(dxf_path) != ""
        graph_confidence   = self._graph2d_confidence(dxf_path)

        # 动态权重
        if has_filename_signal and graph_confidence > 0.8:
            weights = {'filename': 0.5, 'graph': 0.4, 'text': 0.1}
        elif not has_filename_signal:
            weights = {'filename': 0.0, 'graph': 0.5, 'text': 0.5}
        else:
            weights = {'filename': 0.4, 'graph': 0.35, 'text': 0.25}

        return self._weighted_predict(dxf_path, weights)
```

### 验证标准

- [ ] 无文件名场景 acc ≥ 85%（动态权重 vs 固定权重 +10pp）
- [ ] 文件名冲突场景（文件名错误）acc ≥ 70%
- [ ] A/B 测试：自适应 > 固定权重，统计显著

---

## Phase B5.3：在线学习闭环

**原理**：用户纠正错误分类 → 反馈进入训练池 → 定期微调模型。

### 数据飞轮设计

```
用户操作 → 审计日志 → 反馈收集 → 标签验证 → 增量训练
                                              ↓
                                    每日/每周模型更新
```

### 实现要点

```python
# 1. 收集用户纠正
class FeedbackCollector:
    def record_correction(self, file_path: str, predicted: str, corrected: str):
        self.db.insert({'path': file_path, 'pred': predicted, 
                       'true': corrected, 'ts': datetime.now()})

# 2. 增量微调（每周）
class IncrementalTrainer:
    def retrain(self, new_samples: List[Sample]):
        # 混合新样本（10%）+ 原训练集（90%）防止遗忘
        mixed = self.sample_original(ratio=0.9) + new_samples
        self.finetune(mixed, epochs=5, lr=0.00001)
```

### 验证标准

- [ ] 连续 2 周内，模型 acc 提升 ≥ 1pp/周
- [ ] 用户纠正的类别 recall 提升 ≥ 10%
- [ ] 旧类别 acc 不退步（灾难性遗忘率 < 1%）

---

## Phase B5.4：动态标签扩展（新类发现）

**原理**：当多个 DXF 被 vLLM 分类为"未知"且嵌入距离接近时，自动建议新类别。

### 算法

```python
def discover_new_classes(embeddings: np.ndarray, threshold: float = 0.3):
    # DBSCAN 聚类未被已知类覆盖的样本
    from sklearn.cluster import DBSCAN
    clusters = DBSCAN(eps=threshold, min_samples=5).fit(embeddings)
    new_clusters = [c for c in clusters.labels_ if c not in known_classes]
    return new_clusters

# 人工确认后加入标签系统
```

### 验证标准

- [ ] 自动发现新类精度 ≥ 80%（人工确认通过率）
- [ ] 发现间隔 < 1 周（足够多的新文件）

---

## 整体技术栈路线

```
当前（B4）                    目标（B5）
──────────────────────────   ──────────────────────────────
Graph2D (EdgeGraphSAGE)  →   GraphEncoderV2 + vLLM 融合
静态权重 Hybrid          →   自适应权重 Hybrid
离线批量训练             →   在线增量学习
5/24 类固定标签          →   动态标签发现
CPU 推理                 →   可选 MPS/CUDA 加速
```

---

## 优先级与时间估计

| 阶段 | 优先级 | 预计工作量 | 收益 |
|------|--------|-----------|------|
| B4.4 架构升级 | P0（进行中） | ~2h | +3-5pp acc |
| B4.5 Hybrid集成 | P0 | ~3h | 无名 +40pp |
| B5.1 vLLM融合 | P1 | ~1天 | 无名 +10pp，开放集 |
| B5.2 自适应权重 | P1 | ~4h | +5-10pp（特殊场景） |
| B5.3 在线学习 | P2 | ~2天 | 持续改进 |
| B5.4 新类发现 | P3 | ~3天 | 自动扩展 |

---

## 验证框架：Golden Test Set

所有阶段共用的标准测试集，保证可比性：

```python
# 构建 golden test set（每类至少 10 个，总计 ~200 个样本）
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, golden_idx = next(sss.split(X, y))

# 场景覆盖
golden_set = {
    'standard':   golden[golden.has_filename == True],   # 有标准文件名
    'no_name':    golden[golden.has_filename == False],  # 无文件名
    'adversarial': golden[golden.wrong_name == True],    # 错误文件名
}
```

### 每次发布前必须通过

| 检查项 | 阈值 |
|--------|------|
| Golden set overall acc | ≥ 上一版本 |
| 有文件名 acc | ≥ 95% |
| 无文件名 acc | ≥ B4.5 目标 |
| 推理延迟 P99 | < 200ms |
| 模型文件大小 | < 50MB |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
