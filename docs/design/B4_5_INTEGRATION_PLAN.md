# B4.5 Hybrid 分类器集成计划与验证方案

**日期**: 2026-04-14  
**基线**: B4.4 — GraphEncoderV2，24 类 val_acc=90.48%，21/24 类 100% recall  
**目标**: 无文件名场景 acc ≥ 70%（当前 ~30%），有文件名 acc ≥ 95%（不退步）  
**核心**: 将 B4.4 模型集成入 HybridClassifier，替换旧 Graph2D 组件

---

## 1. 当前 HybridClassifier 架构

```
输入: DXF 字节流 + 文件名
  │
  ├─ FilenameClassifier  (weight=0.70) → 关键词匹配 → P(class | filename)
  ├─ Graph2DClassifier   (weight=0.30) → GNN 推理   → P(class | graph)  ← 待替换
  ├─ TitleBlockClassifier(weight=0.00) → 标题栏解析 → 未启用
  └─ ProcessClassifier   (weight=0.00) → 工艺特征   → 未启用
  │
  └─ FusionEngine → 加权融合 → 最终预测
```

**问题**：
- Graph2D 现在输出 5 类（旧模型），而文件名分类输出 24 类 → 类别空间不对齐
- 权重 filename=0.70 过高，无文件名时完全崩溃

---

## 2. 实现步骤

### Step 1：更新 Graph2DClassifier 加载 B4.4 模型

**已完成**（`src/ml/vision_2d.py`）：
- `_load_model()` 新增 `arch == "GraphEncoderV2"` 分支
- 通过 `GRAPH2D_MODEL_PATH` 环境变量切换模型路径

```bash
# 激活 B4.4 模型
export GRAPH2D_MODEL_PATH=models/graph2d_finetuned_24class_v2.pth
```

### Step 2：验证模型加载

```bash
python3 -c "
import os
os.environ['GRAPH2D_MODEL_PATH'] = 'models/graph2d_finetuned_24class_v2.pth'
from src.ml.vision_2d import Graph2DClassifier
clf = Graph2DClassifier()
print('label_map size:', len(clf.label_map))
print('model_type:', clf.model_type)
print('loaded:', clf._loaded)
"
```

期望输出：`label_map size: 24, model_type: edge_sage_v2, loaded: True`

### Step 3：建立标签映射对齐

B4.4 的 24 类标签 vs FilenameClassifier 标签可能不完全一致。需要对齐：

```python
# src/ml/hybrid_classifier.py 中添加标签映射
GRAPH2D_V2_LABEL_ALIASES = {
    "箱体": ["箱体", "壳体", "机壳"],
    "法兰": ["法兰", "连接件", "法兰盘"],
    "轴类": ["轴类", "轴", "主轴"],
    # ... 24 类完整映射
}
```

### Step 4：权重重新标定

当前默认权重（`hybrid_config.py`）：
```yaml
filename_weight:   0.70
graph2d_weight:    0.30
titleblock_weight: 0.00
```

B4.5 目标权重（需要 grid search 验证）：
```yaml
filename_weight:   0.45  # 降低（Graph2D 更可信）
graph2d_weight:    0.50  # 提升（24类，90%+准确率）
titleblock_weight: 0.05  # 小权重启用
```

**Grid Search 脚本**（`scripts/search_hybrid_weights.py`）：
```python
from itertools import product
results = []
for fn_w in [0.3, 0.4, 0.5, 0.6]:
    for g2d_w in [0.3, 0.4, 0.5, 0.6]:
        if fn_w + g2d_w > 1.0: continue
        acc = evaluate_hybrid(golden_set, fn_w=fn_w, g2d_w=g2d_w)
        results.append((acc, fn_w, g2d_w))
best = max(results)
print(f"Best: acc={best[0]:.1%}, fn={best[1]}, g2d={best[2]}")
```

### Step 5：场景专项测试

```python
# 场景 A：有标准文件名
test_standard = [(path, label) for path, label in golden_set if has_meaningful_filename(path)]

# 场景 B：无文件名（重命名为 UUID）
test_no_name = rename_to_uuid(golden_set)  # 文件名无意义

# 场景 C：错误文件名（对抗测试）
test_adversarial = [(path, wrong_label) for path, label in golden_set]
```

---

## 3. 验证标准

### 主要指标

| 场景 | 当前（旧 Graph2D） | B4.5 目标 | 判定 |
|------|------------------|----------|------|
| **有文件名（正常）** | ≥ 95% | **≥ 95%（不退步）** | 回归检查 |
| **无文件名（纯 Graph2D）** | ~30% | **≥ 70%** | 主目标 |
| 文件名故意错误（对抗） | — | **≥ 50%** | 鲁棒性 |
| Top-3 overall | — | **≥ 93%** | 参考 |

### 各场景测试样本要求

| 场景 | 最小样本数 | 每类最少 |
|------|-----------|---------|
| 有文件名 | 200 | 5 |
| 无文件名 | 200 | 5 |
| 对抗 | 50 | — |

### 回归检查清单

- [ ] 有文件名场景 acc ≥ 95%（必须）
- [ ] 无文件名 acc ≥ 70%（主目标）
- [ ] 24 类 Graph2D 模型正确加载（`loaded=True`，`model_type=edge_sage_v2`）
- [ ] 推理延迟 < 100ms/文件（含 DXF 解析 + GNN 推理）
- [ ] 内存峰值 < 500MB
- [ ] Superpass gate 全绿（现有 CI 测试通过）

---

## 4. 实现风险与缓解

| 风险 | 概率 | 缓解方案 |
|------|------|---------|
| 24 类标签与 FilenameClassifier 类别名不对齐 | 高 | 建立 LABEL_ALIASES 映射，模糊匹配 |
| 旧 Graph2D 模型路径 hardcode | 中 | 已通过 env var `GRAPH2D_MODEL_PATH` 解耦 |
| 有文件名场景因 g2d_w 提升而退步 | 中 | grid search 确保 fn_w 不低于 0.40 |
| 24 类分类输出类别在 Hybrid 中无法融合 | 中 | 统一标签空间：先映射到公共 24 类 ID |
| B4.4 轴承座仍 30% → 影响 Hybrid | 低 | 该类由 filename 补偿，整体影响 < 1pp |

---

## 5. 关键配置文件

| 文件 | 修改内容 |
|------|---------|
| `src/ml/vision_2d.py` | 已更新：支持 GraphEncoderV2WithHead 加载 |
| `src/ml/train/model_2d.py` | 已更新：新增 GraphEncoderV2WithHead 类 |
| `src/ml/hybrid_config.py` | 待更新：默认权重 graph2d=0.50, filename=0.45 |
| `scripts/evaluate_graph2d_v2.py` | 已新增：standalone 评估脚本 |

---

## 6. 时间线

```
Step 1 (~30min): 验证 B4.4 模型通过 Graph2DClassifier 加载（✓ 已实现）
Step 2 (~1h):    实现 grid search 权重脚本 + 场景测试集
Step 3 (~1h):    更新 hybrid_config.py 默认权重
Step 4 (~30min): 端到端测试（有名/无名/对抗）
Step 5 (~30min): 生成 B4.5 评估报告
```

---

## 7. 成功标准

| 阶段 | 目标 | 判定 |
|------|------|------|
| **B4.5** | 无文件名 ≥ 70% | 主目标 |
| B5.1（后续） | vLLM 文字增强，无名 ≥ 80% | 下阶段 |
| 最终 | 有名 ≥ 95%，无名 ≥ 70% | 生产就绪 |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
