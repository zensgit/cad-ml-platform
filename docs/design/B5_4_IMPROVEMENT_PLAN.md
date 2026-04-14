# B5.4 提升计划：主动学习增量训练 + TextContent 融合集成

**日期**: 2026-04-14  
**基线**: B5.3 — 监控上线，低置信度队列就绪，三路融合配置完成  
**目标**:
  1. 主动学习增量训练（审核队列 → 训练 → v4 模型）
  2. TextContentClassifier 正式接入 HybridClassifier 推理融合路径
  3. 无文件名场景准确率 ≥ 93%（val set，原始 manifest）

---

## 1. 当前瓶颈分析

### 1.1 三路融合实施现状

| 组件 | 状态 | 问题 |
|------|------|------|
| FilenameClassifier | ✓ 融合路径激活 | fn_w=0.45，正常工作 |
| Graph2DClassifier (v3) | ✓ 融合路径激活 | g2d_w=0.35，正常工作 |
| TextContentClassifier | ⚠️ 仅配置，未接入推理 | 仅在评估脚本中调用，HybridClassifier classify() 未调用 |
| PredictionMonitor | ✓ 集成，text_hit 始终为 False | 因 TextContent 未接入，text_hit 无实际意义 |

### 1.2 无文件名场景精度

| 场景 | 当前精度 | 目标 |
|------|---------|------|
| C: 无名无文字（纯 Graph2D） | 91.0% | ≥ 93% |
| B: 无名有文字（Graph2D + Text） | 90.9%（评估，未接入推理） | ≥ 93% |
| 综合（4 场景加权 avg） | 94.1%（评估） | ≥ 95% |

**主要障碍**：
1. TextContentClassifier 未集成进 HybridClassifier.classify()
2. 低置信度样本（主要是法兰/轴类/箱体 三主类）缺乏高质量增量训练数据
3. 轴承座 precision 仍偏低（43%），假阳性率影响整体 F1

---

## 2. B5.4a：TextContent 正式接入推理路径

### 2.1 HybridClassifier.classify() 修改

在现有 Graph2D 预测之后、融合之前加入文字分类分支：

```python
# src/ml/hybrid_classifier.py — classify() 新增段

# 2.5 文字内容分类（B5.4a：正式接入推理融合路径）
text_content_pred = None
text_hit = False

if (
    getattr(self._config.text_content, "enabled", False)
    and file_bytes
):
    try:
        from src.ml.text_extractor import extract_text_from_bytes
        from src.ml.text_classifier import TextContentClassifier

        text = extract_text_from_bytes(file_bytes)
        if text:
            clf = TextContentClassifier()
            probs = clf.predict_probs(text)
            if probs:
                text_content_pred = {
                    "label": max(probs, key=probs.get),
                    "confidence": max(probs.values()),
                    "probabilities": probs,
                    "status": "ok",
                }
                text_hit = True
                result.decision_path.append("text_content_predicted")
    except Exception as e:
        logger.warning("TextContent classification failed: %s", e)
        result.decision_path.append("text_content_error")

result.text_content_prediction = text_content_pred
```

### 2.2 ClassificationResult 扩展

```python
@dataclass
class ClassificationResult:
    ...
    # B5.4a: 新增文字内容预测字段
    text_content_prediction: Optional[Dict[str, Any]] = None
```

### 2.3 融合引擎接入

在加权融合（`_fuse_predictions()`）中加入文字内容权重：

```python
# 三路融合：filename_pred + graph2d_pred + text_content_pred
# 各自 probabilities 字典按权重叠加，softmax 归一化
text_weight = getattr(self._config.text_content, "fusion_weight", 0.10)

# 当 text_content_pred 为 None（无命中），文字权重分配给 graph2d
effective_g2d_weight = self.graph2d_weight
effective_txt_weight = text_weight if text_content_pred else 0.0
```

### 2.4 懒加载 TextContentClassifier

避免每次推理创建实例：

```python
# HybridClassifier.__init__()
self._text_content_classifier = None

@property
def text_content_classifier(self):
    if self._text_content_classifier is None:
        from src.ml.text_classifier import TextContentClassifier
        self._text_content_classifier = TextContentClassifier()
    return self._text_content_classifier
```

---

## 3. B5.4b：主动学习增量训练

### 3.1 数据收集流程

```
生产推理
  → low_conf_queue（confidence < 0.50）
  → data/review_queue/low_conf.csv

人工审核
  → 填写 reviewed_label 列
  → 保存 CSV

append_reviewed_to_manifest.py
  → 将确认样本追加到 unified_manifest_v3.csv
  → 过滤：仅追加 reviewed_label 非空 且 与 predicted_class 不同（真实纠错样本）

增量训练
  → finetune_graph2d_v2_augmented.py（基于 v3 checkpont）
  → manifest = unified_manifest_v3.csv（原始 + 审核新增）
  → 输出 models/graph2d_finetuned_24class_v4.pth
```

### 3.2 append_reviewed_to_manifest.py（待创建）

```python
#!/usr/bin/env python3
"""Append human-reviewed low-conf samples to training manifest (B5.4b).

Usage:
    python scripts/append_reviewed_to_manifest.py \
        --queue data/review_queue/low_conf.csv \
        --dxf-root /path/to/dxf/files \
        --manifest data/manifests/unified_manifest_v2.csv \
        --output data/manifests/unified_manifest_v3.csv
"""

import csv, sys
from pathlib import Path

def main():
    # 1. 读取 queue 中已审核记录（reviewed_label 非空）
    # 2. 根据 file_hash 查找原始 DXF 路径（需 DXF 存档）
    # 3. 将 (file_path, reviewed_label) 追加到 manifest
    # 4. 去重（避免同一文件多次追加）
    ...
```

### 3.3 增量训练策略

**关键考量**：

| 策略 | 说明 | 推荐 |
|------|------|------|
| 全量重训 | 原始 manifest + 新增样本，从 v3 checkpoint fine-tune | ✓ 首选 |
| 仅新增微调 | 只用新增样本 fine-tune（灾难性遗忘风险） | ✗ 不推荐 |
| 课程学习 | 先训原始，后混入新增（更稳定） | 条件好时考虑 |

**超参建议**（基于 B5.0 经验）：
```bash
python scripts/finetune_graph2d_v2_augmented.py \
    --checkpoint models/graph2d_finetuned_24class_v3.pth \
    --manifest data/manifests/unified_manifest_v3.csv \
    --output models/graph2d_finetuned_24class_v4.pth \
    --epochs 40 --batch-size 32 \
    --encoder-lr 2e-5 \      # 比 B5.0 更低（避免过拟合新增小数据集）
    --head-lr 2e-4 \
    --focal-gamma 1.5 \
    --patience 10 \
    --device cpu
```

---

## 4. B5.4c：关键词词典扩充（法兰/轴类/箱体）

B5.1 分析显示三主类关键词命中率极低（5-25%）。B5.4c 基于生产文字内容日志扩充：

### 4.1 法兰词典扩充方向

| 当前缺陷 | 改进策略 |
|---------|---------|
| DXF 实际用标准号（NB/T 47023）不用"法兰" | 添加更多标准号变体：NB/T、GB/T、HG/T |
| 只匹配"对焊法兰"等完整术语 | 添加："密封面"、"RF面"、"FF面"、"DN100"规格格式 |

```python
"法兰": [
    "对焊法兰", "平焊法兰", "螺纹法兰", "法兰盘",
    "喷涂Halar", "NB/T47010", "NB/T 47010", "flange",
    "法兰密封面",
    # B5.4c 新增
    "NB/T 47023", "GB/T 9119", "HG/T 20592", "HG/T 20615",
    "RF面", "FF面", "密封面粗糙度", "法兰厚度",
    "突面", "全平面", "凹凸面",
],
```

### 4.2 箱体词典扩充方向

| 当前缺陷 | 改进策略 |
|---------|---------|
| "角焊缝"与其他焊接件共享 | 添加箱体特有装配特征："端盖螺栓孔"、"轴承室" |
| 缺少密封相关术语 | 添加："迷宫密封"、"轴端密封"、"箱盖" |

### 4.3 评估方式

扩充后重跑：
```bash
python scripts/audit_text_coverage.py \
    --manifest data/manifests/unified_manifest_v2.csv \
    --output docs/design/B5_4_TEXT_AUDIT.md
```

目标：法兰/轴类/箱体 关键词命中率从 5-25% 提升至 ≥ 40%。

---

## 5. 验收标准

| 指标 | 目标 | 测量方式 |
|------|------|---------|
| TextContent 接入推理路径 | HybridClassifier.classify() 调用 | 代码审查 + 集成测试 |
| text_hit 监控有实际数据 | monitor.text_hit_rate ≈ 14.9% | 生产抽样 |
| 无文件名 + 有文字场景 | ≥ 93% | evaluate_graph2d_v2.py |
| 主动学习数据收集 | ≥ 200 条审核样本 | low_conf_queue.csv |
| v4 模型验证精度 | ≥ 91.0%（原始 manifest） | evaluate_graph2d_v2.py |
| 法兰/轴类/箱体 关键词命中率 | ≥ 40% | audit_text_coverage.py |
| 低置信度队列积压 | < 5%（触发后 7 天内消化） | pending_review() |

---

## 6. 实施步骤

```
Week 1 (B5.4a): TextContent 推理接入
  → 扩展 ClassificationResult 增加 text_content_prediction 字段
  → HybridClassifier.classify() 加入文字分类分支
  → 懒加载 TextContentClassifier
  → 更新融合引擎支持三路加权
  → 集成测试：verify text_hit_rate > 0 in monitor

Week 2 (B5.4b): 主动学习数据工具
  → 创建 scripts/append_reviewed_to_manifest.py
  → 积累生产低置信度样本（至少 200 条）
  → 人工审核 + 标注

Week 3 (B5.4b): 增量训练
  → 基于 v3 + 审核样本 fine-tune → v4
  → 评估 v4 精度（目标 ≥ 91%，轴承座 precision ≥ 60%）
  → 对比 v3 vs v4 混淆矩阵

Week 4 (B5.4c): 关键词词典扩充
  → 扩充法兰/箱体/轴类关键词
  → 重跑 audit_text_coverage.py
  → 目标：主类命中率 ≥ 40%

Week 5 (B5.5): 综合评估
  → 四场景加权 avg ≥ 95% 目标验收
  → 性能基准测试（含 TextContent 的完整推理链路）
  → 生产部署 v4 模型
```

---

## 7. 里程碑追踪

| 里程碑 | 内容 | 目标 | 状态 |
|--------|------|------|------|
| B5.0 | 数据增强 + 模型 v3 | 91.0% | ✓ |
| B5.1 | 文字融合评估（三路） | 94.1% avg | ✓ |
| B5.2a | 文字缓存集成 | 单次 I/O | ✓ |
| B5.2b | INT8 量化 | 待运行 | ⏳ |
| B5.3 | 监控 + 低置信度队列 | 生产监控 | ✓ |
| **B5.4a** | **TextContent 推理接入** | **text_hit 有效** | 待实施 |
| **B5.4b** | **主动学习增量 v4** | **≥ 91% + 轴承座 prec ≥ 60%** | 待实施 |
| **B5.4c** | **关键词词典扩充** | **主类命中率 ≥ 40%** | 待实施 |
| B5.5 | 综合验收 | 4 场景 avg ≥ 95% | 待规划 |

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| 主动学习样本质量低 | 中 | 高 | 要求标注者熟悉领域，双人交叉审核 |
| TextContent 接入后精度下降 | 低 | 中 | 保留 txt_w=0.10 低权重，可临时 disable |
| 低置信度队列积压过多 | 中 | 低 | 批量处理，优先审核量大的错误类别 |
| v4 训练过拟合新增样本 | 低 | 高 | 使用低 encoder-lr（2e-5），FocalLoss 约束 |

---

*文档版本: 1.0*  
*创建日期: 2026-04-14*
