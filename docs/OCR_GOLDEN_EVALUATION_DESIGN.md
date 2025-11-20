## Golden Evaluation Design (v1)

### 目标
建立最小评测闭环度量 OCR 结构化解析质量：
- Dimension Recall / Symbol Recall
- Edge-F1 (bbox匹配 IOU≥0.5)
- Brier Score (置信度校准)
- Dual Tolerance Accuracy

### 样本结构
`tests/ocr/golden/samples/<id>/annotation.json`
```json
{
  "dimensions": [
    {"type":"diameter","value":20.0,"tol_pos":0.02,"tol_neg":0.01,"unit":"mm","bbox":[100,120,40,18]},
    {"type":"radius","value":5.0,"unit":"mm","bbox":[220,140,30,15]}
  ],
  "symbols": [
    {"type":"surface_roughness","value":"3.2","bbox":[300,210,25,12]}
  ]
}
```

### 匹配规则
1. Dimension: 绝对差值 ≤ `max(tol_pos, tol_neg, 0.05*value)`
2. BBox: IOU≥0.5 视为匹配；允许尺寸与 bbox 同时匹配增计。
3. Symbol: 文字规范化匹配 (大小写去除 + 去空格)。
4. Dual tolerance accuracy: 双向容差均正确视为 True。

### 指标公式
`dimension_recall = matched_dimensions / total_dimensions`
`symbol_recall = matched_symbols / total_symbols`
`edge_precision = matched_bboxes / predicted_bboxes`
`edge_recall = matched_bboxes / gt_bboxes`
`edge_f1 = 2 * P * R / (P + R)`
`brier_score = sum((p_i - o_i)^2)/N` (o_i=1 正确, 0 错误)

### 评测脚本流程
1. 加载所有 annotation.json 与对应 OCR 输出缓存 (或实时调用)。
2. 对每个样本运行匹配器 → 累积计数。
3. 计算指标并输出 Markdown 报告 `reports/ocr_evaluation.md`。
4. CI 集成：若 recall 或 edge_f1 低于阈值 -> fail pipeline。

### 阈值建议 (初始)
- dimension_recall ≥ 0.70 (Week1), 0.80 (Week2)
- edge_f1 ≥ 0.60 (Week1), 0.75 (Week2)
- dual_tolerance_accuracy ≥ 0.60 (Week2)

### 风险与缓解
- 样本过少：添加多难度分类 (easy/medium/hard/edge)。
- IOU 过严：若匹配率低可临时降低到 0.45 再调优。
- Brier 偏高：调整 calibrated_confidence 公式权重。

### v2 拓展
- 加入字段级别误差统计 (实际值差距分布直方图)。
- 标注工具自动化（半自动标注 + 可视化审阅）。

