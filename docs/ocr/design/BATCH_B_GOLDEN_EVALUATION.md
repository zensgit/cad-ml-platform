## Batch B: Golden数据集与评测闭环设计总结 (v1)

### 1. 背景与动机
已有OCR解析与阶段计时，但缺乏量化质量基线。需要一个最小闭环来持续测量尺寸/符号召回、边界质量与置信度校准。Batch B 建立 Golden 数据集与评测脚本，为后续解析精度与阈值动态调整提供真实反馈。

### 2. 目标
- 规范 Golden 样本目录结构与标注格式
- 实现评测脚本生成 `reports/ocr_evaluation.md`
- 计算核心指标：dimension_recall / symbol_recall / edge_f1 / brier_score / dual_tolerance_accuracy
- CI 集成基础阈值门控 (失败即退出非零)

### 3. 非目标
- 不进行高级可视化 (Heatmap)。
- 不做自动标注工具 (仅手工JSON)。

### 4. 数据集结构
```
tests/ocr/golden/
  metadata.yaml
  samples/
    sample_001/
      annotation.json
      image.png (占位或真实)
    sample_002/
      annotation.json
```
`metadata.yaml` 记录版本、分类数量与schema。

### 5. 标注 Schema (annotation.json)
```json
{
  "dimensions": [{"type":"diameter","value":20.0,"tol_pos":0.02,"tol_neg":0.01,"unit":"mm","bbox":[100,120,40,18]}],
  "symbols": [{"type":"surface_roughness","value":"3.2","bbox":[300,210,25,12]}]
}
```

### 6. 匹配算法摘要
- 数值匹配: `abs(pred - gt) <= max(gt_tol_pos, gt_tol_neg, 0.05*gt_value)`
- BBox IOU≥0.5 视为边界匹配
- 双向公差: 同时匹配 tol_pos & tol_neg 记为正确
- 置信度校准: 使用 `calibrated_confidence` 计算 Brier Score

### 7. 指标公式
`dimension_recall = matched_dimensions / total_dimensions`
`symbol_recall = matched_symbols / total_symbols`
`edge_precision = matched_bboxes / predicted_bboxes`
`edge_recall = matched_bboxes / gt_bboxes`
`edge_f1 = 2*P*R/(P+R)`
`brier_score = sum((p_i - o_i)^2)/N`
`dual_tolerance_accuracy = correct_dual / total_dual`

### 8. 阈值 (初始门控)
- Week1: dimension_recall ≥0.70, edge_f1 ≥0.60
- Week2: dimension_recall ≥0.80, edge_f1 ≥0.75, dual_tolerance_accuracy ≥0.60

### 9. 实现文件列表
- `tests/ocr/golden/metadata.yaml` (数据集描述)
- `tests/ocr/golden/samples/sample_001/annotation.json` (示例)
- `tests/ocr/golden/run_golden_evaluation.py` (主评测脚本)
- 输出报告: `reports/ocr_evaluation.md`

### 10. 评测脚本流程
1. 遍历 `samples/` 加载标注
2. 调用 `OcrManager.extract` (或读取缓存) 获得预测
3. 逐字段运行匹配器累积统计
4. 计算指标并生成Markdown报告
5. 若指标低于当前阈值 -> `sys.exit(2)`

### 11. 风险与回退
- 样本太少导致指标波动：临时增加 easy 样本数至 5+。
- 评测耗时过长：暂不处理PDF多页；仅单图像。
- 无法加载真实模型：DeepSeek使用 stub 输出仍可跑评测逻辑。

### 12. 验收标准
- 生成报告包含所有指标行与时间戳
- CI中运行脚本且在当前 stub 条件下通过门槛或明确失败原因
- 报告列出每类别 (easy/medium 等占位) 计数

### 13. 后续扩展
- 添加误差分布直方图
- 标注辅助工具 (可视化 bbox 编辑)
- 多批次版本对比差异报告

