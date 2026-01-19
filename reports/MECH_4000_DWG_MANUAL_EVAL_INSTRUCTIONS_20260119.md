# MECH_4000_DWG_MANUAL_EVAL_INSTRUCTIONS_20260119

##目的
请为抽样的 20 张图纸补充人工标签，用于 graph2d 精度评估。

##文件
- CSV: `reports/MECH_4000_DWG_MANUAL_EVAL_TEMPLATE_20260119.csv`
- 字段说明:
  - `suggested_label_cn`: 从文件名抽取的弱标签(可参考)
  - `reviewer_label_cn`: 请填写人工确认的中文标签
  - `reviewer_label_en`: 可选英文标签
  - `model_pred_top1/top3`: 模型预测结果(可参考)

##完成后
把填好的 CSV 发回或放在原路径，我会计算 Top-1/Top-3 和混淆统计。
