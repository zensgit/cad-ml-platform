# Batch H 召回提升与 GD&T 符号支持 设计总结

## 目标
- 提升 CPU 场景 PaddleOCR 的文本召回（基础预处理）。
- 增强解析器单位归一与 GD&T 符号覆盖，利于 Week1 指标（Recall / Edge-F1）。

## 实现
- 图像预处理：`src/core/ocr/preprocessing/image_enhancer.py`
  - 灰度化、等比缩放≤2048px、轻度中值滤波与反锐化
  - 非图片输入自动回退，不破坏请求
- Provider 接入：`PaddleOcrProvider(enable_preprocess=True, max_res=2048)`
- 解析器：`dimension_parser.py`
  - 单位解析并归一到 mm（mm/cm/μm/um）
  - GD&T 关键词代理识别：flatness/position/total runout/profile of a line/surface 等
- 测试：
  - `tests/ocr/test_image_enhancer.py`（预处理烟测）
  - `tests/ocr/test_dimension_parser_precision.py::test_units_normalization_to_mm`
  - `tests/ocr/test_dimension_parser_regex.py::test_parse_geometric_symbols`

## 局限与展望
- GD&T 目前为关键词启发式，后续结合 DeepSeek JSON 输出与版面位置信息降低误报。
- 预处理使用固定参数，可按数据分布调整（例如阈值化、对比度增强）。

## 验收
- 单元测试 100% 通过（51/51）
- 下一步建议在 Golden 集合上进行前后对比并记录报告。

