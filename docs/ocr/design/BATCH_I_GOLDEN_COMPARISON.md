# Batch I 黄金评测对比增强 设计总结

## 目标
- 在 Golden 集合上量化预处理（开启/关闭）对维度/符号召回、Edge-F1、Brier、双向公差准确率的影响。

## 实现
- `tests/ocr/golden/run_golden_evaluation.py` 支持 `--compare-preprocess`：
  - 预处理关闭（PaddleOcrProvider(enable_preprocess=False)）与开启（True）各跑一遍；
  - 输出 Aggregate OFF/ON 指标与 Delta；
  - 仍保留 Week1 基线阈值门控（维度召回与 Edge-F1）。
- 新增示例标注：
  - `sample_002`：直径/半径/螺纹 + Ra
  - `sample_003`：双向公差 + ⊥/∥

## 现状
- 由于目前 provider 与 bbox 仍为桩/启发式，Edge-F1 为 0；当加入真实检测框或映射后可提升。
- 维度/符号召回在样例上为 0.667（占位数据）；双向公差准确率 0.333。

## 下一步
- 引入真实样本图与 bbox，或基于 PaddleOCR 检测框与文本行进行 bbox 对齐评估。
- 将脚本纳入 CI（可使用宽松门槛），作为回归基线。

