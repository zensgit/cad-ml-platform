# Batch C 解析精度增强设计总结

## 目标
在 Week1 内提升基础解析器对 CAD 尺寸/公差/螺纹/几何符号的结构化准确率，完善双向公差绑定策略，减少 pitch / tolerance 解析错误，为后续 Recall / Edge-F1 指标评估奠定可衡量基础。

## 本批次实现内容
- 双向公差 ( +a -b ) 精确绑定：基于字符跨度与最近前置尺寸的距离 (gap_threshold=24)，支持与直径 token 重叠的场景（如 `Φ20 +0.02 -0.01` 中 `+0.02` 被直径正则吸收）。
- 单侧公差补全：若先解析到 `Φ20 +0.02`，后续再出现 `-0.01` 的 dual 模式，可合并为 tol_pos=0.02, tol_neg=0.01。
- 螺纹 pitch 变体：支持 `× x X *` 四种分隔符形式 (`M10×1.5 M10x1.5 M10X1.5 M10*1.5`)。
- 几何符号解析：增加 `⊥` (perpendicular) 与 `∥` (parallel) 的提取与规范化字段 `normalized_form`。
- 宽容跨度阈值：gap 从 12 扩大到 24，用于适配 OCR 结果中的多余空格/噪声。

## 正则模式更新
```python
DIAMETER_PATTERN: [Φ⌀∅]\s*(number)(optional single tol)
RADIUS_PATTERN: R\s*(number)(optional single tol)
THREAD_PATTERN: M(number)([×xX*](pitch))?
DUAL_TOL_PATTERN: +number [-number]
PERP_PATTERN: ⊥
PARA_PATTERN: ∥
```

## 双向公差绑定逻辑
1. 预先记录每个尺寸 token 的起止 span。
2. 扫描 dual tolerance pattern：`+a -b`。
3. 候选集合规则：
   - 若 dual 起始位置位于某尺寸 span 内 → 距离记为 0（优先绑定）。
   - 若 dual 在尺寸之后且距离 ≤ gap_threshold → 收集 (距离, 尺寸)。
4. 选择距离最小尺寸：
   - 若尺寸已有单侧 tol_pos → 补全 tol_neg。
   - 若尺寸已有单侧 tol_neg → 补全 tol_pos。
   - 若都没有 → 直接赋值两侧公差。
5. 统一 tolerance 字段 = max(tol_pos, tol_neg)。

## 新增/更新测试
文件：`tests/ocr/test_dimension_parser_precision.py`
测试覆盖：
- `test_dual_tolerance_binding_gap_limit`: 紧邻场景绑定。
- `test_dual_tolerance_not_bound_if_far_gap`: 跨距过长不绑定。
- `test_thread_pitch_variants`: pitch 分隔符一致性。
- `test_mixed_sequence_robustness`: 混合顺序正确绑定末尾 dual 公差。
- 补充原有 `test_dimension_parser_regex.py` 中的几何符号提取。

## 当前局限 & 后续改进方向
- 仍未处理“多个尺寸后接多个 dual tolerance”一一映射的复杂排布。
- 未解析负号在主尺寸值中的嵌套场景（极少见）。
- 未进行 bbox 级别的空间匹配（需图像坐标）。
- Ra 单位归一暂保持原值，后续可统一 μm→mm。
- 未加入更复杂的 GD&T 特征（平面度、位置度等）。

## 指标影响预期
- Dimension dual tolerance 完整性：预期可从 ~50% 提升到 ≥90%（对简单样例）。
- Thread pitch 解析覆盖率：多分隔符支持后，样例准确率接近 100%。
- Geometric symbol 召回：提供基础支持，后续可纳入 Edge-F1 提评估。

## 变更风险评估
- gap_threshold 增大可能提升误绑定风险（远距离偶然出现 `+a -b`）。目前通过距离排序和单侧已有公差逻辑减轻影响。
- 多 pitch 分隔符正则增广可能导致极端噪声误识别（需后续置信度过滤或上下文约束）。

## 后续批次衔接
为 Batch D 的动态阈值与 Batch E 的分布式限流提供更准确的结构化字段（tol_pos/tol_neg），有助于：
- 更精细的 completeness 计算（含双向公差）。
- 置信度校准特征扩展（单侧 vs 双侧公差提取成功率）。

## 实施清单摘要
- 解析器更新：`src/core/ocr/parsing/dimension_parser.py`。
- 新增测试：`tests/ocr/test_dimension_parser_precision.py`。
- 更新测试：`tests/ocr/test_dimension_parser_regex.py` 增补符号案例。

## 验收
- 所有新增测试通过（4 例）。
- 原有测试保持通过。

---
Batch C 完成，准备进入 Batch D（动态阈值 / RollingStats）。

