## Dual Tolerance Parsing Design (v1)

### 背景
CAD尺寸常出现双向公差: `Φ20 +0.02 -0.01` 表示正负容差不对称。当前实现仅捕获正向或统一容差，需要精确绑定两侧。

### 目标
- 区分 tol_pos / tol_neg
- 评测时匹配规则: `abs(pred - nominal) <= max(tol_pos, tol_neg, 0.05*nominal)`
- 支持多种符号: `+0.02 -0.01`, `＋0.02 －0.01`, 间隔可含空格/换行。

### 解析策略
1. Tokenize: 通过正则抓取 nominal 数值与公差对：
   - Nominal: `(Φ|⌀|∅)?\s*(\d+(?:\.\d+)?)`
   - Dual: `([+＋]\s*\d+(?:\.\d+)?)\s+([-－]\s*\d+(?:\.\d+)?)`
2. 计算 token span 中心位置。
3. 对每个 dual 公差，寻找最近的 nominal span（距离阈值 < 25 字符，可调）。
4. 若 nominal 已有单侧容差 -> 组合补全 (保留原 tol_pos)。
5. 容差归一：将全角符号替换为半角；去除空格。

### 数据模型扩展
`DimensionInfo`: 已加入 `tol_pos`, `tol_neg`。若仅单值公差则 `tolerance=tol_pos` (或 tol_neg)。

### 评测影响
- 解析准确后召回提升 (避免误把双向分离到不同尺寸)。
- Edge-F1 不受影响（bbox不变）。
- 评价指标新增：`dual_tolerance_accuracy = correct_dual / total_dual`。

### 风险与缓解
- 错误绑定到错误 nominal：添加距离阈值 + 排除跨行 (可利用换行计数)。
- 公差顺序错配：若在同一行出现多个 nominal + dual，对每个双向公差选择序号最近的 nominal。

### v2 拓展
- 支持复合标注: `Φ20 (+0.02/-0.01)` 括号形式。
- 支持范围表达: `20 H7` (标准公差代号) 映射到数值区间表。

