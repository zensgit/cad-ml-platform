# Tolerance Knowledge Design (公差知识设计)

## 目标
构建可查询、可扩展、可追溯的公差知识模块，覆盖 ISO 286 / GB/T 1800.2 的孔轴公差带与 IT 等级值，支持：
- 智能助手查询（IT 公差、配合、推荐）
- 图纸 OCR 提取后的公差标注解析
- 规则/知识库驱动的精度推断与分类提示

## 现状盘点
- 现有模块：`src/core/knowledge/tolerance` 提供 IT 等级、配合计算与部分基本偏差。
- 数据来源：`data/knowledge/iso286_hole_deviations.json`（孔 EI 表）
- 现有接口：
  - `get_tolerance_value` / `get_tolerance_table`
  - `get_fit_deviations` / `get_common_fits`
- 动态知识体系已有 PrecisionRule 结构，但暂无精度规则种子文件。

## 设计原则
1. **数据分层**：表格数据与算法逻辑分离，便于更新与校验。
2. **可追溯**：每个表格数据标注来源与版本。
3. **兼容现有**：不破坏已上线接口与默认行为。
4. **防故障**：容错缺失字段或空数据文件。

## 数据模型设计
新增统一数据文件（替代或扩展现有孔偏差文件）：

```json
{
  "source": "GB/T 1800.2-2020 Table 2–16",
  "version": "2026-02-04",
  "units": "um",
  "holes": {
    "H": [[3, 0], [6, 0], [10, 0], ...],
    "G": [[3, 2], [6, 4], ...]
  },
  "shafts": {
    "h": [[3, 0], [6, 0], ...],
    "g": [[3, -2], [6, -4], ...]
  }
}
```

- 文件建议：`data/knowledge/iso286_deviations.json`
- 保留现有 `iso286_hole_deviations.json` 以保证兼容，读取优先级：新文件 > 旧文件 > 内置常量。

## 代码改造点
1. **读取器扩展**（`src/core/knowledge/tolerance/fits.py`）
   - 加载 `iso286_deviations.json`（holes/shafts）
   - 若缺失则回退到 `iso286_hole_deviations.json`
2. **基本偏差 API**
   - 提供 `get_fundamental_deviation(symbol, size)` 返回 EI/ES
3. **容差查询增强**（`src/core/assistant/knowledge_retriever.py`）
   - 支持 “H7/g6 25mm”、“h6 10mm” 等查询
4. **动态精度规则**
   - 增加 `data/knowledge/precision_rules.json`（根据 OCR 文本如 “GB/T 1804-M” 提供精度线索）

## OCR/文本识别策略
- 识别 “未注公差 按 GB/T 1804-M” → 建议公差等级/精度
- 识别 “H7/g6” → 拆分 hole/shaft symbol + grade
- 识别 “IT7 25mm” → 直接查询 IT 值

## MVP 开发步骤
1. **数据层**
   - 构建 `iso286_deviations.json`（孔+轴），保留来源/版本字段
2. **接口层**
   - 扩展 `fits.py` 加载新文件
   - 新增 `get_fundamental_deviation`
3. **助手/检索层**
   - 添加配合与公差查询路径
4. **动态规则**
   - 新增 `precision_rules.json` 种子

## 验证计划
- 单元测试：
  - IT7@25mm 返回非空
  - H7/g6@25mm 计算最小/最大间隙
  - `get_fundamental_deviation("g", 10)` 正常返回
- 契约测试：
  - /assistant/query 对 “IT7公差在25mm时的值是多少？” 返回包含 μm
- 数据校验：
  - JSON 结构完整、无空文件

## 风险与回滚
- 风险：PDF 表格抽取偏差导致数值错误
- 回滚：继续使用旧 `iso286_hole_deviations.json` 与内置偏差

## 下一步
- 如果你确认数据来源 PDF 可用，将实现抽取脚本并生成新数据文件。
- 之后进入接口扩展与测试阶段。
