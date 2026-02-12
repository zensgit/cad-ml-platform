# Tolerance Knowledge Roadmap (公差知识落地清单)

## 当前状态（已完成）
- [x] 设计文档：`docs/TOLERANCE_KNOWLEDGE_DESIGN.md`
- [x] 精度种子规则：`data/knowledge/precision_rules.json`
- [x] 基本偏差查询：`get_fundamental_deviation`
- [x] 助手检索接入 GB/T 1804/1184
- [x] 基本偏差查询单测与检索单测

## 近期开发（1-2 天）
- [ ] 生成 `iso286_deviations.json`（孔+轴）并加入来源/版本字段
- [ ] 扩展 `fits.py` 读取新文件（优先新文件、回退旧文件）
- [ ] 添加 `H7 25mm` / `g6 10mm` 结果展示到 /assistant

## 中期开发（3-5 天）
- [ ] PDF 表格抽取脚本（GB/T 1800.2-2020）
- [ ] 数据校验脚本（表格完整性、数值边界）
- [ ] 端到端契约测试（/assistant/query）

## 风险与回滚
- 风险：PDF 表格抽取误差导致偏差表不准
- 回滚：保留旧 `iso286_hole_deviations.json` + 内置偏差

## 验证清单
- [ ] IT7@25mm 结果一致性验证
- [ ] H7/g6@25mm 计算间隙验证
- [ ] GB/T 1804 / 1184 文本命中验证
