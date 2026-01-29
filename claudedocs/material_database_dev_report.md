# 材料数据库开发及验证报告

## 一、项目概述

- **报告日期**: 2026-01-29
- **项目目标**: 完善 CAD-ML 平台材料数据库
- **数据文件**: `src/core/materials/classifier.py`

---

## 二、开发成果

### 2.1 总体统计

| 指标 | 初始值 | 目标值 | 最终值 | 状态 |
|------|--------|--------|--------|------|
| 总材料数 | 270 | 300+ | **300** | ✅ 达标 |
| elongation 覆盖率 | 65.2% | 85%+ | **89.7%** | ✅ 达标 |
| density 覆盖率 | 99.3% | 99%+ | **99.3%** | ✅ 维持 |
| melting_point 覆盖率 | 99.3% | 99%+ | **99.3%** | ✅ 维持 |
| thermal_conductivity 覆盖率 | 99.3% | 99%+ | **99.3%** | ✅ 维持 |
| tensile_strength 覆盖率 | 93.7% | 95%+ | **96.7%** | ✅ 达标 |
| yield_strength 覆盖率 | 87.4% | 90%+ | **91.0%** | ✅ 达标 |
| hardness 覆盖率 | 94.4% | 95%+ | **95.0%** | ✅ 达标 |
| cost_data 覆盖率 | 98.5% | 99%+ | **100%** | ✅ 达标 |

### 2.2 新增材料列表 (30 种)

**第一批 (17 种)**:
| 材料牌号 | 名称 | 类别 |
|----------|------|------|
| 631 | 沉淀硬化不锈钢 | stainless_steel |
| 15-5PH | 沉淀硬化不锈钢 | stainless_steel |
| Nitronic50 | 高氮奥氏体不锈钢 | stainless_steel |
| 3003 | 防锈铝合金 | aluminum |
| 1060 | 工业纯铝 | aluminum |
| LY12 | 硬铝合金 | aluminum |
| D2 | 高碳高铬冷作模具钢 | tool_steel |
| O1 | 油淬冷作模具钢 | tool_steel |
| Haynes230 | 镍铬钨钼高温合金 | superalloy |
| Ti-6242 | 近α型钛合金 | titanium |
| PVDF | 聚偏氟乙烯 | fluoropolymer |
| LCP | 液晶聚合物 | engineering_plastic |
| NBR | 丁腈橡胶 | rubber |
| FKM | 氟橡胶 | rubber |
| HastelloyX | 哈氏合金X | superalloy |
| MP35N | 钴镍钼钛合金 | superalloy |
| L605 | 钴基高温合金 | superalloy |

**第二批 (13 种)**:
| 材料牌号 | 名称 | 类别 |
|----------|------|------|
| C11000 | 电解铜 | copper |
| C26000 | 黄铜 | copper |
| C52100 | 磷青铜 | tin_bronze |
| WE43 | 稀土镁合金 | magnesium |
| AM60 | 压铸镁合金 | cast_magnesium |
| 13-8Mo | 沉淀硬化不锈钢 | stainless_steel |
| 22Cr双相钢 | 双相不锈钢 | stainless_steel |
| A2 | 空淬工具钢 | tool_steel |
| PPSU | 聚苯砜 | engineering_plastic |
| Waspaloy | 镍基时效高温合金 | superalloy |
| Rene41 | 镍基高温合金 | superalloy |
| Elgiloy | 钴铬镍钼合金 | superalloy |
| Phynox | 医用钴基合金 | superalloy |
| 6005 | 轨道交通铝合金 | aluminum |
| 6101 | 导电铝合金 | aluminum |

### 2.3 材料分类分布

**按大类 (Category)**:
| 类别 | 数量 | 占比 |
|------|------|------|
| metal | 253 | 84.3% |
| non_metal | 42 | 14.0% |
| composite | 5 | 1.7% |

**按材料组 (Top 15)**:
| 材料组 | 数量 |
|--------|------|
| stainless_steel | 21 |
| alloy_steel | 19 |
| aluminum | 18 |
| corrosion_resistant | 14 |
| carbon_steel | 11 |
| tool_steel | 11 |
| superalloy | 11 |
| copper | 11 |
| engineering_plastic | 11 |
| cast_iron | 7 |
| titanium | 7 |
| powder_metallurgy | 5 |
| electrical_steel | 5 |
| tin_bronze | 4 |
| cast_magnesium | 4 |

### 2.4 成本等级分布

| 等级 | 描述 | 材料数 | 占比 |
|------|------|--------|------|
| 1 | 低成本/通用材料 | 28 | 9.3% |
| 2 | 中低成本/合金材料 | 102 | 34.0% |
| 3 | 中等成本/特种钢 | 73 | 24.3% |
| 4 | 高成本/精密合金 | 60 | 20.0% |
| 5 | 极高成本/稀有材料 | 37 | 12.3% |

---

## 三、验证结果

### 3.1 语法检查

```
✓ Python 模块导入成功
✓ MATERIAL_DATABASE 加载正常 (300 条记录)
✓ MATERIAL_COST_DATA 加载正常 (300 条记录)
```

### 3.2 数据一致性检查

| 检查项 | 结果 |
|--------|------|
| yield_strength ≤ tensile_strength | ✅ 通过 |
| density 范围 (0.1-23 g/cm³) | ✅ 通过 |
| melting_point 范围 (50-4000 °C) | ✅ 通过 |
| thermal_conductivity 范围 (0.01-2500 W/m·K) | ✅ 通过 |
| tensile_strength 范围 (0-5000 MPa) | ✅ 通过 |
| elongation 范围 (0-1000 %) | ✅ 通过 |

### 3.3 数据质量说明

1. **yield_strength 缺失材料**: 脆性材料（灰铸铁、陶瓷、硬质合金等）技术上没有明确屈服点，设置 elongation≈0.5% 表示脆性。

2. **组合件材料**: 组焊件、组合件是装配件，没有固定的物理属性和成本数据。

3. **成本基准**: Q235B = 1.0，cost_index 反映相对成本比例。

---

## 四、交付物清单

| 文件 | 说明 |
|------|------|
| `src/core/materials/classifier.py` | 更新后的材料数据库代码 |
| `claudedocs/material_database_dev_plan.md` | 开发计划文档 |
| `claudedocs/material_database_dev_report.md` | 本验证报告 |
| `claudedocs/material_database_report.csv` | 完整数据 CSV 导出 |
| `claudedocs/material_database_summary.md` | 数据统计摘要 |

---

## 五、总结

本次开发工作成功完成以下目标:

1. ✅ **总材料数**: 270 → 300 (增加 30 种)
2. ✅ **elongation 覆盖率**: 65.2% → 89.7% (提升 24.5 个百分点)
3. ✅ **tensile_strength 覆盖率**: 93.7% → 96.7% (提升 3.0 个百分点)
4. ✅ **yield_strength 覆盖率**: 87.4% → 91.0% (提升 3.6 个百分点)
5. ✅ **hardness 覆盖率**: 94.4% → 95.0% (达到目标)
6. ✅ **数据一致性**: 全部通过验证
7. ✅ **成本数据**: 覆盖率达 100%

---

*报告生成时间: 2026-01-29*
