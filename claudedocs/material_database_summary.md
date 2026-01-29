# 材料数据库统计报告

## 概述

- **总材料数**: 300
- **生成日期**: 2026-01-29
- **数据文件**: `src/core/materials/classifier.py`

## 属性覆盖率

| 属性 | 覆盖数 | 覆盖率 |
|------|--------|--------|
| density (密度) | 298/300 | 99.3% |
| melting_point (熔点) | 298/300 | 99.3% |
| thermal_conductivity (导热系数) | 298/300 | 99.3% |
| tensile_strength (抗拉强度) | 294/300 | 98.0% |
| yield_strength (屈服强度) | 277/300 | 92.3% |
| elongation (延伸率) | 269/300 | 89.7% |
| hardness (硬度) | 285/300 | 95.0% |

## 材料分类分布

### 按大类 (Category)
| 类别 | 数量 | 占比 |
|------|------|------|
| metal | 253 | 84.3% |
| non_metal | 42 | 14.0% |
| composite | 5 | 1.7% |

### 按材料组 (Group) - Top 15
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

## 成本等级分布

| 等级 | 描述 | 材料数 | 占比 |
|------|------|--------|------|
| 1 | 低成本/通用材料 | 28 | 9.3% |
| 2 | 中低成本/合金材料 | 102 | 34.0% |
| 3 | 中等成本/特种钢 | 73 | 24.3% |
| 4 | 高成本/精密合金 | 60 | 20.0% |
| 5 | 极高成本/稀有材料 | 37 | 12.3% |

## 数据质量说明

1. **yield_strength 缺失材料** (23个): 灰铸铁、耐磨铸铁、永磁材料、超导体、电池材料、复合材料、导热垫片、玻璃/陶瓷等脆性/粉末/纤维材料，这类材料技术上没有明确的屈服点。

2. **组合件材料** (2个): 组焊件、组合件是装配件，没有固定的物理属性和成本数据。

3. **成本数据**: 以 Q235B = 1.0 为基准，cost_index 反映相对成本比例。

## 详细数据

详见 `material_database_report.csv` 文件。
