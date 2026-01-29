# 材料数据库统计报告

## 概述

- **总材料数**: 270
- **生成日期**: 2026-01-28
- **数据文件**: `src/core/materials/classifier.py`

## 属性覆盖率

| 属性 | 覆盖数 | 覆盖率 |
|------|--------|--------|
| density (密度) | 268/270 | 99.3% |
| melting_point (熔点) | 268/270 | 99.3% |
| thermal_conductivity (导热系数) | 268/270 | 99.3% |
| tensile_strength (抗拉强度) | 253/270 | 93.7% |
| yield_strength (屈服强度) | 236/270 | 87.4% |
| elongation (延伸率) | 176/270 | 65.2% |
| hardness (硬度) | 255/270 | 94.4% |

## 材料分类分布

### 按大类 (Category)
| 类别 | 数量 |
|------|------|
| metal | 228 |
| non_metal | 37 |
| composite | 5 |

### 按材料组 (Group) - Top 20
| 材料组 | 数量 |
|--------|------|
| alloy_steel | 19 |
| stainless_steel | 16 |
| corrosion_resistant | 16 |
| aluminum | 13 |
| carbon_steel | 11 |
| copper | 9 |
| engineering_plastic | 9 |
| tool_steel | 8 |
| cast_iron | 7 |
| titanium | 6 |
| powder_metallurgy | 5 |
| electrical_steel | 5 |
| cemented_carbide | 4 |
| refractory_metal | 4 |
| electrical_contact | 4 |
| precision_alloy | 4 |
| welding_material | 4 |
| bearing_steel | 3 |
| gear_steel | 3 |
| spring_steel | 3 |

## 成本等级分布

| 等级 | 描述 | 材料数 |
|------|------|--------|
| 1 | 低成本/通用材料 | 25 |
| 2 | 中低成本/合金材料 | 94 |
| 3 | 中等成本/特种钢 | 63 |
| 4 | 高成本/精密合金 | 55 |
| 5 | 极高成本/稀有材料 | 31 |

## 数据质量说明

1. **yield_strength 缺失材料** (17个): 灰铸铁、耐磨铸铁、永磁材料、超导体、电池材料、复合材料、导热垫片等脆性/粉末/纤维材料，这类材料技术上没有明确的屈服点。

2. **组合件材料** (2个): 组焊件、组合件是装配件，没有固定的物理属性和成本数据。

3. **成本数据**: 以 Q235B = 1.0 为基准，cost_index 反映相对成本比例。

## 详细数据

详见 `material_database_report.csv` 文件。
