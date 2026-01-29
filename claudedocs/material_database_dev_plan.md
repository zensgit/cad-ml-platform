# 材料数据库开发及验证计划

## 一、项目概述

**目标**: 完善 CAD-ML 平台材料数据库，提升数据覆盖率和质量

**当前状态** (2026-01-28):
- 总材料数: 270
- elongation 覆盖率: 65.2% (176/270)
- 其他主要属性覆盖率: 87-99%

**目标状态**:
- 总材料数: 300+ (新增 30+ 材料)
- elongation 覆盖率: 85%+
- 通过所有代码质量检查

---

## 二、开发计划

### 阶段 1: 提升 elongation 覆盖率 (目标 85%+)

#### 1.1 待添加材料分析

当前缺失 elongation 的材料 (94个):

| 材料组 | 数量 | 典型材料 | 典型 elongation |
|--------|------|----------|-----------------|
| corrosion_resistant | ~11 | Inconel600, Monel K-500 | 20-50% |
| stainless_steel | ~5 | 317L, 904L | 35-45% |
| superalloy | ~3 | K403, K418 | 5-20% |
| cast_aluminum | ~3 | ZL101, ZL104, ZL201 | 1-5% |
| tin_bronze | ~3 | QSn7-0.2, QSn4-0.3 | 3-15% |
| permanent_magnet | 3 | NdFeB, SmCo, Alnico | 脆性材料 |
| superconductor | ~2 | Nb3Sn, YBCO | 脆性材料 |
| battery_material | 3 | LiFePO4, NMC | 粉末材料 |
| structural_ceramic | 3 | Al2O3-99, Si3N4, ZrO2-3Y | 脆性材料 |
| composite | 2 | CFRP, GFRP | 纤维材料 |
| 其他 | ~55 | 各类金属/合金 | 按材料特性 |

#### 1.2 实施策略

1. **金属材料**: 添加标准延伸率数据
2. **脆性材料** (陶瓷、硬质合金、永磁): 设置 elongation=0.5% 或保持 None 并添加注释
3. **粉末/纤维材料**: 保持 None 并添加注释说明

#### 1.3 验证标准

- [ ] elongation 覆盖率 ≥ 85%
- [ ] 所有数据在合理范围内 (金属 0.5-60%, 聚合物 2-500%, 橡胶 100-800%)
- [ ] 脆性材料有适当注释

---

### 阶段 2: 添加更多材料牌号 (目标 300+)

#### 2.1 新增材料规划

| 类别 | 新增数量 | 重点材料 |
|------|----------|----------|
| 不锈钢 | 5 | 631, 15-5PH, 13-8Mo, Nitronic 50, 22Cr双相钢 |
| 铝合金 | 5 | 6005, 6101, 3003, 1060, LY12 |
| 铜合金 | 3 | C11000, C26000, C52100 |
| 工具钢 | 3 | D2, A2, O1 |
| 高温合金 | 3 | Waspaloy, Rene 41, Haynes 230 |
| 钛合金 | 2 | Ti-6242, Ti-5553 |
| 镁合金 | 2 | WE43, AM60 |
| 工程塑料 | 3 | PPSU, LCP, PVDF |
| 橡胶 | 2 | NBR, FKM |
| 特种材料 | 5 | Hastelloy X, MP35N, L605, Elgiloy, Phynox |
| **合计** | **33** | |

#### 2.2 数据要求

每个新材料必须包含:
- 基本信息: grade, name, aliases, category, sub_category, group
- 物理属性: density, melting_point, thermal_conductivity
- 机械属性: tensile_strength, yield_strength, elongation (如适用), hardness
- 加工属性: machinability, weldability
- 成本数据: cost_tier, cost_index

#### 2.3 验证标准

- [ ] 总材料数 ≥ 300
- [ ] 新材料 100% 属性完整
- [ ] 数据来源可靠 (标准/手册)

---

### 阶段 3: 代码质量检查

#### 3.1 检查项目

| 检查类型 | 工具/方法 | 通过标准 |
|----------|-----------|----------|
| 语法检查 | Python import | 无 SyntaxError |
| 数据一致性 | 自定义脚本 | yield ≤ tensile, 范围合理 |
| 单元测试 | pytest | 所有测试通过 |
| 类型检查 | mypy (可选) | 无严重错误 |
| 代码风格 | ruff/flake8 (可选) | 无严重警告 |

#### 3.2 数据一致性检查规则

```python
# 检查规则
1. yield_strength <= tensile_strength (当两者都存在时)
2. density: 0.1 - 23 g/cm³
3. melting_point: 50 - 4000 °C
4. thermal_conductivity: 0.01 - 2500 W/m·K
5. tensile_strength: 0 - 5000 MPa
6. yield_strength: 0 - 5000 MPa
7. elongation: 0 - 1000 %
```

#### 3.3 验证标准

- [ ] 模块可正常导入
- [ ] 数据一致性检查 0 错误
- [ ] 单元测试全部通过

---

## 三、验证计划

### 3.1 验证流程

```
┌─────────────────┐
│ 代码修改完成    │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 语法检查        │ → 失败 → 修复
└────────┬────────┘
         ▼
┌─────────────────┐
│ 数据一致性检查  │ → 失败 → 修复
└────────┬────────┘
         ▼
┌─────────────────┐
│ 单元测试        │ → 失败 → 修复
└────────┬────────┘
         ▼
┌─────────────────┐
│ 覆盖率统计      │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 生成验证报告    │
└─────────────────┘
```

### 3.2 验收标准汇总

| 指标 | 当前值 | 目标值 | 优先级 |
|------|--------|--------|--------|
| 总材料数 | 270 | 300+ | 高 |
| elongation 覆盖率 | 65.2% | 85%+ | 高 |
| density 覆盖率 | 99.3% | 99%+ | 维持 |
| melting_point 覆盖率 | 99.3% | 99%+ | 维持 |
| thermal_conductivity 覆盖率 | 99.3% | 99%+ | 维持 |
| tensile_strength 覆盖率 | 93.7% | 95%+ | 中 |
| yield_strength 覆盖率 | 87.4% | 90%+ | 中 |
| 单元测试 | - | 全部通过 | 高 |
| 数据一致性 | - | 0 错误 | 高 |

---

## 四、时间安排

| 阶段 | 任务 | 预估工作量 |
|------|------|------------|
| 1 | elongation 覆盖率提升 | 批量更新 ~95 个材料 |
| 2 | 新增 33+ 材料 | 逐个添加并验证 |
| 3 | 代码质量检查 | 运行测试并修复问题 |
| 4 | 生成报告 | 汇总统计并输出 |

---

## 五、交付物

1. **更新后的代码**: `src/core/materials/classifier.py`
2. **开发及验证报告**: `claudedocs/material_database_dev_report.md`
3. **数据导出**: `claudedocs/material_database_report.csv` (更新)
4. **统计摘要**: `claudedocs/material_database_summary.md` (更新)

---

*计划制定日期: 2026-01-28*
