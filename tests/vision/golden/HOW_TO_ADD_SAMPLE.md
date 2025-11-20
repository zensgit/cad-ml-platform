# How to Add a Golden Sample

**Purpose**: 指导如何新增 Vision Golden 评估样本

---

## 样本命名规范

### 文件命名
```
tests/vision/golden/annotations/sample_{序号}_{难度}.json

序号: 001, 002, 003, ... (三位数，左补零)
难度: easy, medium, hard, edge_case
```

### 示例
- `sample_001_easy.json` - 简单样本
- `sample_002_medium.json` - 中等难度样本
- `sample_003_hard.json` - 困难样本
- `sample_004_edge_case.json` - 边界情况样本

---

## Annotation Schema

### 必需字段
```json
{
  "id": "sample_XXX_difficulty",
  "description": "样本简要描述",
  "difficulty": "easy|medium|hard|edge_case",
  "expected_keywords": [
    "keyword1",
    "keyword2",
    ...
  ]
}
```

### 可选字段
```json
{
  "expected_category": "mechanical_part|assembly|detail_view",
  "expected_features": [
    "feature1",
    "feature2"
  ],
  "notes": "设计说明或特殊备注"
}
```

---

## 如何设计 expected_keywords

### Step 1: 了解当前 Provider 响应

**Stub Provider 固定响应**（参考 `src/core/vision/providers/deepseek_stub.py`）:
```
Summary:
"This is a mechanical engineering drawing showing a cylindrical part with threaded features."

Details:
- Main body features a diameter dimension of approximately 20mm with bilateral tolerance
- External thread specification visible (M10×1.5 pitch)
- Surface finish requirement indicated (Ra 3.2 or similar)
- Title block present with drawing number and material specification
```

**Stub Provider 包含的关键词**:
- cylindrical, thread/threaded, diameter, mechanical, engineering
- dimension, tolerance, specification, surface, finish, title, material

### Step 2: 根据目标 Hit Rate 设计关键词

| 难度 | 目标 Hit Rate | 策略 |
|------|---------------|------|
| **easy** | 80-100% | 全部或大部分关键词在 stub 响应中 |
| **medium** | 50-70% | 一半关键词在 stub 响应中 |
| **hard** | 20-40% | 少量关键词在 stub 响应中 |
| **edge_case** | 0-10% | 几乎所有关键词不在 stub 响应中 |

### Step 3: 示例设计

**Easy Sample (目标 100%)**:
```json
{
  "expected_keywords": [
    "cylindrical",
    "thread",
    "diameter",
    "mechanical",
    "engineering"
  ]
}
```
预期结果: 5/5 (100%)

**Medium Sample (目标 60%)**:
```json
{
  "expected_keywords": [
    "cylindrical",    // ✅ 在 stub 中
    "threaded",       // ✅ 在 stub 中
    "precision",      // ❌ 不在 stub 中
    "diameter",       // ✅ 在 stub 中
    "fastener"        // ❌ 不在 stub 中
  ]
}
```
预期结果: 3/5 (60%)

**Hard Sample (目标 40%)**:
```json
{
  "expected_keywords": [
    "mechanical",     // ✅ 在 stub 中
    "engineering",    // ✅ 在 stub 中
    "assembly",       // ❌ 不在 stub 中
    "bearing",        // ❌ 不在 stub 中
    "shaft"           // ❌ 不在 stub 中
  ]
}
```
预期结果: 2/5 (40%)

---

## 添加流程

### 1. 创建 Annotation 文件
```bash
cd tests/vision/golden/annotations/

# 复制现有样本作为模板
cp sample_001_easy.json sample_004_new.json

# 编辑新样本
vim sample_004_new.json
```

### 2. 填写字段
```json
{
  "id": "sample_004_edge_case",
  "description": "Edge case with no keyword matches",
  "difficulty": "edge_case",
  "expected_keywords": [
    "hydraulic",
    "pneumatic",
    "electrical",
    "control",
    "sensor"
  ],
  "notes": "Designed for 0% hit rate - all keywords absent from stub response"
}
```

### 3. 验证样本
```bash
# Dry-run 检查样本是否被检测到
python3 scripts/evaluate_vision_golden.py --dry-run

# 运行评估
make eval-vision-golden

# 检查结果是否符合预期 hit_rate
```

### 4. 验证测试
```bash
# 确保所有 Vision 测试仍然通过
pytest tests/vision -v
```

---

## 常见场景

### 场景 1: 验证 Provider 改动
**目的**: 确保 provider 改动不破坏现有能力

**操作**:
1. 运行 baseline 评估，记录结果
2. 修改 provider 代码
3. 再次运行评估，对比变化
4. 如果 hit_rate 显著下降，检查改动

### 场景 2: 扩展难度覆盖
**目的**: 增加某个难度级别的样本数量

**建议**:
- Easy: 2-3 个即可（验证基本能力）
- Medium: 3-5 个（主要测试场景）
- Hard: 2-3 个（挑战性场景）
- Edge Case: 1-2 个（边界验证）

### 场景 3: 测试特定 Feature
**目的**: 针对某个特定特征（如螺纹、孔、表面处理）

**操作**:
1. 设计关键词围绕该特征
2. 创建 2-3 个样本测试该特征
3. 用 expected_features 字段标注
4. 分析该特征的识别准确率

---

## 质量检查清单

新增样本前，确认：
- [ ] 文件命名符合规范（sample_XXX_difficulty.json）
- [ ] ID 与文件名一致
- [ ] difficulty 字段正确（easy/medium/hard/edge_case）
- [ ] expected_keywords 至少有 3 个
- [ ] 预期 hit_rate 符合难度定义
- [ ] 运行 `make eval-vision-golden` 验证
- [ ] 所有 Vision 测试通过（pytest tests/vision -v）

---

## 注意事项

### ⚠️ 避免的陷阱

1. **关键词过于具体**
   - ❌ "M10×1.5 thread pitch"
   - ✅ "thread", "pitch"
   - 原因: Stub provider 不会有这么具体的响应

2. **关键词重复**
   - ❌ ["thread", "threaded", "threading"]
   - ✅ ["thread", "diameter", "tolerance"]
   - 原因: 计算 hit_rate 时会有冗余

3. **难度分级混乱**
   - ❌ Hard 样本的 hit_rate > Easy 样本
   - ✅ 保持 Easy > Medium > Hard 的顺序
   - 原因: 保证评估结果可解释性

### 💡 最佳实践

1. **先预估，后验证**
   - 设计关键词时先估算 hit_rate
   - 运行评估后对比实际结果
   - 如有偏差，调整关键词

2. **保持样本独立性**
   - 每个样本测试不同的方面
   - 避免多个样本测试相同内容

3. **文档化设计意图**
   - 在 notes 字段说明为什么选择这些关键词
   - 方便未来维护和理解

---

## 未来扩展

### 真实图像样本（待 Stage B 实施）
当前 Stage A/B.1 使用 in-memory fixture，未来可扩展：

```
tests/vision/golden/
├── samples/              # 真实图像文件
│   ├── sample_001_easy.png
│   ├── sample_002_medium.png
│   └── ...
├── annotations/          # Annotation JSON
│   ├── sample_001_easy.json
│   └── ...
└── HOW_TO_ADD_SAMPLE.md  # 本文档
```

### Metadata 管理（待 Stage B.2 实施）
```yaml
# tests/vision/golden/metadata.yaml
samples:
  - id: sample_001_easy
    difficulty: easy
    category: mechanical_part
    features: [center_hole, outer_thread]
  - id: sample_002_medium
    ...
```

---

**Last Updated**: 2025-01-16
**Related Docs**:
- `docs/ocr/VISION_GOLDEN_STAGE_A_COMPLETE.md`
- `docs/ocr/VISION_GOLDEN_STAGE_B1_COMPLETE.md`
- `reports/vision_golden_baseline.md`
