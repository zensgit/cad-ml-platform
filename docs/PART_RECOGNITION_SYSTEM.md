# 部件识别系统详解

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CAD 文件输入                                    │
│                         (DXF, DWG, STEP, IGES)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            第一阶段：信号提取                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   几何特征提取   │  │   实体统计分析   │  │    OCR 文本识别   │             │
│  │                 │  │                 │  │                 │             │
│  │ • 长宽比方差    │  │ • CIRCLE 数量   │  │ • 尺寸标注      │             │
│  │ • 球度指数     │  │ • LINE 数量     │  │ • 材料标注      │             │
│  │ • 紧凑度      │  │ • ARC 数量      │  │ • 公差标注      │             │
│  │ • 复杂度分数   │  │ • SPLINE 数量   │  │ • GD&T 符号     │             │
│  │ • 圆弧比例    │  │ • TEXT 数量     │  │ • 标题栏信息    │             │
│  │ • 实体熵      │  │ • ...           │  │ • 标准代号      │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
└───────────┼────────────────────┼────────────────────┼───────────────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          第二阶段：多模态融合分类                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    EnhancedPartClassifier                            │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │              基础分类器 (MechanicalPartKnowledgeBase)           │   │   │
│  │  │                        权重: 30%                               │   │   │
│  │  │                                                                │   │   │
│  │  │  几何模式 (40%)  +  实体模式 (30%)  +  OCR模式 (30%)           │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                       │   │
│  │  ┌───────────────┬───────────┼───────────┬───────────────┐          │   │
│  │  ▼               ▼           ▼           ▼               ▼          │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │ │ 材料KB  │ │ 精度KB  │ │ 标准KB  │ │ 功能KB  │ │ 装配KB  │        │   │
│  │ │  12%   │ │  10%   │ │  13%   │ │  10%   │ │   8%   │        │   │
│  │ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  │                                                                      │   │
│  │ ┌─────────┐ ┌─────────┐                                              │   │
│  │ │ 制造KB  │ │ 几何KB  │  ← 动态知识库 (JSON配置)                     │   │
│  │ │  10%   │ │   7%   │                                              │   │
│  │ └─────────┘ └─────────┘                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           第三阶段：输出结果                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  {                                                                          │
│    "part_type": "bearing",           // 识别的零件类型                      │
│    "display_name": "轴承零件",        // 中文显示名                          │
│    "confidence": 0.85,               // 置信度                              │
│    "alternatives": [                 // 备选分类                            │
│      {"part_type": "shaft", "confidence": 0.12},                           │
│      {"part_type": "coupling", "confidence": 0.08}                         │
│    ],                                                                       │
│    "knowledge_source": "dynamic",    // 知识来源                            │
│    "knowledge_version": "2025-11-30" // 知识版本                            │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 信号提取详解

### 2.1 几何特征 (Geometric Features)

从 CAD 文件提取的几何特征向量，用于描述零件的形状特性：

| 特征名 | 描述 | 计算方式 | 典型值范围 |
|--------|------|----------|-----------|
| `aspect_variance` | 长宽比方差 | 边界框各轴尺寸的方差 | 0.0 ~ 1.0 |
| `sphericity` | 球度指数 | 形状接近球体的程度 | 0.0 ~ 1.0 |
| `compactness` | 紧凑度 | 实体面积/边界框面积 | 0.0 ~ 1.0 |
| `complexity_score` | 复杂度分数 | 实体数量和类型的综合评估 | 0.0 ~ 1.0 |
| `circle_ratio` | 圆形比例 | 圆形实体数/总实体数 | 0.0 ~ 1.0 |
| `arc_ratio` | 圆弧比例 | 圆弧实体数/总实体数 | 0.0 ~ 1.0 |
| `line_ratio` | 直线比例 | 直线实体数/总实体数 | 0.0 ~ 1.0 |
| `entity_entropy` | 实体熵 | 实体类型分布的多样性 | 0.0 ~ 1.0 |

**特征版本:**
- v1: 7维基础几何特征
- v2: 12维 (+ 归一化尺寸 + 长宽比)
- v3: 22维 (+ 实体类型频率)
- v4: 24维 (+ 表面数 + 形状熵)
- v5: 26维 (旋转/缩放不变)
- v6: 32维 (矩不变量)

### 2.2 实体统计 (Entity Counts)

CAD 文件中各类几何实体的数量统计：

```python
entity_counts = {
    "CIRCLE": 50,    # 圆形数量
    "LINE": 120,     # 直线数量
    "ARC": 30,       # 圆弧数量
    "ELLIPSE": 0,    # 椭圆数量
    "SPLINE": 5,     # 样条曲线数量
    "POLYLINE": 10,  # 多段线数量
    "TEXT": 8,       # 文本数量
    "DIMENSION": 15, # 尺寸标注数量
    "HATCH": 3,      # 填充区域数量
}
```

### 2.3 OCR 数据 (OCR Data)

从图纸中识别的文本信息：

```python
ocr_data = {
    "text": "GCr15 轴承钢 HRC60-65",  # 原始文本
    "dimensions": [                    # 尺寸标注
        {"value": 50.0, "tolerance": "±0.02"},
        {"value": 30.0, "tolerance": "H7"}
    ],
    "symbols": ["⌀", "Ra0.8", "//0.02"],  # GD&T符号
    "title_block": {                    # 标题栏
        "material": "GCr15",
        "part_name": "滚动轴承",
        "standard": "GB/T 276-2013"
    }
}
```

## 3. 知识模块详解

### 3.1 基础分类器 (MechanicalPartKnowledgeBase)

**权重: 30%**

基于专家规则的模式匹配：

```python
# 轴类零件识别规则示例
shaft_patterns = {
    "geometric": [
        ("high_aspect_variance", lambda f: f.get("aspect_variance", 0) > 0.25, 0.3),
        ("medium_sphericity", lambda f: 0.5 < f.get("sphericity", 0) < 0.8, 0.2),
    ],
    "entity": [
        ("has_circles", lambda e: e.get("CIRCLE", 0) > 5, 0.25),
        ("line_dominant", lambda e: e.get("LINE", 0) > e.get("ARC", 0), 0.15),
    ],
    "ocr": [
        ("shaft_keyword", lambda o: "轴" in o.get("text", ""), 0.3),
        ("turning_process", lambda o: "车削" in o.get("text", ""), 0.2),
    ]
}
```

**内部权重分配:**
- 几何模式: 40%
- 实体模式: 30%
- OCR模式: 30%

### 3.2 材料知识库 (MaterialKnowledgeBase)

**权重: 12%**

根据材料标注推断零件类型：

| 材料关键词 | 推断零件类型 | 置信度 |
|-----------|-------------|--------|
| GCr15, SUJ2, 52100 | bearing (轴承) | 0.85 |
| 20CrMnTi, 17CrNiMo6 | gear (齿轮) | 0.80 |
| 65Mn, 60Si2Mn | spring (弹簧) | 0.90 |
| 45#, Q235 | shaft (轴) | 0.30 |
| 铸铁, HT200 | housing (壳体) | 0.70 |

**几何推断规则:**
- 高球度 + 低长宽比方差 → 轴承材料可能性高
- 高长宽比方差 + 中等复杂度 → 轴类材料可能性高

### 3.3 精度知识库 (PrecisionKnowledgeBase)

**权重: 10%**

根据公差和表面粗糙度推断零件类型：

| 精度特征 | 推断零件类型 | 置信度 |
|---------|-------------|--------|
| IT5-IT6 + Ra0.2-0.4 | bearing (轴承) | 0.70 |
| IT7-IT8 + Ra0.8-1.6 | shaft (轴) | 0.50 |
| 齿形公差 | gear (齿轮) | 0.60 |
| 平面度 < 0.01 | plate (板) | 0.40 |

### 3.4 标准知识库 (StandardsKnowledgeBase)

**权重: 13%**

根据图纸引用的标准代号推断：

| 标准代号模式 | 推断零件类型 | 置信度 |
|-------------|-------------|--------|
| GB/T 276, ISO 15 | bearing (轴承) | 0.90 |
| GB/T 1096, DIN 6885 | gear (齿轮) | 0.75 |
| GB/T 5782, ISO 4014 | bolt (螺栓) | 0.85 |
| GB/T 9112, ANSI B16.5 | flange (法兰) | 0.80 |

### 3.5 功能特征知识库 (FunctionalKnowledgeBase)

**权重: 10%**

根据功能特征推断零件类型：

| 功能特征 | 推断零件类型 | 置信度 |
|---------|-------------|--------|
| 键槽 (keyway) | shaft (轴) | 0.60 |
| 花键 (spline) | shaft (轴) / gear (齿轮) | 0.55 |
| 轴承座 (bearing_seat) | housing (壳体) | 0.50 |
| 螺纹孔 (threaded_hole) | housing (壳体) / flange (法兰) | 0.40 |
| 法兰面 (flange_face) | flange (法兰) | 0.70 |

### 3.6 装配知识库 (AssemblyKnowledgeBase)

**权重: 8%**

根据配合关系推断零件类型：

| 配合特征 | 推断关系 | 置信度 |
|---------|---------|--------|
| 过盈配合 H7/p6 | shaft ↔ bearing | 0.60 |
| 间隙配合 H7/g6 | shaft ↔ housing | 0.50 |
| 过渡配合 H7/k6 | gear ↔ shaft | 0.55 |

### 3.7 制造知识库 (ManufacturingKnowledgeBase)

**权重: 10%**

根据制造工艺标注推断零件类型：

| 工艺特征 | 推断零件类型 | 置信度 |
|---------|-------------|--------|
| 车削 + 磨削 | shaft (轴) | 0.50 |
| 滚齿 + 渗碳淬火 | gear (齿轮) | 0.65 |
| 超精磨 + 套圈加工 | bearing (轴承) | 0.70 |
| 铸造 + 机加工 | housing (壳体) | 0.60 |

### 3.8 几何模式知识库 (GeometryPatternKnowledgeBase) - 动态

**权重: 7%**

基于 JSON 配置的几何条件匹配（可热更新）：

```json
{
  "name": "bearing_geometry",
  "conditions": {
    "sphericity": {"min": 0.85},
    "aspect_variance": {"max": 0.15},
    "circle_ratio": {"min": 0.35}
  },
  "part_hints": {"bearing": 0.30}
}
```

## 4. 融合算法

### 4.1 分数计算

```python
# 对于每个零件类型，计算加权融合分数
final_score[part] = (
    base_score[part]         * 0.30 +  # 基础分类器
    material_hint[part]      * 0.12 +  # 材料知识
    precision_hint[part]     * 0.10 +  # 精度知识
    standards_hint[part]     * 0.13 +  # 标准知识
    functional_hint[part]    * 0.10 +  # 功能特征
    assembly_hint[part]      * 0.08 +  # 装配关系
    manufacturing_hint[part] * 0.10 +  # 制造工艺
    geometry_hint[part]      * 0.07    # 几何模式 (动态)
)
```

### 4.2 决策逻辑

```python
# 1. 对所有零件类型按分数排序
sorted_parts = sorted(scores.items(), key=lambda x: x[1], reverse=True)

# 2. 选择最高分作为主分类
best_part = sorted_parts[0][0]
best_score = sorted_parts[0][1]

# 3. 选择分数 > 0.1 的作为备选
alternatives = [p for p in sorted_parts[1:4] if p[1] > 0.1]

# 4. 返回结果
return {
    "part_type": best_part,
    "confidence": best_score,
    "alternatives": alternatives
}
```

## 5. 支持的零件类型

| 零件类型 | 中文名称 | 典型特征 |
|---------|---------|---------|
| shaft | 轴类零件 | 高长宽比方差，圆形主导 |
| gear | 齿轮零件 | 齿形轮廓，圆弧丰富 |
| bearing | 轴承零件 | 高球度，同心圆多 |
| bolt | 螺栓紧固件 | 螺纹特征，标准件 |
| flange | 法兰零件 | 多孔螺栓模式，圆盘形 |
| housing | 壳体/箱体 | 高复杂度，多腔体 |
| plate | 板类零件 | 低球度，平面主导 |
| washer | 垫圈 | 简单环形，标准尺寸 |
| spring | 弹簧 | 螺旋特征，弹簧钢 |
| pulley | 皮带轮 | 槽形轮廓，回转体 |
| coupling | 联轴器 | 连接特征，配合面 |

## 6. API 使用示例

### 6.1 基础分类 (静态知识库)

```bash
POST /api/v1/analyze
Content-Type: application/json

{
    "file_path": "drawing.dxf",
    "classify_parts": true,
    "enhanced_classification": true
}
```

### 6.2 动态知识库分类

```bash
POST /api/v1/analyze
Content-Type: application/json

{
    "file_path": "drawing.dxf",
    "classify_parts": true,
    "enhanced_classification": true,
    "use_dynamic_knowledge": true
}
```

### 6.3 返回结果示例

```json
{
    "part_recognition": {
        "part_type": "bearing",
        "display_name": "轴承零件",
        "confidence": 0.8523,
        "alternatives": [
            {"part_type": "shaft", "display_name": "轴类零件", "confidence": 0.1234},
            {"part_type": "coupling", "display_name": "联轴器", "confidence": 0.0856}
        ],
        "classifier_mode": "enhanced_dynamic_dynamic",
        "knowledge_source": "dynamic",
        "knowledge_version": "2025-11-30T01:33:58",
        "match_details": {
            "geometric_matches": ["high_sphericity", "low_aspect_variance"],
            "entity_matches": ["high_circle_count", "concentric_circles"],
            "ocr_matches": ["bearing_material", "precision_tolerance"]
        },
        "score_breakdown": {
            "base_classifier": 0.2100,
            "material": 0.1020,
            "precision": 0.0850,
            "standards": 0.1170,
            "functional": 0.0800,
            "assembly": 0.0640,
            "manufacturing": 0.0900,
            "geometry": 0.0210
        },
        "modules_contributed": 7
    }
}
```

## 7. 知识库热更新

### 7.1 添加新规则

```bash
POST /api/v1/knowledge/rules/material
Content-Type: application/json

{
    "name": "titanium_alloy",
    "chinese_name": "钛合金",
    "keywords": ["TC4", "Ti-6Al-4V", "钛合金"],
    "part_hints": {"aerospace_part": 0.90}
}
```

### 7.2 触发热更新

```bash
POST /api/v1/knowledge/reload
```

### 7.3 验证规则

```bash
POST /api/v1/knowledge/test-hints
Content-Type: application/json

{
    "text": "TC4 钛合金",
    "geometric_features": {"sphericity": 0.7},
    "entity_counts": {"CIRCLE": 20}
}
```

## 8. 识别流程图

```
输入: CAD文件 (DXF/DWG/STEP)
         │
         ▼
┌─────────────────────┐
│  1. 解析CAD文件      │
│  提取实体和图层      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. 特征提取         │
│  geometric_features │
│  entity_counts      │
│  ocr_data (可选)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. 基础分类器评估   │
│  遍历11种零件类型   │
│  评估模式匹配规则   │
│  生成base_scores   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. 扩展知识模块     │
│  材料→part_hints   │
│  精度→part_hints   │
│  标准→part_hints   │
│  功能→part_hints   │
│  装配→part_hints   │
│  制造→part_hints   │
│  几何→part_hints   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. 加权融合         │
│  final_score =     │
│  Σ(weight × hint)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  6. 排序输出         │
│  part_type (最高分) │
│  confidence        │
│  alternatives      │
│  score_breakdown   │
└─────────────────────┘
```

## 9. 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 分类延迟 | < 100ms | 单文件分类时间 |
| 准确率 | > 85% | 主分类正确率 |
| Top-3准确率 | > 95% | 前3个分类包含正确答案 |
| 知识热更新 | < 1s | 规则更新生效时间 |

## 10. 总结

当前系统采用**多模态融合**方法进行部件识别：

1. **信号层**: 从CAD文件提取三类信号（几何特征、实体统计、OCR文本）
2. **知识层**: 8个专业知识模块分别给出分类建议（hints）
3. **融合层**: 加权融合所有知识模块的建议，得出最终分类
4. **输出层**: 返回主分类、置信度、备选分类和详细分数分解

系统支持**静态知识**(代码内置)和**动态知识**(JSON配置)两种模式，动态知识支持热更新，无需重启服务即可生效。
