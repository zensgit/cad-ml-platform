# 动态知识库集成方案

## 1. 当前识别架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      CAD 文件上传                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    特征提取 (FeatureExtractor)                   │
│  ├─ v1: 7维基础几何特征                                          │
│  ├─ v2: 12维 (+ 归一化尺寸 + 长宽比)                             │
│  ├─ v3: 22维 (+ 实体类型频率)                                    │
│  ├─ v4: 24维 (+ 表面数 + 形状熵)                                 │
│  ├─ v5: 26维 (旋转/缩放不变)                                     │
│  └─ v6: 32维 (矩不变量)                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              EnhancedPartClassifier (多模态融合)                 │
│                                                                  │
│  输入信号:                                                       │
│  ├─ geometric_features: {aspect_variance, sphericity, ...}      │
│  ├─ entity_counts: {CIRCLE: 50, LINE: 30, ARC: 15, ...}         │
│  └─ ocr_data: {dimensions, symbols, title_block, ...}           │
│                                                                  │
│  知识模块 (当前静态):                                            │
│  ├─ MechanicalPartKnowledgeBase (35%) - 基础分类                │
│  ├─ MaterialKnowledgeBase (12%) - 材料识别                      │
│  ├─ PrecisionKnowledgeBase (10%) - 精度识别                     │
│  ├─ StandardsKnowledgeBase (13%) - 标准识别                     │
│  ├─ FunctionalKnowledgeBase (12%) - 功能特征                    │
│  ├─ AssemblyKnowledgeBase (8%) - 装配关系                       │
│  └─ ManufacturingKnowledgeBase (10%) - 制造工艺                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 EnhancedClassificationResult                     │
│  ├─ part_type: 最终分类                                          │
│  ├─ confidence: 置信度                                           │
│  ├─ alternatives: 备选分类                                       │
│  └─ score_breakdown: 各模块贡献明细                              │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 集成方案

### 方案概述

将 `EnhancedPartClassifier` 升级为支持动态知识库的 `DynamicEnhancedClassifier`，同时保持向后兼容。

### 2.1 架构改动

```
┌─────────────────────────────────────────────────────────────────┐
│              DynamicEnhancedClassifier (升级版)                  │
│                                                                  │
│  知识来源:                                                       │
│  ├─ DynamicKnowledgeBase ─┬─ 动态规则 (JSON配置)                │
│  │                        └─ 静态规则 (代码回退)                 │
│  └─ KnowledgeManager ──── 热更新 + 缓存 + 版本控制              │
│                                                                  │
│  新增功能:                                                       │
│  ├─ 运行时规则更新 (无需重启)                                   │
│  ├─ 动态/静态规则混合                                           │
│  ├─ 规则优先级控制                                              │
│  └─ 知识版本追踪                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 代码改动

#### 改动1: 创建 DynamicEnhancedClassifier

```python
# src/core/knowledge/dynamic_classifier.py

from src.core.knowledge.dynamic.loader import DynamicKnowledgeBase
from src.core.knowledge.dynamic.manager import get_knowledge_manager

class DynamicEnhancedClassifier:
    """支持动态知识库的增强分类器"""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_dynamic: bool = True,
        fallback_to_static: bool = True,
    ):
        # 动态知识库 (内部会处理静态回退)
        self.dynamic_kb = DynamicKnowledgeBase(
            use_static_fallback=fallback_to_static
        )
        self.manager = get_knowledge_manager()
        self.use_dynamic = use_dynamic

        # 保留静态分类器用于混合模式
        if not use_dynamic:
            self._init_static_modules()

        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._normalize_weights()

    def classify(
        self,
        geometric_features: Optional[Dict[str, Any]] = None,
        entity_counts: Optional[Dict[str, int]] = None,
        ocr_data: Optional[Dict[str, Any]] = None,
    ) -> EnhancedClassificationResult:
        """使用动态知识库进行分类"""

        if self.use_dynamic:
            # 使用动态知识库获取所有hints
            all_hints = self.dynamic_kb.get_all_hints(
                ocr_data or {},
                geometric_features or {},
                entity_counts or {},
            )

            # 获取几何模式匹配
            matched_patterns = self.manager.match_geometry(
                geometric_features or {},
                entity_counts or {},
            )

            # 融合动态规则hints
            return self._fuse_dynamic_hints(
                all_hints,
                matched_patterns,
                geometric_features,
                entity_counts,
                ocr_data,
            )
        else:
            # 回退到原有静态逻辑
            return self._classify_static(...)
```

#### 改动2: 更新 EnhancedPartClassifier 添加动态模式选项

```python
# src/core/knowledge/enhanced_classifier.py (修改)

class EnhancedPartClassifier:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_dynamic_knowledge: bool = False,  # 新增参数
    ) -> None:
        self.use_dynamic = use_dynamic_knowledge

        if use_dynamic_knowledge:
            # 使用动态知识库
            from src.core.knowledge.dynamic.loader import DynamicKnowledgeBase
            self.dynamic_kb = DynamicKnowledgeBase(use_static_fallback=True)
        else:
            # 原有静态模块
            self.base_kb = MechanicalPartKnowledgeBase()
            self.material_kb = MaterialKnowledgeBase()
            # ... 其他模块
```

#### 改动3: 更新 API 支持动态知识库

```python
# src/api/v1/analyze.py (修改)

class AnalysisOptions(BaseModel):
    # ... 现有字段 ...
    use_dynamic_knowledge: bool = False  # 新增字段

async def analyze_cad_file(...):
    # 根据选项决定使用哪种分类器
    if options.enhanced_classification:
        classifier = EnhancedPartClassifier(
            use_dynamic_knowledge=options.use_dynamic_knowledge
        )
        result = classifier.classify(...)
```

## 3. 具体实施步骤

### Step 1: 更新 EnhancedPartClassifier (低风险)

```python
# 在 __init__ 中添加 use_dynamic_knowledge 参数
# 在 classify() 中根据参数选择知识来源
```

### Step 2: 更新 DynamicKnowledgeBase.get_all_hints()

确保返回格式与静态模块一致:
```python
{
    "material": {"bearing": 0.85, "shaft": 0.15},
    "precision": {"bearing": 0.10},
    "standards": {"bolt": 0.20},
    "functional": {"gear": 0.15},
    "assembly": {"shaft": 0.10},
    "manufacturing": {"bearing": 0.15},
    "geometry": {"shaft": 0.25},  # 新增几何模式
}
```

### Step 3: 更新 API 选项

```python
class AnalysisOptions(BaseModel):
    use_dynamic_knowledge: bool = Field(
        False,
        description="使用动态知识库进行分类 (支持热更新规则)"
    )
```

### Step 4: 添加知识版本到结果

```python
@dataclass
class EnhancedClassificationResult:
    # ... 现有字段 ...
    knowledge_version: str = ""  # 新增: 知识库版本
    knowledge_source: str = "static"  # 新增: "static" | "dynamic" | "hybrid"
```

## 4. API 使用示例

### 4.1 使用静态知识库 (默认)

```bash
POST /api/v1/analyze
{
    "file_path": "drawing.dxf",
    "enhanced_classification": true
}
```

### 4.2 使用动态知识库

```bash
POST /api/v1/analyze
{
    "file_path": "drawing.dxf",
    "enhanced_classification": true,
    "use_dynamic_knowledge": true
}
```

### 4.3 热更新知识后重新分类

```bash
# 1. 添加新规则
POST /api/v1/knowledge/rules/material
{
    "name": "titanium_alloy",
    "chinese_name": "钛合金",
    "keywords": ["TC4", "Ti-6Al-4V", "钛合金"],
    "part_hints": {"aerospace_part": 0.9}
}

# 2. 触发热更新
POST /api/v1/knowledge/reload

# 3. 使用新规则分类
POST /api/v1/analyze
{
    "file_path": "titanium_part.dxf",
    "use_dynamic_knowledge": true
}
```

## 5. 回退与兼容性

### 5.1 静态回退机制

当动态规则为空时，自动使用静态知识库:

```python
class DynamicKnowledgeBase:
    def get_material_hints(self, ...):
        # 优先使用动态规则
        if self._has_dynamic_rules(KnowledgeCategory.MATERIAL):
            return self._get_dynamic_hints(...)

        # 回退到静态知识库
        if self._use_static_fallback:
            return self._get_static_material_kb().get_material_hints(...)

        return {}
```

### 5.2 混合模式

支持动态规则与静态规则混合使用:

```python
def _merge_hints(self, dynamic_hints, static_hints):
    """合并动态和静态hints，动态规则优先"""
    merged = static_hints.copy()
    for part, score in dynamic_hints.items():
        if part in merged:
            # 动态规则分数更高时覆盖
            merged[part] = max(merged[part], score)
        else:
            merged[part] = score
    return merged
```

## 6. 权重配置

### 6.1 默认权重 (包含几何模式)

```python
DEFAULT_WEIGHTS = {
    "base_classifier": 0.30,  # 基础分类 (略降)
    "material": 0.12,          # 材料识别
    "precision": 0.10,         # 精度识别
    "standards": 0.13,         # 标准识别
    "functional": 0.10,        # 功能特征 (略降)
    "assembly": 0.08,          # 装配关系
    "manufacturing": 0.10,     # 制造工艺
    "geometry": 0.07,          # 新增: 几何模式匹配
}
```

### 6.2 运行时权重调整

```bash
POST /api/v1/analyze
{
    "file_path": "drawing.dxf",
    "use_dynamic_knowledge": true,
    "classification_weights": {
        "material": 0.20,  # 提高材料权重
        "geometry": 0.15   # 提高几何权重
    }
}
```

## 7. 测试策略

### 7.1 A/B 测试

```python
# 同时运行静态和动态分类，对比结果
async def analyze_with_comparison(...):
    static_result = EnhancedPartClassifier(
        use_dynamic_knowledge=False
    ).classify(...)

    dynamic_result = EnhancedPartClassifier(
        use_dynamic_knowledge=True
    ).classify(...)

    return {
        "static": static_result,
        "dynamic": dynamic_result,
        "match": static_result.part_type == dynamic_result.part_type,
    }
```

### 7.2 回归测试

保留已知正确分类的测试集:
```python
def test_regression_with_dynamic_kb():
    classifier = EnhancedPartClassifier(use_dynamic_knowledge=True)

    for test_case in REGRESSION_TEST_SET:
        result = classifier.classify(
            test_case.geometric_features,
            test_case.entity_counts,
            test_case.ocr_data,
        )
        assert result.part_type == test_case.expected_type
```

## 8. 监控指标

### 8.1 分类准确率追踪

```python
# 添加 Prometheus 指标
classification_accuracy = Gauge(
    "cad_classification_accuracy",
    "Classification accuracy by knowledge source",
    ["knowledge_source"]  # static / dynamic / hybrid
)

knowledge_version = Info(
    "cad_knowledge_version",
    "Current knowledge base version"
)
```

### 8.2 规则命中率

```python
rule_hit_counter = Counter(
    "cad_knowledge_rule_hits_total",
    "Knowledge rule hit count",
    ["category", "rule_id"]
)
```

## 9. 实施时间线

| 阶段 | 任务 | 风险 |
|------|------|------|
| Phase 1 | 更新 EnhancedPartClassifier 添加 use_dynamic_knowledge 参数 | 低 |
| Phase 2 | 更新 API 支持 use_dynamic_knowledge 选项 | 低 |
| Phase 3 | 完善 DynamicKnowledgeBase.get_all_hints() | 中 |
| Phase 4 | 添加几何模式权重到融合算法 | 中 |
| Phase 5 | A/B 测试与调优 | 低 |
| Phase 6 | 生产环境灰度发布 | 中 |

## 10. 总结

集成动态知识库需要更新的关键位置:

1. **EnhancedPartClassifier** (`src/core/knowledge/enhanced_classifier.py`)
   - 添加 `use_dynamic_knowledge` 参数
   - 根据参数选择知识来源

2. **DynamicKnowledgeBase** (`src/core/knowledge/dynamic/loader.py`)
   - 确保 `get_all_hints()` 返回格式与静态模块一致
   - 完善几何模式匹配

3. **API 层** (`src/api/v1/analyze.py`)
   - 添加 `use_dynamic_knowledge` 选项
   - 传递给分类器

4. **结果模型** (`EnhancedClassificationResult`)
   - 添加知识版本追踪字段
