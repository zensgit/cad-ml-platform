# 动态知识库系统设计文档

## 概述

动态知识库系统为 CAD ML Platform 提供可热加载、可配置的机械知识管理能力。通过将知识规则从硬编码迁移到外部配置文件，实现：

- **运行时更新**：无需重启服务即可添加/修改知识规则
- **非开发人员维护**：通过 JSON 文件或 REST API 管理知识
- **版本追踪**：自动记录每条规则的创建和更新时间
- **静态回退**：动态规则为空时自动使用内置静态知识库

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层 (API)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ /knowledge/ │  │ /analyze/   │  │ EnhancedPartClassifier  │  │
│  │   REST API  │  │   端点      │  │                         │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     动态知识库层                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              DynamicKnowledgeBase (loader.py)           │    │
│  │  - 统一接口                                              │    │
│  │  - 静态/动态规则融合                                      │    │
│  │  - 分类提示聚合                                          │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │              KnowledgeManager (manager.py)              │    │
│  │  - 单例模式                                              │    │
│  │  - 关键词/模式缓存                                        │    │
│  │  - 热加载支持                                            │    │
│  │  - 变更通知回调                                          │    │
│  └──────────────────────────┬──────────────────────────────┘    │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐    │
│  │              KnowledgeStore (store.py)                  │    │
│  │  - JSONKnowledgeStore (JSON文件存储)                     │    │
│  │  - InMemoryKnowledgeStore (内存存储，测试用)              │    │
│  │  - [可扩展] RedisKnowledgeStore                         │    │
│  └──────────────────────────┬──────────────────────────────┘    │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     存储层 (data/knowledge/)                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ material_rules   │  │ geometry_rules   │  │ standard_rules│  │
│  │     .json        │  │     .json        │  │    .json      │  │
│  └──────────────────┘  └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
src/core/knowledge/
├── dynamic/                    # 动态知识库模块
│   ├── __init__.py            # 模块导出
│   ├── models.py              # 数据模型定义
│   ├── store.py               # 存储后端实现
│   ├── manager.py             # 知识管理器
│   └── loader.py              # 动态加载器
│
├── part_knowledge.py          # [静态] 基础零件知识
├── material_knowledge.py      # [静态] 材料知识
├── precision_knowledge.py     # [静态] 精度知识
├── standards_knowledge.py     # [静态] 标准知识
├── functional_knowledge.py    # [静态] 功能特征知识
├── assembly_knowledge.py      # [静态] 装配关系知识
├── manufacturing_knowledge.py # [静态] 制造工艺知识
└── enhanced_classifier.py     # 增强分类器

src/api/v1/
└── knowledge.py               # 知识库管理 REST API

data/knowledge/                # 知识规则存储目录
├── material_rules.json
├── precision_rules.json
├── standard_rules.json
├── functional_rules.json
├── assembly_rules.json
├── manufacturing_rules.json
└── geometry_rules.json
```

---

## 数据模型

### 基础规则结构 (KnowledgeEntry)

所有规则类型的基类，定义通用字段：

```python
@dataclass
class KnowledgeEntry:
    id: str                      # 唯一标识符 (UUID)
    category: KnowledgeCategory  # 规则类别
    name: str                    # 英文名称
    chinese_name: str            # 中文名称
    description: str             # 描述
    keywords: List[str]          # 匹配关键词
    ocr_patterns: List[str]      # OCR正则模式
    part_hints: Dict[str, float] # 零件类型提示 {part_type: score}
    enabled: bool                # 是否启用
    priority: int                # 优先级 (越高越重要)
    source: str                  # 来源 ("builtin"/"user"/"imported")
    created_at: str              # 创建时间
    updated_at: str              # 更新时间
    metadata: Dict[str, Any]     # 扩展元数据
```

### 规则类别 (KnowledgeCategory)

```python
class KnowledgeCategory(str, Enum):
    MATERIAL = "material"           # 材料规则
    PRECISION = "precision"         # 精度规则
    STANDARD = "standard"           # 标准规则
    FUNCTIONAL = "functional"       # 功能特征规则
    ASSEMBLY = "assembly"           # 装配关系规则
    MANUFACTURING = "manufacturing" # 制造工艺规则
    GEOMETRY = "geometry"           # 几何模式规则
    PART_TYPE = "part_type"        # 零件类型规则
```

### 专用规则类型

#### MaterialRule - 材料规则

```python
@dataclass
class MaterialRule(KnowledgeEntry):
    material_type: str              # 材料类型 (steel/aluminum/bronze)
    material_grades: List[str]      # 材料牌号
    hardness_range: Tuple[int, int] # 硬度范围 (HRC)
    typical_applications: List[str] # 典型应用
```

#### PrecisionRule - 精度规则

```python
@dataclass
class PrecisionRule(KnowledgeEntry):
    tolerance_grade: str                    # 公差等级 (IT5/IT6/IT7)
    surface_roughness_range: Tuple[float, float]  # Ra范围
    gdt_symbols: List[str]                  # GD&T符号
    fit_types: List[str]                    # 配合类型
```

#### StandardRule - 标准规则

```python
@dataclass
class StandardRule(KnowledgeEntry):
    standard_org: str        # 标准组织 (GB/ISO/DIN/ANSI)
    standard_number: str     # 标准号
    designation_pattern: str # 标识正则模式
    year: int               # 标准年份
```

#### FunctionalFeatureRule - 功能特征规则

```python
@dataclass
class FunctionalFeatureRule(KnowledgeEntry):
    feature_type: str                    # 特征类型 (keyway/spline/thread)
    typical_parts: List[str]             # 典型零件
    geometric_indicators: Dict[str, Any] # 几何指标
    weight: float                        # 特征权重
```

#### AssemblyRule - 装配关系规则

```python
@dataclass
class AssemblyRule(KnowledgeEntry):
    part_a: str              # 第一零件类型
    part_b: str              # 第二零件类型
    connection_type: str     # 连接类型 (fit/fastener/weld)
    typical_fits: List[str]  # 典型配合
```

#### ManufacturingRule - 制造工艺规则

```python
@dataclass
class ManufacturingRule(KnowledgeEntry):
    process_type: str                       # 工艺类型
    process_name: str                       # 工艺名称
    surface_finish_range: Tuple[float, float] # 表面粗糙度范围
    tolerance_capability: str               # 公差能力
```

#### GeometryPattern - 几何模式规则

```python
@dataclass
class GeometryPattern(KnowledgeEntry):
    conditions: Dict[str, Any]  # 几何条件
    # 示例:
    # {
    #     "aspect_variance": {"min": 0.2, "max": 0.5},
    #     "sphericity": {"min": 0.6},
    #     "circle_ratio": {"min": 0.1, "max": 0.3}
    # }
```

---

## JSON 配置格式

### 文件结构

每个类别对应一个 JSON 文件，格式如下：

```json
{
  "version": "2024-01-01T00:00:00",
  "category": "material",
  "count": 3,
  "updated_at": "2024-01-01T00:00:00",
  "rules": [
    {
      "id": "mat-001",
      "category": "material",
      "name": "bearing_steel",
      "chinese_name": "轴承钢",
      "description": "高碳铬轴承钢，用于滚动轴承",
      "keywords": ["GCr15", "轴承钢", "SUJ2", "100Cr6"],
      "ocr_patterns": ["GCr\\d+", "SUJ\\d"],
      "part_hints": {"bearing": 0.85, "shaft": 0.15},
      "enabled": true,
      "priority": 100,
      "source": "builtin",
      "material_type": "steel",
      "material_grades": ["GCr15", "GCr15SiMn"],
      "hardness_range": [60, 65],
      "typical_applications": ["滚动轴承", "精密轴承"]
    }
  ]
}
```

### 几何模式配置示例

```json
{
  "category": "geometry",
  "rules": [
    {
      "id": "geo-001",
      "name": "shaft_geometry",
      "chinese_name": "轴类几何特征",
      "description": "高长径比的圆柱形零件",
      "part_hints": {"shaft": 0.25},
      "conditions": {
        "aspect_variance": {"min": 0.25},
        "sphericity": {"min": 0.6},
        "circle_ratio": {"min": 0.1, "max": 0.4}
      }
    },
    {
      "id": "geo-002",
      "name": "bearing_geometry",
      "chinese_name": "轴承几何特征",
      "part_hints": {"bearing": 0.3},
      "conditions": {
        "sphericity": {"min": 0.85},
        "aspect_variance": {"max": 0.15},
        "circle_ratio": {"min": 0.35}
      }
    }
  ]
}
```

### 支持的几何条件

| 条件名称 | 描述 | 示例 |
|---------|------|------|
| `aspect_variance` | 长宽比方差 | `{"min": 0.2, "max": 0.5}` |
| `sphericity` | 球度（圆柱度） | `{"min": 0.6}` |
| `compactness` | 紧凑度 | `{"min": 0.4}` |
| `complexity_score` | 复杂度分数 | `{"min": 0.5}` |
| `entity_entropy` | 实体熵 | `{"min": 0.6}` |
| `circle_ratio` | 圆实体比例 | `{"min": 0.1, "max": 0.4}` |
| `line_ratio` | 线实体比例 | `{"max": 0.5}` |
| `arc_ratio` | 弧实体比例 | `{"min": 0.2}` |
| `total_entities` | 总实体数 | `{"min": 20}` |

---

## REST API

### 端点列表

| 方法 | 端点 | 描述 |
|------|------|------|
| `GET` | `/v1/knowledge/stats` | 获取统计信息 |
| `GET` | `/v1/knowledge/version` | 获取当前版本 |
| `POST` | `/v1/knowledge/reload` | 强制重载规则 |
| `GET` | `/v1/knowledge/rules` | 列出/搜索规则 |
| `GET` | `/v1/knowledge/rules/{id}` | 获取单条规则 |
| `DELETE` | `/v1/knowledge/rules/{id}` | 删除规则 |
| `POST` | `/v1/knowledge/rules/material` | 创建材料规则 |
| `POST` | `/v1/knowledge/rules/precision` | 创建精度规则 |
| `POST` | `/v1/knowledge/rules/standard` | 创建标准规则 |
| `POST` | `/v1/knowledge/rules/functional` | 创建功能特征规则 |
| `POST` | `/v1/knowledge/rules/assembly` | 创建装配关系规则 |
| `POST` | `/v1/knowledge/rules/manufacturing` | 创建制造工艺规则 |
| `POST` | `/v1/knowledge/rules/geometry` | 创建几何模式规则 |
| `GET` | `/v1/knowledge/export` | 导出全部知识 |
| `POST` | `/v1/knowledge/import` | 导入知识 |
| `POST` | `/v1/knowledge/test-hints` | 测试规则匹配 |

### API 使用示例

#### 获取统计信息

```bash
curl http://localhost:8000/v1/knowledge/stats
```

响应：
```json
{
  "total_rules": 8,
  "version": "2024-01-01T12:00:00",
  "categories": {
    "material": {"total": 3, "enabled": 3},
    "geometry": {"total": 5, "enabled": 5}
  },
  "cache_stats": {
    "keywords": 15,
    "patterns": 8,
    "geometry_patterns": 5
  },
  "static_fallback_enabled": true
}
```

#### 创建材料规则

```bash
curl -X POST http://localhost:8000/v1/knowledge/rules/material \
  -H "Content-Type: application/json" \
  -d '{
    "name": "titanium_alloy",
    "chinese_name": "钛合金",
    "description": "航空钛合金",
    "keywords": ["TC4", "Ti-6Al-4V", "钛合金"],
    "ocr_patterns": ["TC\\d+", "Ti-\\d+Al-\\d+V"],
    "part_hints": {"housing": 0.6, "plate": 0.4},
    "material_type": "titanium",
    "material_grades": ["TC4", "TC11"],
    "typical_applications": ["航空结构件"]
  }'
```

#### 创建几何模式规则

```bash
curl -X POST http://localhost:8000/v1/knowledge/rules/geometry \
  -H "Content-Type: application/json" \
  -d '{
    "name": "spring_geometry",
    "chinese_name": "弹簧几何特征",
    "description": "螺旋弹簧的几何特征",
    "part_hints": {"spring": 0.4},
    "conditions": {
      "arc_ratio": {"min": 0.5},
      "sphericity": {"max": 0.4}
    }
  }'
```

#### 搜索规则

```bash
# 按关键词搜索
curl "http://localhost:8000/v1/knowledge/rules?search=轴承"

# 按类别过滤
curl "http://localhost:8000/v1/knowledge/rules?category=material"

# 组合查询
curl "http://localhost:8000/v1/knowledge/rules?category=geometry&limit=10"
```

#### 热加载规则

```bash
curl -X POST http://localhost:8000/v1/knowledge/reload
```

响应：
```json
{
  "status": "reloaded",
  "version": "2024-01-01T12:30:00"
}
```

#### 测试规则匹配

```bash
curl -X POST http://localhost:8000/v1/knowledge/test-hints \
  -H "Content-Type: application/json" \
  -d '{
    "text": "GCr15 轴承钢 HRC60-65",
    "geometric_features": {
      "sphericity": 0.9,
      "aspect_variance": 0.1
    },
    "entity_counts": {
      "CIRCLE": 20,
      "LINE": 10
    }
  }'
```

响应：
```json
{
  "by_category": {
    "material": {"bearing": 0.85, "shaft": 0.15},
    "geometry": {"bearing": 0.3}
  },
  "aggregated": {
    "bearing": 1.0,
    "shaft": 0.15
  },
  "version": "2024-01-01T12:00:00"
}
```

#### 导出/导入知识

```bash
# 导出
curl http://localhost:8000/v1/knowledge/export > knowledge_backup.json

# 导入（合并模式）
curl -X POST http://localhost:8000/v1/knowledge/import \
  -H "Content-Type: application/json" \
  -d @knowledge_backup.json

# 导入（替换模式）
curl -X POST "http://localhost:8000/v1/knowledge/import?merge=false" \
  -H "Content-Type: application/json" \
  -d @knowledge_backup.json
```

---

## 使用指南

### 方式一：直接编辑 JSON 文件

1. 编辑 `data/knowledge/` 目录下的 JSON 文件
2. 调用 API 热加载：`POST /v1/knowledge/reload`
3. 规则立即生效

```json
// data/knowledge/material_rules.json
{
  "rules": [
    {
      "id": "custom-001",
      "name": "custom_steel",
      "chinese_name": "自定义钢材",
      "keywords": ["自定义关键词", "Custom"],
      "part_hints": {"shaft": 0.8}
    }
  ]
}
```

### 方式二：通过 REST API

```python
import requests

# 添加新规则
response = requests.post(
    "http://localhost:8000/v1/knowledge/rules/material",
    json={
        "name": "new_material",
        "chinese_name": "新材料",
        "keywords": ["新关键词"],
        "part_hints": {"shaft": 0.7}
    }
)
rule_id = response.json()["id"]

# 删除规则
requests.delete(f"http://localhost:8000/v1/knowledge/rules/{rule_id}")
```

### 方式三：程序化使用

```python
from src.core.knowledge.dynamic import (
    KnowledgeManager,
    DynamicKnowledgeBase,
    MaterialRule,
    GeometryPattern,
)

# 获取管理器
manager = KnowledgeManager()

# 添加规则
rule = MaterialRule(
    name="api_created_steel",
    chinese_name="API创建的钢材",
    keywords=["API钢", "测试"],
    part_hints={"shaft": 0.6}
)
manager.add_rule(rule)

# 使用动态知识库
kb = DynamicKnowledgeBase()
hints = kb.get_material_hints(
    ocr_data={"text": "API钢 材质"},
    geometric_features={"sphericity": 0.8}
)
print(hints)  # {"shaft": 0.6}

# 搜索规则
results = manager.search_rules("钢")
for r in results:
    print(f"{r.name}: {r.chinese_name}")
```

---

## 静态知识库回退

当动态规则为空时，系统自动使用静态知识库：

```python
class DynamicKnowledgeBase:
    def get_material_hints(self, ocr_data, ...):
        # 优先使用动态规则
        if self._has_dynamic_rules(KnowledgeCategory.MATERIAL):
            return self._get_dynamic_hints(...)

        # 回退到静态知识库
        if self._use_static_fallback:
            kb = self._get_static_material_kb()
            return kb.get_material_hints(...)

        return {}
```

### 混合使用策略

| 场景 | 策略 |
|------|------|
| 全新部署 | 静态知识库自动生效 |
| 添加自定义规则 | 动态规则优先 |
| 覆盖内置规则 | 创建同名动态规则，设置更高优先级 |
| 禁用某条静态规则 | 创建对应动态规则并设置 `enabled: false` |

---

## 热加载机制

### 自动热加载

```python
# 启动时配置自动热加载间隔
manager = KnowledgeManager(
    auto_reload_interval=60  # 每60秒检查一次
)
```

### 手动热加载

```python
# 程序化触发
manager.reload()

# 或通过 API
# POST /v1/knowledge/reload
```

### 变更通知

```python
def on_knowledge_changed(version: str):
    print(f"知识库已更新: {version}")
    # 执行自定义逻辑，如清理缓存

manager.on_change(on_knowledge_changed)
```

---

## 性能优化

### 缓存机制

管理器内置三级缓存：

1. **关键词缓存** - `Dict[keyword, List[Rule]]`
2. **模式缓存** - `Dict[pattern, List[Rule]]`
3. **几何模式列表** - `List[GeometryPattern]`

缓存在以下时机重建：
- 初始化时
- 热加载时
- 添加/删除规则时

### 线程安全

所有操作使用 `threading.RLock` 保护：

```python
class KnowledgeManager:
    def __init__(self):
        self._cache_lock = threading.RLock()

    def match_keywords(self, text):
        with self._cache_lock:
            # 安全访问缓存
            ...
```

---

## 扩展指南

### 添加新规则类型

1. 在 `models.py` 中定义新的数据类：

```python
@dataclass
class CustomRule(KnowledgeEntry):
    category: KnowledgeCategory = field(default=KnowledgeCategory.CUSTOM)
    custom_field: str = ""
```

2. 在 `RULE_TYPE_MAP` 中注册：

```python
RULE_TYPE_MAP = {
    ...
    KnowledgeCategory.CUSTOM: CustomRule,
}
```

3. 在 `knowledge.py` API 中添加端点：

```python
@router.post("/rules/custom")
async def create_custom_rule(rule_data: CustomRuleCreate):
    ...
```

### 添加新存储后端

实现 `KnowledgeStore` 接口：

```python
class RedisKnowledgeStore(KnowledgeStore):
    def __init__(self, redis_url: str):
        self._client = redis.from_url(redis_url)

    def get(self, rule_id: str) -> Optional[KnowledgeEntry]:
        data = self._client.hget("knowledge:rules", rule_id)
        return create_rule_from_dict(json.loads(data)) if data else None

    def save(self, rule: KnowledgeEntry) -> str:
        self._client.hset("knowledge:rules", rule.id, json.dumps(rule.to_dict()))
        return rule.id

    # ... 实现其他方法
```

---

## 最佳实践

### 规则命名规范

- `name`: 使用小写英文 + 下划线，如 `bearing_steel`
- `chinese_name`: 简洁的中文名称
- `id`: 推荐使用 `{category}-{number}` 格式，如 `mat-001`

### 关键词设计

```json
{
  "keywords": [
    "GCr15",        // 精确匹配
    "轴承钢",       // 中文名称
    "bearing steel", // 英文名称
    "SUJ2",         // 日本标准
    "100Cr6"        // 德国标准
  ]
}
```

### OCR 模式设计

```json
{
  "ocr_patterns": [
    "GCr\\d+",              // 匹配 GCr15, GCr4 等
    "(?i)bearing\\s*steel", // 不区分大小写
    "\\d+Cr\\d+"            // 匹配 100Cr6, 52100 等
  ]
}
```

### 零件提示权重

- 总和建议 ≤ 1.0
- 主要零件类型：0.6-0.9
- 次要零件类型：0.1-0.3

```json
{
  "part_hints": {
    "bearing": 0.85,  // 主要用于轴承
    "shaft": 0.15    // 少量用于轴
  }
}
```

---

## 故障排除

### 规则不生效

1. 检查 `enabled` 是否为 `true`
2. 检查 JSON 语法是否正确
3. 调用 `/v1/knowledge/reload` 强制重载
4. 查看日志确认规则是否加载

### 几何模式不匹配

1. 使用 `/v1/knowledge/test-hints` 测试
2. 检查条件阈值是否合理
3. 确认输入的几何特征名称正确

### 性能问题

1. 减少正则模式复杂度
2. 控制规则总数（建议 < 1000）
3. 调整自动热加载间隔

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2024-01 | 初始版本：JSON存储、基础CRUD、热加载 |

---

## 相关文档

- [增强分类器设计](./ENHANCED_CLASSIFIER.md)
- [机械知识库概述](./MECHANICAL_KNOWLEDGE.md)
- [API 参考文档](./API_REFERENCE.md)
