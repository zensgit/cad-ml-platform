# CAD Assistant 增强功能设计文档

## 概述

本文档描述了 CAD Assistant 模块的四个增强功能设计：

| 优先级 | 功能 | 模块 | PR |
|--------|------|------|-----|
| P0 | 对话历史持久化 | `persistence.py` | #54 |
| P1 | 语义检索增强 | `semantic_retrieval.py` | #55 |
| P2 | 响应质量评估 | `quality_evaluation.py` | #56 |
| P3 | API 服务封装 | `api_service.py` | #57 |

---

## P0: 对话历史持久化

### 设计目标

- 支持对话历史的持久化存储和恢复
- 提供多种存储后端（JSON 文件、SQLite 数据库）
- 支持自动保存和会话管理

### 架构设计

```
┌─────────────────────────────────────────────────────┐
│               ConversationPersistence               │
│  ┌─────────────────────────────────────────────┐   │
│  │           StorageBackend (Abstract)          │   │
│  └─────────────────────────────────────────────┘   │
│           ▲                     ▲                   │
│           │                     │                   │
│  ┌────────┴────────┐   ┌───────┴────────┐         │
│  │ JSONStorageBackend │ │ SQLiteStorageBackend │   │
│  └─────────────────┘   └──────────────────┘        │
└─────────────────────────────────────────────────────┘
```

### 核心类

#### `StorageBackend` (抽象基类)
```python
class StorageBackend(ABC):
    @abstractmethod
    def save_conversation(self, conversation_id: str, data: Dict) -> bool

    @abstractmethod
    def load_conversation(self, conversation_id: str) -> Optional[Dict]

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> bool

    @abstractmethod
    def list_conversations(self) -> List[str]
```

#### `JSONStorageBackend`
- 基于文件系统的 JSON 存储
- 每个对话一个 JSON 文件
- 自动创建目录结构
- 支持索引文件快速列表

#### `SQLiteStorageBackend`
- 基于 SQLite 的数据库存储
- 单文件数据库
- 支持事务和并发
- 自动表创建和迁移

#### `ConversationPersistence`
```python
class ConversationPersistence:
    def __init__(
        self,
        storage_backend: Optional[StorageBackend] = None,
        storage_path: Optional[str] = None,
        auto_save: bool = True,
        auto_save_interval: int = 300,  # 5 minutes
    )
```

### 存储格式

```json
{
  "id": "conv-uuid",
  "created_at": 1706000000.0,
  "updated_at": 1706001000.0,
  "messages": [
    {
      "role": "user",
      "content": "304不锈钢的强度？",
      "timestamp": 1706000000.0
    },
    {
      "role": "assistant",
      "content": "304不锈钢的抗拉强度约为520MPa...",
      "timestamp": 1706000001.0,
      "metadata": {"confidence": 0.9}
    }
  ],
  "metadata": {
    "domain": "materials",
    "language": "zh"
  }
}
```

---

## P1: 语义检索增强

### 设计目标

- 基于向量嵌入的语义相似度搜索
- 支持多种嵌入提供者（简单 TF-IDF、SentenceTransformers）
- 混合检索（语义 + 关键词）

### 架构设计

```
┌─────────────────────────────────────────────────────┐
│                  SemanticRetriever                  │
│  ┌─────────────────┐   ┌───────────────────────┐   │
│  │ EmbeddingProvider │   │      VectorStore      │   │
│  └─────────────────┘   └───────────────────────┘   │
│          ▲                                          │
│          │                                          │
│  ┌───────┴────────┐   ┌────────────────────────┐   │
│  │ SimpleProvider │   │ SentenceTransformerProvider│
│  │ (TF-IDF/N-gram)│   │   (Neural Embeddings)  │   │
│  └────────────────┘   └────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 核心类

#### `SimpleEmbeddingProvider`
```python
class SimpleEmbeddingProvider(EmbeddingProvider):
    """基于字符 n-gram TF-IDF 的简单嵌入提供者"""

    def __init__(
        self,
        dimension: int = 256,
        ngram_range: Tuple[int, int] = (2, 4),
    )

    def embed_text(self, text: str) -> List[float]
    def embed_batch(self, texts: List[str]) -> List[List[float]]
```

算法流程:
1. 提取字符 n-grams (2-4 字符)
2. 计算 TF (词频)
3. 哈希映射到固定维度
4. L2 归一化

#### `SentenceTransformerProvider`
```python
class SentenceTransformerProvider(EmbeddingProvider):
    """基于 sentence-transformers 的神经嵌入"""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
    )
```

特点:
- 惰性加载模型
- 支持多语言
- GPU/CPU 自动选择

#### `VectorStore`
```python
class VectorStore:
    def add(self, text: str, vector: List[float], source: str = "", metadata: Optional[Dict] = None) -> int

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        min_score: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> List[SemanticSearchResult]
```

特点:
- 余弦相似度计算
- 支持来源过滤
- JSON 持久化

#### `SemanticRetriever`
```python
class SemanticRetriever:
    def index_knowledge_base(self, knowledge_items: List[Dict]) -> int

    def search(self, query: str, top_k: int = 10) -> List[SemanticSearchResult]

    def hybrid_search(
        self,
        query: str,
        keyword_results: List[Dict],
        top_k: int = 10,
    ) -> List[SemanticSearchResult]
```

### 混合检索公式

```
final_score = semantic_score × hybrid_weight + keyword_score × (1 - hybrid_weight)
```

默认 `hybrid_weight = 0.7`，语义检索权重更高。

---

## P2: 响应质量评估

### 设计目标

- 多维度评估响应质量
- 支持相关性、完整性、清晰度、技术深度、可操作性
- 提供整体评分和字母等级

### 架构设计

```
┌─────────────────────────────────────────────────────┐
│              ResponseQualityEvaluator               │
│  ┌───────────────────────────────────────────────┐ │
│  │ RelevanceEvaluator │ CompletenessEvaluator    │ │
│  │ ClarityEvaluator   │ TechnicalDepthEvaluator  │ │
│  │ ActionabilityEvaluator                        │ │
│  └───────────────────────────────────────────────┘ │
│                         │                          │
│                         ▼                          │
│               EvaluationResult                     │
│  ┌───────────────────────────────────────────────┐ │
│  │ overall_score: float  │  grade: str (A-F)     │ │
│  │ dimension_scores: Dict[QualityDimension, ...]│ │
│  │ strengths: List[str]  │  weaknesses: List    │ │
│  │ suggestions: List[str]                       │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 评估维度

| 维度 | 权重 | 评估内容 |
|------|------|----------|
| `RELEVANCE` | 0.25 | 关键词覆盖、领域匹配 |
| `COMPLETENESS` | 0.25 | 数值数据、单位、范围、引用 |
| `CLARITY` | 0.20 | 模糊语言检测、结构清晰度 |
| `TECHNICAL_DEPTH` | 0.15 | 技术术语、数据模式、公式 |
| `ACTIONABILITY` | 0.15 | 行动导向语言、步骤结构 |

### 评分规则

```python
# 整体分数计算
overall_score = sum(
    dim_score.score * dim_score.weight
    for dim_score in dimension_scores.values()
)

# 等级映射
def _score_to_grade(score: float) -> str:
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.7:
        return "C"
    elif score >= 0.6:
        return "D"
    else:
        return "F"
```

### 评估结果示例

```python
EvaluationResult(
    overall_score=0.82,
    grade="B",
    dimension_scores={
        QualityDimension.RELEVANCE: DimensionScore(
            dimension=QualityDimension.RELEVANCE,
            score=0.85,
            weight=0.25,
            details=["关键词覆盖率: 85%", "领域匹配度: 高"]
        ),
        # ... 其他维度
    },
    strengths=["高相关性", "技术数据完整"],
    weaknesses=["缺少具体步骤"],
    suggestions=["添加应用场景示例"]
)
```

---

## P3: API 服务封装

### 设计目标

- 提供 REST API 接口
- 支持请求验证和速率限制
- 兼容 Flask 和 FastAPI

### 架构设计

```
┌─────────────────────────────────────────────────────┐
│                  CADAssistantAPI                    │
│  ┌─────────────────────────────────────────────┐   │
│  │ Endpoints                                    │   │
│  │  /ask          /conversation   /evaluate    │   │
│  │  /health       /info                        │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────┐   ┌───────────────────────┐   │
│  │   RateLimiter   │   │   Request Validators  │   │
│  └─────────────────┘   └───────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │ Error Handling                              │   │
│  │  APIError │ ValidationError │ RateLimitError│   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
          │                    │
          ▼                    ▼
    ┌───────────┐        ┌───────────┐
    │   Flask   │        │  FastAPI  │
    │   App     │        │   App     │
    └───────────┘        └───────────┘
```

### API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/ask` | POST | 查询助手 |
| `/conversation` | POST | 管理对话 (create/get/delete/list/clear) |
| `/evaluate` | POST | 评估响应质量 |
| `/health` | GET | 健康检查 |
| `/info` | GET | API 信息 |

### 请求/响应格式

#### `/ask` 请求
```json
{
  "query": "304不锈钢的强度是多少？",
  "conversation_id": "conv-123",
  "options": {}
}
```

#### `/ask` 响应
```json
{
  "success": true,
  "data": {
    "answer": "304不锈钢的抗拉强度约为520MPa...",
    "confidence": 0.9,
    "sources": [...],
    "conversation_id": "conv-123",
    "intent": "material_property"
  },
  "request_id": "req-uuid",
  "timestamp": 1706000000.0
}
```

### 速率限制

```python
class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    )
```

特点:
- 基于客户端 ID 的限制
- 滑动窗口算法
- 自动清理过期记录

### 错误处理

| 错误类型 | HTTP 状态码 | 错误码 |
|----------|-------------|--------|
| `APIError` | 500 | INTERNAL_ERROR |
| `ValidationError` | 400 | VALIDATION_ERROR |
| `NotFoundError` | 404 | NOT_FOUND |
| `RateLimitError` | 429 | RATE_LIMIT_EXCEEDED |

---

## 集成示例

### 完整工作流

```python
from src.core.assistant import (
    CADAssistant,
    ConversationPersistence,
    SemanticRetriever,
    ResponseQualityEvaluator,
    CADAssistantAPI,
)

# 1. 初始化持久化
persistence = ConversationPersistence(
    storage_path="./data/conversations",
    auto_save=True,
)

# 2. 初始化语义检索
retriever = SemanticRetriever()
retriever.index_knowledge_base(knowledge_items)

# 3. 初始化助手
assistant = CADAssistant()

# 4. 初始化评估器
evaluator = ResponseQualityEvaluator()

# 5. 初始化 API
api = CADAssistantAPI(
    rate_limit_enabled=True,
    requests_per_minute=60,
)

# 6. 创建 Web 应用
from src.core.assistant import create_flask_app
app = create_flask_app(api)
app.run(host="0.0.0.0", port=5000)
```

---

## 文件结构

```
src/core/assistant/
├── __init__.py              # 模块导出
├── persistence.py           # P0: 对话持久化
├── semantic_retrieval.py    # P1: 语义检索
├── quality_evaluation.py    # P2: 质量评估
├── api_service.py           # P3: API 服务
└── ...                      # 其他已有模块

tests/unit/assistant/
├── test_persistence.py      # 27 tests
├── test_semantic_retrieval.py # 28 tests
├── test_quality_evaluation.py # 38 tests
└── test_api_service.py      # 46 tests (44 passed, 2 skipped)
```

---

## 依赖关系

```
P0 (持久化) ──────┐
                  │
P1 (语义检索) ────┼──→ P3 (API 服务)
                  │
P2 (质量评估) ────┘
```

- P3 (API 服务) 集成了 P0-P2 的所有功能
- 各模块可独立使用，也可组合使用
