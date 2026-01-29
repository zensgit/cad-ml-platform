"""
CAD-ML Intelligent Assistant Module.

Provides a RAG (Retrieval-Augmented Generation) based intelligent assistant
for CAD design and manufacturing knowledge queries.

Architecture:
    User Query → Query Analysis → Knowledge Retrieval → Context Assembly → LLM → Response

Components:
    - QueryAnalyzer: Analyze user intent and extract keywords
    - KnowledgeRetriever: Retrieve relevant knowledge from domain databases
    - ContextAssembler: Assemble retrieved knowledge into prompt context
    - LLM Providers: Multiple LLM integrations (Claude, GPT, Qwen, Ollama)
"""

from .query_analyzer import QueryAnalyzer, QueryIntent, AnalyzedQuery
from .knowledge_retriever import KnowledgeRetriever, RetrievalResult, RetrievalMode, RetrievalSource
from .context_assembler import ContextAssembler, AssembledContext
from .assistant import CADAssistant, AssistantConfig, LLMProvider
from .llm_providers import (
    LLMConfig,
    BaseLLMProvider,
    ClaudeProvider,
    OpenAIProvider,
    QwenProvider,
    OllamaProvider,
    OfflineProvider,
    get_provider,
    get_best_available_provider,
)
from .embedding_retriever import (
    EmbeddingConfig,
    EmbeddingProvider,
    KnowledgeIndex,
    KnowledgeItem,
    SemanticRetriever,
    get_semantic_retriever,
)
from .conversation import (
    ConversationManager,
    Conversation,
    ConversationContext,
    Message,
    MessageRole,
    get_conversation_manager,
)
from .persistence import (
    StorageBackend,
    JSONStorageBackend,
    SQLiteStorageBackend,
    ConversationPersistence,
)
from .semantic_retrieval import (
    EmbeddingResult,
    SemanticSearchResult,
    EmbeddingProvider as SemanticEmbeddingProvider,
    SimpleEmbeddingProvider,
    SentenceTransformerProvider,
    VectorStore,
    SemanticRetriever as VectorSemanticRetriever,
    create_semantic_retriever,
)
from .quality_evaluation import (
    QualityDimension,
    DimensionScore,
    EvaluationResult,
    ResponseQualityEvaluator,
    EvaluationHistory,
)
from .api_service import (
    APIError,
    ValidationError,
    NotFoundError,
    RateLimitError,
    APIRequest,
    APIResponse,
    AskRequest,
    ConversationRequest,
    EvaluationRequest,
    RateLimiter,
    CADAssistantAPI,
    create_flask_app,
    create_fastapi_app,
)

__all__ = [
    # Query Analysis
    "QueryAnalyzer",
    "QueryIntent",
    "AnalyzedQuery",
    # Knowledge Retrieval
    "KnowledgeRetriever",
    "RetrievalResult",
    "RetrievalMode",
    "RetrievalSource",
    # Semantic Retrieval
    "EmbeddingConfig",
    "EmbeddingProvider",
    "KnowledgeIndex",
    "KnowledgeItem",
    "SemanticRetriever",
    "get_semantic_retriever",
    # Conversation
    "ConversationManager",
    "Conversation",
    "ConversationContext",
    "Message",
    "MessageRole",
    "get_conversation_manager",
    # Persistence
    "StorageBackend",
    "JSONStorageBackend",
    "SQLiteStorageBackend",
    "ConversationPersistence",
    # Vector Semantic Retrieval
    "EmbeddingResult",
    "SemanticSearchResult",
    "SemanticEmbeddingProvider",
    "SimpleEmbeddingProvider",
    "SentenceTransformerProvider",
    "VectorStore",
    "VectorSemanticRetriever",
    "create_semantic_retriever",
    # Quality Evaluation
    "QualityDimension",
    "DimensionScore",
    "EvaluationResult",
    "ResponseQualityEvaluator",
    "EvaluationHistory",
    # API Service
    "APIError",
    "ValidationError",
    "NotFoundError",
    "RateLimitError",
    "APIRequest",
    "APIResponse",
    "AskRequest",
    "ConversationRequest",
    "EvaluationRequest",
    "RateLimiter",
    "CADAssistantAPI",
    "create_flask_app",
    "create_fastapi_app",
    # Context Assembly
    "ContextAssembler",
    "AssembledContext",
    # Main Assistant
    "CADAssistant",
    "AssistantConfig",
    "LLMProvider",
    # LLM Providers
    "LLMConfig",
    "BaseLLMProvider",
    "ClaudeProvider",
    "OpenAIProvider",
    "QwenProvider",
    "OllamaProvider",
    "OfflineProvider",
    "get_provider",
    "get_best_available_provider",
]
