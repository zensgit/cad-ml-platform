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
from .knowledge_retriever import KnowledgeRetriever, RetrievalResult
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

__all__ = [
    # Query Analysis
    "QueryAnalyzer",
    "QueryIntent",
    "AnalyzedQuery",
    # Knowledge Retrieval
    "KnowledgeRetriever",
    "RetrievalResult",
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
