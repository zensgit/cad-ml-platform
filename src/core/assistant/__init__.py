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
    - ResponseGenerator: Generate response using LLM with domain context
"""

from .query_analyzer import QueryAnalyzer, QueryIntent, AnalyzedQuery
from .knowledge_retriever import KnowledgeRetriever, RetrievalResult
from .context_assembler import ContextAssembler, AssembledContext
from .assistant import CADAssistant, AssistantConfig

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
]
