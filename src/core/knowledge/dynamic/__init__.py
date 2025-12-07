"""
Dynamic Knowledge Base System.

Provides hot-reloadable, externally configurable mechanical knowledge
that can be updated at runtime without code changes.

Components:
- KnowledgeStore: Abstract storage interface
- JSONKnowledgeStore: JSON file-based storage
- RedisKnowledgeStore: Redis-based storage (optional)
- KnowledgeManager: Central management and hot-reload
- KnowledgeAPI: REST API for CRUD operations
"""

from src.core.knowledge.dynamic.models import (
    KnowledgeEntry,
    MaterialRule,
    PrecisionRule,
    StandardRule,
    FunctionalFeatureRule,
    AssemblyRule,
    ManufacturingRule,
    GeometryPattern,
    KnowledgeCategory,
)
from src.core.knowledge.dynamic.store import (
    KnowledgeStore,
    JSONKnowledgeStore,
)
from src.core.knowledge.dynamic.manager import KnowledgeManager
from src.core.knowledge.dynamic.loader import DynamicKnowledgeBase

__all__ = [
    # Models
    "KnowledgeEntry",
    "MaterialRule",
    "PrecisionRule",
    "StandardRule",
    "FunctionalFeatureRule",
    "AssemblyRule",
    "ManufacturingRule",
    "GeometryPattern",
    "KnowledgeCategory",
    # Storage
    "KnowledgeStore",
    "JSONKnowledgeStore",
    # Management
    "KnowledgeManager",
    "DynamicKnowledgeBase",
]
