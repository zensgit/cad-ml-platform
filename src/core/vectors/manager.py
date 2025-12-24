"""
Layered Vector Manager.

Manages multiple vector indexes (Public/Private) and merges search results.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.core.vectors.stores.base import BaseVectorStore
from src.core.vectors.stores.memory_store import MemoryVectorStore

# Optional Faiss import inside logic

logger = logging.getLogger(__name__)


class LayeredVectorManager:
    """
    Orchestrates searches across Public (Shared) and Private (Tenant) indexes.
    """

    def __init__(self, public_path: Optional[str] = None):
        self.partitions: Dict[str, BaseVectorStore] = {}

        # 1. Initialize Public Index (Read-Only)
        self.public_path = public_path or os.getenv("PUBLIC_INDEX_PATH", "data/public_index.bin")
        self._load_public_index()

        # 2. Initialize Private Index (Default Tenant)
        # In multi-tenant env, this would be a map of tenant_id -> store
        self.private_store = MemoryVectorStore()  # Start with memory, can swap to Faiss
        self.partitions["private"] = self.private_store

    def _load_public_index(self):
        # Try to load Faiss if available and file exists
        try:
            from src.core.vectors.stores.faiss_store import HAS_FAISS, FaissStore

            if HAS_FAISS and os.path.exists(f"{os.path.splitext(self.public_path)[0]}.index"):
                store = FaissStore()
                store.load(self.public_path)
                self.partitions["public"] = store
                logger.info(f"Loaded Public Index from {self.public_path} ({store.size()} vectors)")
            else:
                logger.info("Public Index not found or Faiss missing. Skipping.")
        except Exception as e:
            logger.error(f"Failed to load public index: {e}")

    def search(
        self,
        vector: List[float],
        top_k: int = 5,
        partitions: List[str] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Search across specified partitions.
        Returns: [(id, score, partition_name), ...]
        """
        if partitions is None:
            partitions = ["public", "private"]

        all_results = []

        for p_name in partitions:
            store = self.partitions.get(p_name)
            if store:
                hits = store.search(vector, top_k)
                # Tag with partition name
                for hit_id, score in hits:
                    all_results.append((hit_id, score, p_name))

        # Global Sort
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def add_private(self, id: str, vector: List[float], meta: Optional[Dict[str, Any]] = None):
        """Add to private index."""
        self.private_store.add(id, vector, meta)

    def get_meta(self, id: str, partition: str) -> Optional[Dict[str, Any]]:
        store = self.partitions.get(partition)
        if store:
            return store.get_meta(id)
        return None


# Singleton
_manager = LayeredVectorManager()


def get_vector_manager():
    return _manager
