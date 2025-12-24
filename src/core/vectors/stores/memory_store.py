"""
Simple In-Memory Vector Store.
Useful for testing or small-scale private indexes.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.core.vectors.stores.base import BaseVectorStore

logger = logging.getLogger(__name__)


class MemoryVectorStore(BaseVectorStore):
    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def add(self, id: str, vector: List[float], meta: Optional[Dict[str, Any]] = None) -> bool:
        self.vectors[id] = vector
        if meta:
            self.metadata[id] = meta
        return True

    def search(self, vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        # Cosine similarity
        results = []

        # Precompute query norm
        q_norm = sum(x * x for x in vector) ** 0.5
        if q_norm == 0:
            return []

        for vid, v in self.vectors.items():
            dot = sum(a * b for a, b in zip(vector, v))
            v_norm = sum(x * x for x in v) ** 0.5
            if v_norm == 0:
                continue
            score = dot / (q_norm * v_norm)
            results.append((vid, score))

        # Sort desc
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_meta(self, id: str) -> Optional[Dict[str, Any]]:
        return self.metadata.get(id)

    def size(self) -> int:
        return len(self.vectors)

    def save(self, path: str):
        data = {"vectors": self.vectors, "metadata": self.metadata}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                self.vectors = data.get("vectors", {})
                self.metadata = data.get("metadata", {})
        except Exception as e:
            logger.error(f"Failed to load memory store: {e}")
