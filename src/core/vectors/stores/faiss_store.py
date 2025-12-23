"""
Faiss Vector Store Wrapper.
Provides high-performance indexing for Public/Private libraries.
"""

import os
import logging
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from src.core.vectors.stores.base import BaseVectorStore

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("Faiss not installed. FaissStore will fail if used.")

class FaissStore(BaseVectorStore):
    def __init__(self, dimension: int = 128, index_key: str = "Flat"):
        if not HAS_FAISS:
            raise ImportError("Faiss required")

        self.dimension = dimension
        self.index = faiss.index_factory(dimension, index_key, faiss.METRIC_INNER_PRODUCT)
        self.id_map: Dict[int, str] = {} # Int ID -> Str ID
        self.rev_map: Dict[str, int] = {} # Str ID -> Int ID
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.counter = 0

    def add(self, id: str, vector: List[float], meta: Optional[Dict[str, Any]] = None) -> bool:
        if len(vector) != self.dimension:
            logger.error(f"Vector dim {len(vector)} != {self.dimension}")
            return False

        vec_np = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vec_np)

        self.index.add(vec_np)

        self.id_map[self.counter] = id
        self.rev_map[id] = self.counter
        if meta:
            self.metadata[id] = meta

        self.counter += 1
        return True

    def search(self, vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:
            return []

        vec_np = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vec_np)

        scores, ids = self.index.search(vec_np, top_k)

        results = []
        for score, int_id in zip(scores[0], ids[0]):
            if int_id != -1 and int_id in self.id_map:
                str_id = self.id_map[int_id]
                results.append((str_id, float(score)))

        return results

    def get_meta(self, id: str) -> Optional[Dict[str, Any]]:
        return self.metadata.get(id)

    def size(self) -> int:
        return self.index.ntotal

    def save(self, path: str):
        # Save index + auxiliary data
        base = os.path.splitext(path)[0]
        faiss.write_index(self.index, f"{base}.index")
        with open(f"{base}.meta", 'wb') as f:
            pickle.dump({
                "id_map": self.id_map,
                "metadata": self.metadata,
                "counter": self.counter
            }, f)

    def load(self, path: str):
        base = os.path.splitext(path)[0]
        if os.path.exists(f"{base}.index"):
            self.index = faiss.read_index(f"{base}.index")
        if os.path.exists(f"{base}.meta"):
            with open(f"{base}.meta", 'rb') as f:
                data = pickle.load(f)
                self.id_map = data["id_map"]
                self.metadata = data["metadata"]
                self.counter = data["counter"]
