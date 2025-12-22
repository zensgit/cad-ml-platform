"""
Abstract Base Class for Vector Stores.
Allows interchangeable backends (Faiss, Memory, Milvus).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

class BaseVectorStore(ABC):
    
    @abstractmethod
    def add(self, id: str, vector: List[float], meta: Optional[Dict[str, Any]] = None) -> bool:
        pass
        
    @abstractmethod
    def search(self, vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Returns list of (id, score) tuples."""
        pass
        
    @abstractmethod
    def get_meta(self, id: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def size(self) -> int:
        pass
    
    @abstractmethod
    def save(self, path: str):
        pass
    
    @abstractmethod
    def load(self, path: str):
        pass
