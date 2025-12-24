"""
Feature Cache Manager.

Handles caching of expensive feature extraction (especially 3D B-Rep).
"""

import hashlib
import json
from typing import Any, Dict, List, Optional


class FeatureCache:
    """
    Simple In-Memory LRU Cache for Feature Vectors.
    In production, this should wrap Redis.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache: Dict[str, Any] = {}
        self.lru: List[str] = []

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.lru.remove(key)
            self.lru.append(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any):
        if key in self.cache:
            self.lru.remove(key)
        elif len(self.cache) >= self.capacity:
            # Evict oldest
            oldest = self.lru.pop(0)
            del self.cache[oldest]

        self.cache[key] = value
        self.lru.append(key)

    def generate_key(self, content: bytes, version: str) -> str:
        """Generate a stable key based on content hash and feature version."""
        h = hashlib.sha256(content).hexdigest()
        return f"feat:{version}:{h}"


# Global instance
_feature_cache = FeatureCache()


def get_feature_cache():
    return _feature_cache
