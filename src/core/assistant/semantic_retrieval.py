"""
Semantic Retrieval Module.

Provides vector-based semantic search for knowledge retrieval,
enhancing keyword matching with embedding-based similarity.
"""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""

    text: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticSearchResult:
    """Result of a semantic search."""

    text: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class SimpleEmbeddingProvider(EmbeddingProvider):
    """
    Simple TF-IDF based embedding provider.

    Uses character n-grams and term frequency for embedding.
    No external dependencies required.
    """

    def __init__(
        self,
        dimension: int = 256,
        ngram_range: Tuple[int, int] = (2, 4),
    ):
        """
        Initialize simple embedding provider.

        Args:
            dimension: Output embedding dimension
            ngram_range: Character n-gram range (min, max)
        """
        self._dimension = dimension
        self.ngram_range = ngram_range
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_ngrams(self, text: str) -> List[str]:
        """Extract character n-grams from text."""
        ngrams = []
        text = text.lower()
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i : i + n])
        return ngrams

    def _hash_to_index(self, ngram: str) -> int:
        """Hash n-gram to embedding index."""
        # MD5 used only for hash distribution, not security - nosec B324
        hash_val = int(hashlib.md5(ngram.encode(), usedforsecurity=False).hexdigest(), 16)
        return hash_val % self._dimension

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using hashed n-gram TF."""
        vector = [0.0] * self._dimension
        ngrams = self._get_ngrams(text)

        if not ngrams:
            return vector

        # Count n-gram frequencies
        freq: Dict[str, int] = {}
        for ng in ngrams:
            freq[ng] = freq.get(ng, 0) + 1

        # Build vector
        for ng, count in freq.items():
            idx = self._hash_to_index(ng)
            tf = count / len(ngrams)
            vector[idx] += tf

        # L2 normalize
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_text(text) for text in texts]


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Sentence Transformers based embedding provider.

    Requires sentence-transformers library.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
    ):
        """
        Initialize Sentence Transformer provider.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ("cpu", "cuda", "mps")
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using sentence transformer."""
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class VectorStore:
    """
    Simple in-memory vector store with persistence.

    Supports adding, searching, and managing embeddings.
    """

    def __init__(
        self,
        dimension: int,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize vector store.

        Args:
            dimension: Embedding dimension
            storage_path: Path for persistence (optional)
        """
        self.dimension = dimension
        self.storage_path = Path(storage_path) if storage_path else None
        self._vectors: List[List[float]] = []
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

        if self.storage_path and self.storage_path.exists():
            self._load()

    def add(
        self,
        text: str,
        vector: List[float],
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a vector to the store.

        Args:
            text: Original text
            vector: Embedding vector
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Index of added vector
        """
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: {len(vector)} != {self.dimension}")

        self._vectors.append(vector)
        self._texts.append(text)
        self._sources.append(source)
        self._metadata.append(metadata or {})

        return len(self._vectors) - 1

    def add_batch(
        self,
        texts: List[str],
        vectors: List[List[float]],
        sources: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Add multiple vectors to the store."""
        if sources is None:
            sources = [""] * len(texts)
        if metadata is None:
            metadata = [{}] * len(texts)

        indices = []
        for text, vector, source, meta in zip(texts, vectors, sources, metadata):
            idx = self.add(text, vector, source, meta)
            indices.append(idx)

        return indices

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        min_score: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> List[SemanticSearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            min_score: Minimum similarity score
            source_filter: Filter by source

        Returns:
            List of search results sorted by score
        """
        if not self._vectors:
            return []

        # Calculate cosine similarities
        query_np = np.array(query_vector)
        query_norm = np.linalg.norm(query_np)
        if query_norm == 0:
            return []

        scores = []
        for i, vec in enumerate(self._vectors):
            if source_filter and self._sources[i] != source_filter:
                continue

            vec_np = np.array(vec)
            vec_norm = np.linalg.norm(vec_np)
            if vec_norm == 0:
                continue

            score = float(np.dot(query_np, vec_np) / (query_norm * vec_norm))
            if score >= min_score:
                scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:top_k]

        return [
            SemanticSearchResult(
                text=self._texts[idx],
                score=score,
                source=self._sources[idx],
                metadata=self._metadata[idx],
            )
            for idx, score in scores
        ]

    def save(self) -> bool:
        """Save store to disk."""
        if not self.storage_path:
            return False

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "dimension": self.dimension,
                "vectors": self._vectors,
                "texts": self._texts,
                "sources": self._sources,
                "metadata": self._metadata,
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            return True
        except IOError:
            return False

    def _load(self) -> bool:
        """Load store from disk."""
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data["dimension"] != self.dimension:
                return False

            self._vectors = data["vectors"]
            self._texts = data["texts"]
            self._sources = data.get("sources", [""] * len(self._texts))
            self._metadata = data.get("metadata", [{}] * len(self._texts))
            return True
        except (IOError, json.JSONDecodeError, KeyError):
            return False

    def clear(self) -> None:
        """Clear all vectors from the store."""
        self._vectors = []
        self._texts = []
        self._sources = []
        self._metadata = []

    def __len__(self) -> int:
        return len(self._vectors)


class SemanticRetriever:
    """
    Semantic retrieval system for knowledge base.

    Combines keyword matching with semantic similarity
    for improved retrieval accuracy.

    Example:
        >>> retriever = SemanticRetriever()
        >>> retriever.index_knowledge(knowledge_items)
        >>> results = retriever.search("304不锈钢的强度")
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        storage_path: Optional[str] = None,
        hybrid_weight: float = 0.7,  # Weight for semantic vs keyword
    ):
        """
        Initialize semantic retriever.

        Args:
            embedding_provider: Provider for generating embeddings
            storage_path: Path for vector store persistence
            hybrid_weight: Weight for semantic score (0-1)
        """
        self.embedding_provider = embedding_provider or SimpleEmbeddingProvider()
        self.hybrid_weight = hybrid_weight

        dimension = self.embedding_provider.dimension
        self.vector_store = VectorStore(
            dimension=dimension,
            storage_path=storage_path,
        )

        self._indexed_count = 0

    def index_text(
        self,
        text: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Index a single text.

        Args:
            text: Text to index
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Index of added item
        """
        vector = self.embedding_provider.embed_text(text)
        idx = self.vector_store.add(text, vector, source, metadata)
        self._indexed_count += 1
        return idx

    def index_batch(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """
        Index multiple texts.

        Args:
            texts: Texts to index
            sources: Source identifiers
            metadata: Additional metadata per text

        Returns:
            Indices of added items
        """
        vectors = self.embedding_provider.embed_batch(texts)
        indices = self.vector_store.add_batch(texts, vectors, sources, metadata)
        self._indexed_count += len(texts)
        return indices

    def index_knowledge_base(
        self,
        knowledge_items: List[Dict[str, Any]],
        text_key: str = "content",
        source_key: str = "source",
    ) -> int:
        """
        Index items from knowledge base.

        Args:
            knowledge_items: List of knowledge items
            text_key: Key for text content
            source_key: Key for source identifier

        Returns:
            Number of items indexed
        """
        texts = []
        sources = []
        metadata_list = []

        for item in knowledge_items:
            text = item.get(text_key, "")
            if not text:
                continue

            texts.append(text)
            sources.append(item.get(source_key, ""))
            metadata_list.append({k: v for k, v in item.items() if k not in [text_key]})

        if texts:
            self.index_batch(texts, sources, metadata_list)

        return len(texts)

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.3,
        source_filter: Optional[str] = None,
    ) -> List[SemanticSearchResult]:
        """
        Search for relevant content.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity score
            source_filter: Filter by source

        Returns:
            List of search results
        """
        query_vector = self.embedding_provider.embed_text(query)
        return self.vector_store.search(
            query_vector,
            top_k=top_k,
            min_score=min_score,
            source_filter=source_filter,
        )

    def hybrid_search(
        self,
        query: str,
        keyword_results: List[Dict[str, Any]],
        top_k: int = 10,
        text_key: str = "content",
    ) -> List[SemanticSearchResult]:
        """
        Combine semantic and keyword search results.

        Args:
            query: Search query
            keyword_results: Results from keyword search
            top_k: Number of results
            text_key: Key for text content in keyword results

        Returns:
            Combined and re-ranked results
        """
        # Get semantic results
        semantic_results = self.search(query, top_k=top_k * 2)
        semantic_map = {r.text: r.score for r in semantic_results}

        # Combine scores
        combined: Dict[str, Tuple[float, Any]] = {}

        # Add semantic results
        for result in semantic_results:
            combined[result.text] = (
                result.score * self.hybrid_weight,
                result.metadata,
                result.source,
            )

        # Add keyword results with boost
        keyword_weight = 1.0 - self.hybrid_weight
        for i, item in enumerate(keyword_results):
            text = item.get(text_key, "")
            if not text:
                continue

            # Score based on rank position
            keyword_score = 1.0 - (i / (len(keyword_results) + 1))

            if text in combined:
                # Boost existing semantic result
                semantic_score, meta, source = combined[text]
                combined[text] = (
                    semantic_score + keyword_score * keyword_weight,
                    meta,
                    source,
                )
            else:
                combined[text] = (
                    keyword_score * keyword_weight,
                    item,
                    item.get("source", ""),
                )

        # Sort and return top_k
        sorted_results = sorted(combined.items(), key=lambda x: x[1][0], reverse=True)

        return [
            SemanticSearchResult(
                text=text,
                score=score,
                source=source,
                metadata=meta if isinstance(meta, dict) else {},
            )
            for text, (score, meta, source) in sorted_results[:top_k]
        ]

    def save(self) -> bool:
        """Save the vector store to disk."""
        return self.vector_store.save()

    def clear(self) -> None:
        """Clear all indexed content."""
        self.vector_store.clear()
        self._indexed_count = 0

    @property
    def indexed_count(self) -> int:
        """Number of indexed items."""
        return len(self.vector_store)


def create_semantic_retriever(
    use_transformers: bool = False,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    storage_path: Optional[str] = None,
    hybrid_weight: float = 0.7,
) -> SemanticRetriever:
    """
    Factory function to create a semantic retriever.

    Args:
        use_transformers: Use sentence-transformers (requires library)
        model_name: Model name if using transformers
        storage_path: Path for vector store persistence
        hybrid_weight: Weight for semantic vs keyword search

    Returns:
        Configured SemanticRetriever
    """
    if use_transformers:
        provider = SentenceTransformerProvider(model_name=model_name)
    else:
        provider = SimpleEmbeddingProvider()

    return SemanticRetriever(
        embedding_provider=provider,
        storage_path=storage_path,
        hybrid_weight=hybrid_weight,
    )
