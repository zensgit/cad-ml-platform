"""
Embedding-based Knowledge Retriever.

Uses sentence embeddings for semantic similarity search,
improving retrieval accuracy beyond keyword matching.
"""

import os
import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np


@dataclass
class EmbeddingConfig:
    """Configuration for embedding retriever."""

    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"  # Good for Chinese
    cache_dir: str = "data/embeddings"
    use_cache: bool = True
    similarity_threshold: float = 0.3
    max_results: int = 10
    normalize_embeddings: bool = True


@dataclass
class KnowledgeItem:
    """A knowledge item with text and metadata."""

    id: str
    text: str
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class EmbeddingProvider:
    """
    Provides text embeddings using sentence-transformers.

    Falls back to simple TF-IDF if sentence-transformers is unavailable.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._fallback_mode = False
        self._init_model()

    def _init_model(self) -> None:
        """Initialize embedding model."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model_name)
            self._fallback_mode = False
        except ImportError:
            # Fallback to simple vectorization
            self._fallback_mode = True
            self._vocab: Dict[str, int] = {}

    def is_available(self) -> bool:
        """Check if embedding model is available."""
        return self._model is not None

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False,
            )
            return np.array(embeddings)
        else:
            return self._fallback_embed(texts)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Simple character n-gram based embedding fallback."""
        # Fixed dimension for consistency
        dim = 512

        embeddings = np.zeros((len(texts), dim))

        for i, text in enumerate(texts):
            ngrams = self._get_ngrams(text)
            for ng in ngrams:
                # Hash to fixed dimension
                idx = hash(ng) % dim
                embeddings[i, idx] += 1

            # Normalize
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm

        return embeddings

    def _get_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Extract character n-grams from text."""
        text = text.lower()
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i + n])
        return ngrams

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return 512  # Fixed fallback dimension


class KnowledgeIndex:
    """
    Index of knowledge items with embeddings for semantic search.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: Optional[EmbeddingConfig] = None,
    ):
        self.config = config or EmbeddingConfig()
        self.embedding_provider = embedding_provider
        self.items: List[KnowledgeItem] = []
        self._embeddings: Optional[np.ndarray] = None

    def add_items(self, items: List[KnowledgeItem]) -> None:
        """Add knowledge items to the index."""
        if not items:
            return

        # Generate embeddings for new items
        texts = [item.text for item in items]
        new_embeddings = self.embedding_provider.embed(texts)

        for item, emb in zip(items, new_embeddings):
            item.embedding = emb
            self.items.append(item)

        # Rebuild combined embedding matrix
        self._rebuild_embeddings()

    def _rebuild_embeddings(self) -> None:
        """Rebuild the embedding matrix."""
        if self.items:
            self._embeddings = np.vstack([item.embedding for item in self.items])
        else:
            self._embeddings = None

    def search(
        self,
        query: str,
        max_results: int = 5,
        threshold: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[KnowledgeItem, float]]:
        """
        Search for similar knowledge items.

        Args:
            query: Search query text
            max_results: Maximum number of results
            threshold: Minimum similarity threshold
            source_filter: Filter by source (optional)

        Returns:
            List of (KnowledgeItem, similarity_score) tuples
        """
        if not self.items or self._embeddings is None:
            return []

        # Embed query
        query_emb = self.embedding_provider.embed_single(query)

        # Compute cosine similarities
        similarities = np.dot(self._embeddings, query_emb)

        # Apply source filter if specified
        if source_filter:
            mask = np.array([item.source == source_filter for item in self.items])
            similarities = np.where(mask, similarities, -1)

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:max_results]

        results = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim >= threshold:
                results.append((self.items[idx], float(sim)))

        return results

    def save(self, path: str) -> None:
        """Save index to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save items metadata
        items_data = []
        for item in self.items:
            items_data.append({
                "id": item.id,
                "text": item.text,
                "source": item.source,
                "data": item.data,
            })

        with open(save_dir / "items.json", "w", encoding="utf-8") as f:
            json.dump(items_data, f, ensure_ascii=False, indent=2)

        # Save embeddings
        if self._embeddings is not None:
            np.save(save_dir / "embeddings.npy", self._embeddings)

    def load(self, path: str) -> bool:
        """Load index from disk."""
        save_dir = Path(path)

        items_path = save_dir / "items.json"
        embeddings_path = save_dir / "embeddings.npy"

        if not items_path.exists():
            return False

        try:
            with open(items_path, "r", encoding="utf-8") as f:
                items_data = json.load(f)

            self.items = []
            for data in items_data:
                self.items.append(KnowledgeItem(
                    id=data["id"],
                    text=data["text"],
                    source=data["source"],
                    data=data.get("data", {}),
                ))

            if embeddings_path.exists():
                self._embeddings = np.load(embeddings_path)
                # Assign embeddings to items
                for i, item in enumerate(self.items):
                    if i < len(self._embeddings):
                        item.embedding = self._embeddings[i]

            return True
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self.items)


class SemanticRetriever:
    """
    Semantic knowledge retriever using embeddings.

    Combines embedding-based similarity search with the existing
    keyword-based retrieval for improved accuracy.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.embedding_provider = EmbeddingProvider(self.config)
        self.index = KnowledgeIndex(self.embedding_provider, self.config)
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the retriever with knowledge base content."""
        if self._initialized:
            return

        # Try to load cached index
        cache_path = self.config.cache_dir
        if self.config.use_cache and self.index.load(cache_path):
            self._initialized = True
            return

        # Build index from knowledge bases
        items = self._build_knowledge_items()
        self.index.add_items(items)

        # Save cache
        if self.config.use_cache:
            self.index.save(cache_path)

        self._initialized = True

    def _build_knowledge_items(self) -> List[KnowledgeItem]:
        """Build knowledge items from all knowledge sources."""
        items = []

        # Materials
        items.extend(self._build_material_items())

        # Tolerances
        items.extend(self._build_tolerance_items())

        # Standards (threads, bearings, seals)
        items.extend(self._build_standard_items())

        # Machining
        items.extend(self._build_machining_items())

        # Design standards
        items.extend(self._build_design_standard_items())

        return items

    def _build_material_items(self) -> List[KnowledgeItem]:
        """Build material knowledge items."""
        items = []

        try:
            from src.core.materials.classifier import MATERIAL_DATABASE
        except ImportError:
            return items

        for grade, info in MATERIAL_DATABASE.items():
            # Create searchable text
            name = info.name if hasattr(info, 'name') else str(info.get('name', ''))
            aliases = info.aliases if hasattr(info, 'aliases') else info.get('aliases', [])
            props = info.properties if hasattr(info, 'properties') else info.get('properties', {})

            text_parts = [
                f"材料 {grade} {name}",
                f"别名: {', '.join(aliases)}" if aliases else "",
            ]

            if hasattr(props, 'tensile_strength') and props.tensile_strength:
                text_parts.append(f"抗拉强度 {props.tensile_strength} MPa")
            if hasattr(props, 'yield_strength') and props.yield_strength:
                text_parts.append(f"屈服强度 {props.yield_strength} MPa")
            if hasattr(props, 'density') and props.density:
                text_parts.append(f"密度 {props.density} g/cm³")
            if hasattr(props, 'hardness') and props.hardness:
                text_parts.append(f"硬度 {props.hardness}")

            text = " ".join(filter(None, text_parts))

            items.append(KnowledgeItem(
                id=f"material_{grade}",
                text=text,
                source="materials",
                data={"grade": grade, "name": name},
            ))

        return items

    def _build_tolerance_items(self) -> List[KnowledgeItem]:
        """Build tolerance knowledge items."""
        items = []

        try:
            from src.core.knowledge.tolerance import get_common_fits, IT_GRADES
        except ImportError:
            return items

        # IT grades
        for grade in IT_GRADES:
            items.append(KnowledgeItem(
                id=f"it_{grade}",
                text=f"IT{grade} 公差等级 标准公差 ISO 286",
                source="tolerance",
                data={"grade": f"IT{grade}"},
            ))

        # Common fits
        fits = get_common_fits()
        for code, data in fits.items():
            name_zh = data.get("name_zh", "")
            fit_type = data.get("fit_type", "")
            items.append(KnowledgeItem(
                id=f"fit_{code}",
                text=f"配合 {code} {name_zh} {fit_type}配合 公差配合",
                source="tolerance",
                data={"fit_code": code, "name": name_zh},
            ))

        return items

    def _build_standard_items(self) -> List[KnowledgeItem]:
        """Build standard parts knowledge items."""
        items = []

        # Threads
        try:
            from src.core.knowledge.standards import METRIC_THREADS
            for designation, spec in METRIC_THREADS.items():
                items.append(KnowledgeItem(
                    id=f"thread_{designation}",
                    text=f"螺纹 {designation} 公制螺纹 底孔 螺距 {spec.pitch}mm",
                    source="threads",
                    data={"designation": designation, "pitch": spec.pitch},
                ))
        except ImportError:
            pass

        # Bearings
        try:
            from src.core.knowledge.standards import DEEP_GROOVE_BEARINGS
            for designation, spec in DEEP_GROOVE_BEARINGS.items():
                items.append(KnowledgeItem(
                    id=f"bearing_{designation}",
                    text=f"轴承 {designation} 深沟球轴承 内径{spec.bore_d}mm 外径{spec.outer_d}mm",
                    source="bearings",
                    data={"designation": designation, "bore": spec.bore_d},
                ))
        except ImportError:
            pass

        return items

    def _build_machining_items(self) -> List[KnowledgeItem]:
        """Build machining knowledge items."""
        items = []

        try:
            from src.core.knowledge.machining import WORKPIECE_MATERIALS, CUTTING_OPERATIONS
        except ImportError:
            return items

        # Workpiece materials
        for key, mat in WORKPIECE_MATERIALS.items():
            items.append(KnowledgeItem(
                id=f"machining_mat_{key}",
                text=f"加工 {mat.name_zh} {mat.name_en} 切削 可加工性{mat.machinability_rating}%",
                source="machining",
                data={"material_key": key, "name": mat.name_zh},
            ))

        # Cutting operations
        for op_key, op in CUTTING_OPERATIONS.items():
            items.append(KnowledgeItem(
                id=f"operation_{op_key}",
                text=f"工序 {op.name_zh} {op.name_en} 切削参数 进给 转速",
                source="machining",
                data={"operation": op_key, "name": op.name_zh},
            ))

        return items

    def _build_design_standard_items(self) -> List[KnowledgeItem]:
        """Build design standards knowledge items."""
        items = []

        try:
            from src.core.knowledge.design_standards import (
                SurfaceFinishGrade,
                SURFACE_FINISH_TABLE,
                GeneralToleranceClass,
                PREFERRED_DIAMETERS,
            )
        except ImportError:
            return items

        # Surface finish grades
        for grade in SurfaceFinishGrade:
            data = SURFACE_FINISH_TABLE.get(grade)
            if data:
                ra = data[0]
                items.append(KnowledgeItem(
                    id=f"surface_{grade.value}",
                    text=f"表面粗糙度 {grade.value} Ra{ra}μm 表面光洁度 ISO 1302",
                    source="design_standards",
                    data={"grade": grade.value, "ra": ra},
                ))

        # General tolerance classes
        for cls in GeneralToleranceClass:
            items.append(KnowledgeItem(
                id=f"tolerance_class_{cls.value}",
                text=f"一般公差 {cls.value}级 ISO 2768 线性公差 角度公差",
                source="design_standards",
                data={"class": cls.value},
            ))

        return items

    def search(
        self,
        query: str,
        max_results: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[KnowledgeItem, float]]:
        """
        Search for relevant knowledge.

        Args:
            query: Search query
            max_results: Maximum results to return
            source_filter: Filter by knowledge source

        Returns:
            List of (item, score) tuples
        """
        if not self._initialized:
            self.initialize()

        return self.index.search(
            query=query,
            max_results=max_results,
            threshold=self.config.similarity_threshold,
            source_filter=source_filter,
        )

    def is_available(self) -> bool:
        """Check if semantic retrieval is available."""
        return self.embedding_provider.is_available()


# Singleton instance
_semantic_retriever: Optional[SemanticRetriever] = None


def get_semantic_retriever(config: Optional[EmbeddingConfig] = None) -> SemanticRetriever:
    """Get or create semantic retriever singleton."""
    global _semantic_retriever
    if _semantic_retriever is None:
        _semantic_retriever = SemanticRetriever(config)
    return _semantic_retriever
