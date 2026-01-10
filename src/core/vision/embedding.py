"""Image embedding and similarity for vision analysis.

Provides:
- Image hash computation
- Perceptual hashing
- Similarity scoring
- Duplicate detection
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .base import VisionDescription, VisionProvider

logger = logging.getLogger(__name__)


class HashAlgorithm(Enum):
    """Supported hash algorithms."""

    SHA256 = "sha256"
    DHASH = "dhash"  # Difference hash (perceptual)
    PHASH = "phash"  # Perceptual hash
    AHASH = "ahash"  # Average hash


class SimilarityMetric(Enum):
    """Similarity metrics."""

    HAMMING = "hamming"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    JACCARD = "jaccard"


@dataclass
class ImageHash:
    """Hash of an image."""

    algorithm: HashAlgorithm
    hash_value: str
    bit_length: int = 64

    def hamming_distance(self, other: "ImageHash") -> int:
        """Calculate Hamming distance to another hash."""
        if self.algorithm != other.algorithm:
            raise ValueError("Cannot compare hashes from different algorithms")

        # For hex hashes, convert to binary
        if self.algorithm == HashAlgorithm.SHA256:
            b1 = bin(int(self.hash_value, 16))[2:].zfill(len(self.hash_value) * 4)
            b2 = bin(int(other.hash_value, 16))[2:].zfill(len(other.hash_value) * 4)
        else:
            b1 = self.hash_value
            b2 = other.hash_value

        return sum(c1 != c2 for c1, c2 in zip(b1, b2))

    def similarity(self, other: "ImageHash") -> float:
        """Calculate similarity (0.0 to 1.0)."""
        distance = self.hamming_distance(other)
        max_distance = self.bit_length
        return 1.0 - (distance / max_distance)


@dataclass
class EmbeddingVector:
    """An embedding vector for an image."""

    vector: List[float]
    model: str = "unknown"
    dimensions: int = 0

    def __post_init__(self) -> None:
        if not self.dimensions:
            self.dimensions = len(self.vector)

    def cosine_similarity(self, other: "EmbeddingVector") -> float:
        """Calculate cosine similarity with another vector."""
        if len(self.vector) != len(other.vector):
            raise ValueError("Vectors must have same dimensions")

        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))
        norm_a = math.sqrt(sum(a * a for a in self.vector))
        norm_b = math.sqrt(sum(b * b for b in other.vector))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def euclidean_distance(self, other: "EmbeddingVector") -> float:
        """Calculate Euclidean distance to another vector."""
        if len(self.vector) != len(other.vector):
            raise ValueError("Vectors must have same dimensions")

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.vector, other.vector)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "vector": self.vector[:10] + ["..."] if len(self.vector) > 10 else self.vector,
        }


@dataclass
class SimilarityResult:
    """Result of a similarity comparison."""

    score: float
    metric: SimilarityMetric
    hash_a: Optional[ImageHash] = None
    hash_b: Optional[ImageHash] = None
    embedding_a: Optional[EmbeddingVector] = None
    embedding_b: Optional[EmbeddingVector] = None

    @property
    def is_similar(self) -> bool:
        """Check if images are similar (threshold: 0.9)."""
        return self.score >= 0.9

    @property
    def is_duplicate(self) -> bool:
        """Check if images are duplicates (threshold: 0.99)."""
        return self.score >= 0.99

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "metric": self.metric.value,
            "is_similar": self.is_similar,
            "is_duplicate": self.is_duplicate,
        }


class HashGenerator(ABC):
    """Abstract base class for hash generators."""

    @abstractmethod
    def hash(self, image_data: bytes) -> ImageHash:
        """Generate hash for image."""
        pass


class CryptographicHashGenerator(HashGenerator):
    """Generate cryptographic hashes (SHA256)."""

    def __init__(self, algorithm: HashAlgorithm = HashAlgorithm.SHA256):
        if algorithm != HashAlgorithm.SHA256:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        self._algorithm = algorithm

    def hash(self, image_data: bytes) -> ImageHash:
        """Generate cryptographic hash."""
        h = hashlib.sha256(image_data)
        bit_length = 256

        return ImageHash(
            algorithm=self._algorithm,
            hash_value=h.hexdigest(),
            bit_length=bit_length,
        )


class PerceptualHashGenerator(HashGenerator):
    """Generate perceptual hashes (dHash, pHash, aHash)."""

    def __init__(
        self,
        algorithm: HashAlgorithm = HashAlgorithm.DHASH,
        hash_size: int = 8,
    ):
        if algorithm not in (HashAlgorithm.DHASH, HashAlgorithm.PHASH, HashAlgorithm.AHASH):
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        self._algorithm = algorithm
        self._hash_size = hash_size

    def hash(self, image_data: bytes) -> ImageHash:
        """Generate perceptual hash."""
        try:
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(image_data)).convert("L")

            if self._algorithm == HashAlgorithm.DHASH:
                hash_value = self._dhash(img)
            elif self._algorithm == HashAlgorithm.PHASH:
                hash_value = self._phash(img)
            else:  # AHASH
                hash_value = self._ahash(img)

            return ImageHash(
                algorithm=self._algorithm,
                hash_value=hash_value,
                bit_length=self._hash_size * self._hash_size,
            )

        except ImportError:
            logger.warning("Pillow not available, falling back to SHA256")
            return CryptographicHashGenerator(HashAlgorithm.SHA256).hash(image_data)

    def _dhash(self, img: Any) -> str:
        """Compute difference hash."""
        # Resize to (hash_size + 1) x hash_size
        img = img.resize((self._hash_size + 1, self._hash_size))
        pixels = list(img.getdata())

        # Compare adjacent pixels
        bits = []
        for row in range(self._hash_size):
            for col in range(self._hash_size):
                idx = row * (self._hash_size + 1) + col
                bits.append("1" if pixels[idx] > pixels[idx + 1] else "0")

        return "".join(bits)

    def _ahash(self, img: Any) -> str:
        """Compute average hash."""
        img = img.resize((self._hash_size, self._hash_size))
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)

        bits = ["1" if p > avg else "0" for p in pixels]
        return "".join(bits)

    def _phash(self, img: Any) -> str:
        """Compute perceptual hash using DCT."""
        # Simplified pHash without full DCT
        # Uses larger resize then takes top-left
        size = self._hash_size * 4
        img = img.resize((size, size))
        pixels = list(img.getdata())

        # Take average of blocks
        block_size = 4
        block_values = []
        for by in range(self._hash_size):
            for bx in range(self._hash_size):
                total = 0
                for y in range(block_size):
                    for x in range(block_size):
                        idx = (by * block_size + y) * size + (bx * block_size + x)
                        total += pixels[idx]
                block_values.append(total / (block_size * block_size))

        avg = sum(block_values) / len(block_values)
        bits = ["1" if v > avg else "0" for v in block_values]
        return "".join(bits)


@dataclass
class ImageRecord:
    """A stored image record for similarity search."""

    image_id: str
    hash: ImageHash
    embedding: Optional[EmbeddingVector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


class SimilarityIndex:
    """
    Index for efficient similarity search.

    Stores image hashes and embeddings for fast lookup.
    """

    def __init__(
        self,
        hash_algorithm: HashAlgorithm = HashAlgorithm.DHASH,
        similarity_threshold: float = 0.9,
    ):
        """
        Initialize similarity index.

        Args:
            hash_algorithm: Algorithm for hashing
            similarity_threshold: Default similarity threshold
        """
        self._records: Dict[str, ImageRecord] = {}
        self._hash_algorithm = hash_algorithm
        self._threshold = similarity_threshold

        # Initialize hash generator
        if hash_algorithm == HashAlgorithm.SHA256:
            self._hasher = CryptographicHashGenerator(hash_algorithm)
        else:
            self._hasher = PerceptualHashGenerator(hash_algorithm)

    def add(
        self,
        image_id: str,
        image_data: bytes,
        embedding: Optional[EmbeddingVector] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ImageRecord:
        """
        Add an image to the index.

        Args:
            image_id: Unique identifier for the image
            image_data: Raw image bytes
            embedding: Optional pre-computed embedding
            metadata: Additional metadata

        Returns:
            Created ImageRecord
        """
        hash_value = self._hasher.hash(image_data)
        record = ImageRecord(
            image_id=image_id,
            hash=hash_value,
            embedding=embedding,
            metadata=metadata or {},
            created_at=datetime.now(),
        )
        self._records[image_id] = record
        return record

    def remove(self, image_id: str) -> bool:
        """Remove an image from the index."""
        if image_id in self._records:
            del self._records[image_id]
            return True
        return False

    def get(self, image_id: str) -> Optional[ImageRecord]:
        """Get an image record by ID."""
        return self._records.get(image_id)

    def find_similar(
        self,
        image_data: bytes,
        threshold: Optional[float] = None,
        limit: int = 10,
    ) -> List[Tuple[ImageRecord, float]]:
        """
        Find similar images.

        Args:
            image_data: Query image bytes
            threshold: Similarity threshold (0.0 to 1.0)
            limit: Maximum results to return

        Returns:
            List of (record, similarity_score) tuples
        """
        threshold = threshold or self._threshold
        query_hash = self._hasher.hash(image_data)

        results = []
        for record in self._records.values():
            try:
                similarity = query_hash.similarity(record.hash)
                if similarity >= threshold:
                    results.append((record, similarity))
            except ValueError:
                continue

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def find_duplicates(
        self,
        image_data: bytes,
    ) -> List[ImageRecord]:
        """
        Find duplicate images (similarity >= 0.99).

        Args:
            image_data: Query image bytes

        Returns:
            List of duplicate image records
        """
        similar = self.find_similar(image_data, threshold=0.99)
        return [record for record, _ in similar]

    def is_duplicate(self, image_data: bytes) -> bool:
        """Check if image is a duplicate."""
        return len(self.find_duplicates(image_data)) > 0

    def size(self) -> int:
        """Return number of indexed images."""
        return len(self._records)

    def clear(self) -> int:
        """Clear all records. Returns count of removed records."""
        count = len(self._records)
        self._records.clear()
        return count


class ImageEmbedder:
    """
    Generate embeddings for images.

    Uses various techniques depending on available libraries.
    """

    def __init__(self, model: str = "simple"):
        """
        Initialize embedder.

        Args:
            model: Embedding model to use
        """
        self._model = model

    def embed(self, image_data: bytes) -> EmbeddingVector:
        """
        Generate embedding for image.

        Args:
            image_data: Raw image bytes

        Returns:
            EmbeddingVector
        """
        if self._model == "simple":
            return self._simple_embed(image_data)
        else:
            return self._simple_embed(image_data)

    def _simple_embed(self, image_data: bytes) -> EmbeddingVector:
        """Generate simple embedding from image statistics."""
        try:
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(image_data))

            # Get basic statistics
            vector = []

            # Color histogram (simplified)
            if img.mode == "RGB":
                for channel in range(3):
                    channel_data = img.split()[channel]
                    hist = channel_data.histogram()
                    # Downsample histogram to 16 bins
                    bin_size = 256 // 16
                    for i in range(16):
                        start = i * bin_size
                        end = start + bin_size
                        vector.append(sum(hist[start:end]) / sum(hist) if sum(hist) > 0 else 0)
            else:
                # Grayscale
                hist = img.histogram()
                bin_size = 256 // 16
                for i in range(16):
                    start = i * bin_size
                    end = start + bin_size
                    vector.append(sum(hist[start:end]) / sum(hist) if sum(hist) > 0 else 0)
                # Pad to same size as RGB
                vector.extend([0] * 32)

            # Add size features
            w, h = img.size
            max_dim = max(w, h)
            vector.extend(
                [
                    w / max_dim,
                    h / max_dim,
                    (w * h) / 1_000_000,  # Megapixels normalized
                    w / h if h > 0 else 1,  # Aspect ratio
                ]
            )

            return EmbeddingVector(
                vector=vector,
                model=self._model,
                dimensions=len(vector),
            )

        except ImportError:
            # Fallback: hash-based embedding
            hash_value = hashlib.sha256(image_data).hexdigest()
            vector = [int(hash_value[i : i + 2], 16) / 255.0 for i in range(0, 64, 2)]
            return EmbeddingVector(
                vector=vector,
                model="hash_fallback",
                dimensions=len(vector),
            )


class SimilarityVisionProvider:
    """
    Wrapper that adds similarity checking to VisionProvider.

    Can detect duplicates and skip redundant processing.
    """

    def __init__(
        self,
        provider: VisionProvider,
        index: SimilarityIndex,
        skip_duplicates: bool = True,
        cache_results: bool = True,
    ):
        """
        Initialize similarity provider.

        Args:
            provider: The underlying vision provider
            index: SimilarityIndex for duplicate detection
            skip_duplicates: Skip processing for duplicates
            cache_results: Cache results for similar images
        """
        self._provider = provider
        self._index = index
        self._skip_duplicates = skip_duplicates
        self._cache_results = cache_results
        self._result_cache: Dict[str, VisionDescription] = {}

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        check_similarity: bool = True,
    ) -> Tuple[VisionDescription, Optional[SimilarityResult]]:
        """
        Analyze image with similarity checking.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description
            check_similarity: Whether to check for similar images

        Returns:
            Tuple of (VisionDescription, SimilarityResult or None)
        """
        similarity_result = None

        if check_similarity and self._skip_duplicates:
            # Check for duplicates
            duplicates = self._index.find_duplicates(image_data)
            if duplicates:
                dup = duplicates[0]
                # Return cached result if available
                if self._cache_results and dup.image_id in self._result_cache:
                    result = self._result_cache[dup.image_id]
                    similarity_result = SimilarityResult(
                        score=1.0,
                        metric=SimilarityMetric.HAMMING,
                    )
                    return result, similarity_result

        # Process normally
        result = await self._provider.analyze_image(image_data, include_description)

        # Add to index and cache
        import uuid

        image_id = str(uuid.uuid4())[:8]
        self._index.add(image_id, image_data)

        if self._cache_results:
            self._result_cache[image_id] = result

        return result, similarity_result

    def find_similar(
        self,
        image_data: bytes,
        threshold: float = 0.9,
        limit: int = 10,
    ) -> List[Tuple[ImageRecord, float]]:
        """Find similar images in the index."""
        return self._index.find_similar(image_data, threshold, limit)

    def is_duplicate(self, image_data: bytes) -> bool:
        """Check if image is a duplicate."""
        return self._index.is_duplicate(image_data)

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name

    @property
    def similarity_index(self) -> SimilarityIndex:
        """Get the similarity index."""
        return self._index


# Global similarity index
_global_index: Optional[SimilarityIndex] = None


def get_similarity_index() -> SimilarityIndex:
    """
    Get the global similarity index instance.

    Returns:
        SimilarityIndex singleton
    """
    global _global_index
    if _global_index is None:
        _global_index = SimilarityIndex()
    return _global_index


def create_similarity_provider(
    provider: VisionProvider,
    index: Optional[SimilarityIndex] = None,
    hash_algorithm: HashAlgorithm = HashAlgorithm.DHASH,
    skip_duplicates: bool = True,
) -> SimilarityVisionProvider:
    """
    Factory to create a similarity-checking provider wrapper.

    Args:
        provider: The underlying vision provider
        index: Optional existing similarity index
        hash_algorithm: Hash algorithm for similarity
        skip_duplicates: Skip processing for duplicates

    Returns:
        SimilarityVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> similar = create_similarity_provider(provider)
        >>> result, sim_result = await similar.analyze_image(image_bytes)
        >>> if similar.is_duplicate(new_image):
        ...     print("Duplicate detected!")
    """
    if index is None:
        index = SimilarityIndex(hash_algorithm=hash_algorithm)

    return SimilarityVisionProvider(
        provider=provider,
        index=index,
        skip_duplicates=skip_duplicates,
    )
