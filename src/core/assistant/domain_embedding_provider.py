"""
Domain Embedding Provider.

Wraps the fine-tuned manufacturing domain embedding model
(:class:`~src.ml.embeddings.model.DomainEmbeddingModel`) as an
:class:`EmbeddingProvider` so it can be plugged into the
:class:`SemanticRetriever` pipeline.

Falls back gracefully to a zero-vector provider when the domain model
(or its dependencies) is unavailable.
"""

import logging
from typing import List

from .semantic_retrieval import EmbeddingProvider

logger = logging.getLogger(__name__)


class DomainEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using the fine-tuned manufacturing domain model.

    Parameters
    ----------
    model_path : str
        Path to the fine-tuned model directory.  Defaults to the
        ``models/embeddings/manufacturing_v2/`` checkpoint shipped with
        the repository.
    """

    def __init__(self, model_path: str = "models/embeddings/manufacturing_v2"):
        # Try to load DomainEmbeddingModel, fall back gracefully
        try:
            from src.ml.embeddings.model import DomainEmbeddingModel

            self._model = DomainEmbeddingModel(model_path=model_path)
            self._dim = self._model.get_model_info().get("dimension", 384)
            self._available = True
            logger.info(
                "DomainEmbeddingProvider loaded (dim=%d, model=%s)",
                self._dim,
                self._model.get_model_info().get("name", "unknown"),
            )
        except Exception:
            logger.warning(
                "DomainEmbeddingModel unavailable — "
                "DomainEmbeddingProvider will return zero vectors",
                exc_info=True,
            )
            self._model = None  # type: ignore[assignment]
            self._available = False
            self._dim = 384

    # ------------------------------------------------------------------
    # EmbeddingProvider interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self._available:
            return [0.0] * self._dim
        arr = self._model.encode([text], normalize=True)
        return arr[0].tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self._available:
            return [[0.0] * self._dim for _ in texts]
        arr = self._model.encode(texts, normalize=True)
        return arr.tolist()
