"""
Domain Embedding Model — inference wrapper.

Loads a fine-tuned manufacturing-domain embedding model (or falls back to
the base ``paraphrase-multilingual-MiniLM-L12-v2``, or to a lightweight
TF-IDF vectoriser when no GPU / sentence-transformers is available).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Re-use the lightweight fallback from the trainer module so we have a single
# implementation.
from src.ml.embeddings.trainer import _TfidfFallbackModel

_BASE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class DomainEmbeddingModel:
    """Manufacturing domain-optimised embedding model.

    Resolution order when choosing which model to load:

    1. Explicit *model_path* pointing to a fine-tuned model directory.
    2. The default base model via ``sentence-transformers``.
    3. A TF-IDF character-ngram fallback (always available).
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model: Any = None
        self._model_name: str = ""
        self._is_fallback: bool = False
        self._is_fine_tuned: bool = False
        self._corpus_size: int = 0
        self._load(model_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, model_path: Optional[str]) -> None:
        """Attempt to load a model following the resolution order."""
        # 1. Fine-tuned model from explicit path
        if model_path and Path(model_path).is_dir():
            if self._try_load_sentence_transformer(model_path, fine_tuned=True):
                return
            # Check for TF-IDF fallback marker
            if (Path(model_path) / "_tfidf_fallback_marker").exists():
                self._model = _TfidfFallbackModel()
                self._model_name = "tfidf-fallback (fine-tuned dir)"
                self._is_fallback = True
                self._is_fine_tuned = False
                logger.info("Loaded TF-IDF fallback from %s", model_path)
                return

        # 2. Base model via sentence-transformers
        if self._try_load_sentence_transformer(_BASE_MODEL_NAME, fine_tuned=False):
            return

        # 3. TF-IDF fallback
        logger.warning(
            "sentence-transformers not available — using TF-IDF fallback"
        )
        self._model = _TfidfFallbackModel()
        self._model_name = "tfidf-fallback"
        self._is_fallback = True
        self._is_fine_tuned = False

    def _try_load_sentence_transformer(
        self, name_or_path: str, fine_tuned: bool
    ) -> bool:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(name_or_path)
            self._model_name = name_or_path
            self._is_fallback = False
            self._is_fine_tuned = fine_tuned
            logger.info("Loaded model: %s (fine_tuned=%s)", name_or_path, fine_tuned)
            return True
        except ImportError:
            return False
        except Exception:
            logger.debug("Failed to load model from %s", name_or_path, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode *texts* into dense embeddings.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(texts), embedding_dim)``.
        """
        if self._is_fallback:
            return self._model.encode(texts, normalize_embeddings=normalize)

        return np.asarray(
            self._model.encode(
                texts,
                normalize_embeddings=normalize,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        )

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        embs = self.encode([text_a, text_b], normalize=True)
        return float(np.dot(embs[0], embs[1]))

    def search(
        self,
        query: str,
        corpus: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search *corpus* for the most similar texts to *query*.

        Returns
        -------
        list of dict
            Each dict has ``text``, ``score``, ``rank``.
        """
        if not corpus:
            return []

        query_emb = self.encode([query], normalize=True)  # (1, dim)
        corpus_emb = self.encode(corpus, normalize=True)   # (N, dim)

        scores = corpus_emb @ query_emb.T  # (N, 1)
        scores = scores.flatten()

        top_k = min(top_k, len(corpus))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {"text": corpus[idx], "score": float(scores[idx]), "rank": rank}
            )
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        if hasattr(self._model, "get_sentence_embedding_dimension"):
            dim = self._model.get_sentence_embedding_dimension()
        else:
            dim = 512  # fallback dimension

        return {
            "name": self._model_name,
            "dimension": dim,
            "fine_tuned": self._is_fine_tuned,
            "fallback": self._is_fallback,
            "corpus_size": self._corpus_size,
        }

    @property
    def embedding_dim(self) -> int:
        if hasattr(self._model, "get_sentence_embedding_dimension"):
            return int(self._model.get_sentence_embedding_dimension())
        return 512
