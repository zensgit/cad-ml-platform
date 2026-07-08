"""
Domain Embedding Provider.

Wraps the fine-tuned manufacturing domain embedding model
(:class:`~src.ml.embeddings.model.DomainEmbeddingModel`) as an
:class:`EmbeddingProvider` so it can be plugged into the
:class:`SemanticRetriever` pipeline.

.. warning:: **This provider is currently serving TF-IDF, not a fine-tuned model.**

   ``models/embeddings/manufacturing_v2/`` ships a tokenizer, a config, and a
   training corpus, but **no encoder weights** (no ``*.bin`` / ``*.safetensors``
   / ``*.pth``). :class:`DomainEmbeddingModel` has a three-tier load --
   fine-tuned sentence-transformer, then base sentence-transformer, then a
   **TF-IDF character-ngram fallback that is "always available"** -- so it does
   not fail. It lands on the fallback and reports, truthfully::

       get_model_info() == {"name": "tfidf-fallback", "fine_tuned": False,
                            "fallback": True, "corpus_size": 0, "dimension": 512}

   The model layer therefore *does* disclose the degradation. The bug was here:
   this provider read that dict, **discarded** ``fallback``, set
   ``available = True``, and logged a cheerful "DomainEmbeddingProvider loaded".
   :func:`~src.core.assistant.semantic_retrieval.create_semantic_retriever` then
   adopted it and logged "Using DomainEmbeddingProvider" -- so a disclosed
   fallback was laundered into an undisclosed one, and callers believed they
   were getting fine-tuned manufacturing-domain embeddings.

   :attr:`is_fallback` and :attr:`model_info` now surface it, and construction
   logs a warning when the fine-tuned encoder is absent.

   Note the vectors are *not* zeros: TF-IDF char-ngrams are real, unit-norm, and
   distinct per input -- just far weaker than the advertised domain model, and
   fitted on ``corpus_size == 0``. The zero-vector branch below fires only if
   :class:`DomainEmbeddingModel` raises outright, which the always-available
   fallback prevents in practice.

   Whether :attr:`available` should become ``False`` on fallback (which would
   make ``create_semantic_retriever`` skip this provider entirely) is a
   behavior change and is left to the owner -- see the slice's design MD.
"""

import logging
from typing import List

from .semantic_retrieval import EmbeddingProvider

logger = logging.getLogger(__name__)


class DomainEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by :class:`DomainEmbeddingModel`.

    Despite the name, this may be serving the model's TF-IDF fallback rather
    than a fine-tuned domain encoder -- check :attr:`is_fallback`.

    Parameters
    ----------
    model_path : str
        Path to the fine-tuned model directory.  Defaults to
        ``models/embeddings/manufacturing_v2/``, which currently ships **no
        encoder weights**, so the load lands on the TF-IDF fallback.  See the
        module docstring.
    """

    def __init__(self, model_path: str = "models/embeddings/manufacturing_v2"):
        # Construct speculatively: callers rely on this never raising.
        self._info: dict = {}
        self._is_fallback = False
        try:
            from src.ml.embeddings.model import DomainEmbeddingModel

            self._model = DomainEmbeddingModel(model_path=model_path)
            self._info = dict(self._model.get_model_info())
            self._dim = self._info.get("dimension", 384)
            self._available = True
            # DomainEmbeddingModel discloses `fallback`; do not discard it.
            self._is_fallback = bool(self._info.get("fallback", False))

            if self._is_fallback:
                logger.warning(
                    "DomainEmbeddingProvider is serving a FALLBACK, not the "
                    "fine-tuned domain model (model=%s, dim=%d, fine_tuned=%s, "
                    "corpus_size=%s, model_path=%s). Retrieval quality will be "
                    "materially worse than the 'manufacturing domain embedding' "
                    "name implies. Usual cause: the checkpoint dir has a "
                    "tokenizer/config but no encoder weights "
                    "(*.bin / *.safetensors / *.pth). Inspect "
                    "`provider.is_fallback` / `provider.model_info`.",
                    self._info.get("name", "unknown"),
                    self._dim,
                    self._info.get("fine_tuned"),
                    self._info.get("corpus_size"),
                    model_path,
                )
            else:
                logger.info(
                    "DomainEmbeddingProvider loaded fine-tuned model "
                    "(dim=%d, model=%s)",
                    self._dim,
                    self._info.get("name", "unknown"),
                )
        except Exception:
            logger.warning(
                "DomainEmbeddingProvider is UNAVAILABLE (model_path=%s) and will "
                "return ZERO VECTORS -- every pairwise similarity collapses to a "
                "constant, so any retrieval built on them is meaningless. Check "
                "`provider.available` before using the vectors; "
                "create_semantic_retriever() does this and falls through.",
                model_path,
                exc_info=True,
            )
            self._model = None  # type: ignore[assignment]
            self._available = False
            self._dim = 384

    @property
    def available(self) -> bool:
        """Whether :class:`DomainEmbeddingModel` constructed at all.

        ``False`` means :meth:`embed_text` / :meth:`embed_batch` return zero
        vectors rather than embeddings. Public so callers stop reaching into
        the private ``_available`` attribute.

        .. note::
           ``available is True`` does **not** imply a fine-tuned encoder --
           the model's TF-IDF fallback is "always available". Check
           :attr:`is_fallback`.
        """
        return self._available

    @property
    def is_fallback(self) -> bool:
        """Whether the loaded model is a fallback rather than the fine-tuned encoder.

        ``True`` today, because ``manufacturing_v2/`` ships no encoder weights.
        The underlying model reports this via ``get_model_info()["fallback"]``;
        this provider previously discarded it.
        """
        return self._is_fallback

    @property
    def model_info(self) -> dict:
        """The underlying model's self-report (``name``, ``fine_tuned``, ``fallback``...).

        Empty dict when the model failed to construct.
        """
        return dict(self._info)

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
