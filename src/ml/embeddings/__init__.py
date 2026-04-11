"""Domain embedding fine-tuning for manufacturing terminology.

Exports
-------
DomainEmbeddingTrainer
    Fine-tune a sentence-transformer on manufacturing contrastive pairs.
DomainEmbeddingModel
    Inference wrapper with encode / similarity / search.
ManufacturingCorpusBuilder
    Generate anchor/positive/negative training triplets from domain knowledge.
"""

from src.ml.embeddings.corpus_builder import ManufacturingCorpusBuilder
from src.ml.embeddings.model import DomainEmbeddingModel
from src.ml.embeddings.trainer import DomainEmbeddingTrainer

__all__ = [
    "DomainEmbeddingTrainer",
    "DomainEmbeddingModel",
    "ManufacturingCorpusBuilder",
]
