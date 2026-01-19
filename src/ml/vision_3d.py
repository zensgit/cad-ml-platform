"""
3D Vision Module (L3 Capability).

Handles Deep Geometric Learning inference.
Designed to interface with UV-Net Graph Models.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import numpy as np
    import torch

    from src.ml.train.model import UVNetGraphModel

    HAS_TORCH = True
except ImportError:
    logger.warning("Torch not found. 3D Vision module running in mock mode.")


class UVNetEncoder:
    """
    Encoder for 3D B-Rep data using Deep Learning (GNN).
    Generates semantic embeddings from geometric shapes.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv("UVNET_MODEL_PATH", "models/uvnet_v1.pth")
        self.model = None
        self.device = "cpu"
        self._loaded = False

        if HAS_TORCH:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            self._load_model()

    def _load_model(self):
        """Load the pre-trained model."""
        if not HAS_TORCH:
            return

        if not os.path.exists(self.model_path):
            logger.info(
                "UV-Net model not found at %s. Will use mock encoder until model is trained.",
                self.model_path,
            )
            return

        try:
            # Load state dict and config
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Initialize model architecture (Config-driven if available)
            config = checkpoint.get("config", {})
            self.model = UVNetGraphModel(
                node_input_dim=config.get("node_input_dim", 12),
                hidden_dim=config.get("hidden_dim", 64),
                embedding_dim=config.get("embedding_dim", 1024),
                num_classes=config.get("num_classes", 11),
                dropout_rate=config.get("dropout_rate", 0.3),
                node_schema=config.get("node_schema"),
                edge_schema=config.get("edge_schema"),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            logger.info(f"UV-Net GNN model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load UV-Net model: {e}")

    def encode(
        self,
        data_source: Union[Dict[str, Any], Any],
    ) -> List[float]:
        """
        Generate an embedding vector for a 3D shape.

        Args:
            data_source: Either:
                1. A dictionary of legacy features (fallback/mock path).
                2. A Graph Data dictionary/object with keys 'x' and 'edge_index'.

        Returns:
            List[float]: A normalized embedding vector (e.g., 1024 dimensions).
        """
        dim = self.model.embedding_dim if self._loaded and self.model is not None else 1024

        # 1. Mock Path (If no model loaded or explicit legacy feature dict)
        if not self._loaded or (isinstance(data_source, dict) and "edge_index" not in data_source):
            return self._mock_embedding(data_source, dim)

        # 2. Inference Path (Graph Data)
        if HAS_TORCH and self._loaded:
            try:
                # Prepare Inputs
                if isinstance(data_source, dict):
                    x = data_source["x"]
                    edge_index = data_source["edge_index"]
                    node_schema = data_source.get("node_schema")
                    edge_schema = data_source.get("edge_schema")
                else:
                    # Assume PyG Data object
                    x = data_source.x
                    edge_index = data_source.edge_index
                    node_schema = getattr(data_source, "node_schema", None)
                    edge_schema = getattr(data_source, "edge_schema", None)

                # Ensure tensors and device
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float)
                if not isinstance(edge_index, torch.Tensor):
                    edge_index = torch.tensor(edge_index, dtype=torch.long)

                x = x.to(self.device)
                edge_index = edge_index.to(self.device)

                model_node_schema = getattr(self.model, "node_schema", None)
                model_edge_schema = getattr(self.model, "edge_schema", None)
                if model_node_schema is not None and node_schema is not None:
                    if tuple(model_node_schema) != tuple(node_schema):
                        logger.error(
                            "Graph node schema mismatch: expected %s, got %s",
                            model_node_schema,
                            node_schema,
                        )
                        return [0.0] * dim
                elif model_node_schema is not None and node_schema is None:
                    logger.warning("Graph node schema missing from input data.")
                elif model_node_schema is None and node_schema is not None:
                    logger.warning("Model node schema missing; input schema provided.")

                if model_edge_schema is not None and edge_schema is not None:
                    if tuple(model_edge_schema) != tuple(edge_schema):
                        logger.error(
                            "Graph edge schema mismatch: expected %s, got %s",
                            model_edge_schema,
                            edge_schema,
                        )
                        return [0.0] * dim
                elif model_edge_schema is not None and edge_schema is None:
                    logger.warning("Graph edge schema missing from input data.")
                elif model_edge_schema is None and edge_schema is not None:
                    logger.warning("Model edge schema missing; input schema provided.")

                if x.dim() != 2:
                    logger.error("Graph node feature tensor must be 2D, got shape %s", tuple(x.shape))
                    return [0.0] * dim

                expected_dim = getattr(self.model, "node_input_dim", None)
                if expected_dim is not None and x.size(1) != expected_dim:
                    logger.error(
                        "Graph node feature dim mismatch: expected %s, got %s",
                        expected_dim,
                        x.size(1),
                    )
                    return [0.0] * dim

                if x.size(0) == 0:
                    logger.error("Graph node feature tensor is empty; returning zeros.")
                    return [0.0] * dim

                # Create Batch Index (All 0s for single sample)
                batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

                with torch.no_grad():
                    _, embedding = self.model(x, edge_index, batch)

                return embedding.cpu().numpy().flatten().tolist()

            except Exception as e:
                logger.error(f"Inference failed: {e}")
                # Fallback to zeros on error
                return [0.0] * dim

        return [0.0] * dim

    def _mock_embedding(self, features: Dict[str, Any], dim: int) -> List[float]:
        """
        Generate a heuristic embedding based on available scalar features.
        Used when the deep learning model is not available.
        """
        seed_val = (
            features.get("faces", 0) * 1000
            + features.get("edges", 0)
            + int(features.get("volume", 0) * 100)
        )
        vec = [0.0] * dim

        # Use legacy scalar mapping for first few dimensions if available
        if "surface_types" in features:
            surfaces = features.get("surface_types", {})
            vec[0] = surfaces.get("plane", 0) / 100.0
            vec[1] = surfaces.get("cylinder", 0) / 50.0

        # Fill rest with pseudo-random noise
        import random

        random.seed(seed_val)
        for i in range(10, dim):
            vec[i] = random.random() * 0.1

        # Normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]

        return vec


# Singleton
_encoder = UVNetEncoder()


def get_3d_encoder():
    return _encoder
