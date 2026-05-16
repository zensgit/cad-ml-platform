"""
PointNet 3D Analyzer -- High-Level Inference API.

Provides a simple interface for classifying 3D files, extracting feature
vectors, and finding similar parts. When no trained model is available
the analyser returns deterministic fallback results so downstream code
(including the REST API) can function without a checkpoint.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.ml.pointnet.preprocessor import PointCloudPreprocessor

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    pass

# Default class labels for CAD part classification
DEFAULT_LABELS = [
    "bracket",
    "gear",
    "housing",
    "shaft",
    "plate",
    "fitting",
    "flange",
    "connector",
]


class PointNet3DAnalyzer:
    """High-level API for 3D point cloud analysis using PointNet.

    If a trained model checkpoint is provided it will be loaded and used
    for inference. Otherwise all prediction methods return structured
    fallback results with ``status: "model_unavailable"`` so that the
    API layer can remain functional.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 8,
        feature_dim: int = 256,
        num_points: int = 2048,
        labels: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_points = num_points
        self.labels = labels or DEFAULT_LABELS[: num_classes]

        self.preprocessor = PointCloudPreprocessor(
            num_points=num_points, normalize_default=True
        )

        self._classifier = None
        self._feature_extractor = None
        self._device = "cpu"
        self._model_loaded = False
        # Set when the checkpoint exists but load raised, so the readiness
        # registry can distinguish load-failure from cold/missing state.
        self._load_error: Optional[str] = None

        self._try_load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """Attempt to load a trained checkpoint. Fail silently."""
        if not HAS_TORCH:
            logger.info("PyTorch unavailable -- running in fallback mode.")
            return

        if self.model_path is None or not os.path.exists(self.model_path):
            logger.info(
                "No model checkpoint at %s -- running in fallback mode.",
                self.model_path,
            )
            return

        try:
            from src.ml.pointnet.model import (
                PointNetClassifier,
                PointNetFeatureExtractor,
            )

            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            self._device = device

            checkpoint = torch.load(
                self.model_path, map_location=device, weights_only=False
            )

            self._classifier = PointNetClassifier(
                num_classes=self.num_classes
            )
            if "classifier_state_dict" in checkpoint:
                self._classifier.load_state_dict(checkpoint["classifier_state_dict"])
            elif "state_dict" in checkpoint:
                self._classifier.load_state_dict(checkpoint["state_dict"])
            else:
                self._classifier.load_state_dict(checkpoint)
            self._classifier.to(device)
            self._classifier.eval()

            self._feature_extractor = PointNetFeatureExtractor(
                feature_dim=self.feature_dim
            )
            if "extractor_state_dict" in checkpoint:
                self._feature_extractor.load_state_dict(
                    checkpoint["extractor_state_dict"]
                )
            self._feature_extractor.to(device)
            self._feature_extractor.eval()

            self._model_loaded = True
            logger.info("PointNet model loaded from %s on %s.", self.model_path, device)
        except Exception as exc:
            logger.exception("Failed to load PointNet model -- using fallback mode.")
            self._classifier = None
            self._feature_extractor = None
            self._model_loaded = False
            self._load_error = str(exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, file_path: str) -> Dict[str, Any]:
        """Classify a 3D file.

        Args:
            file_path: Path to STL / OBJ / PLY / XYZ file.

        Returns:
            Dict with keys:
            - label (str): predicted class label
            - confidence (float): confidence score [0, 1]
            - probabilities (dict[str, float]): per-class probabilities
            - status (str): "ok" or "model_unavailable"
        """
        points = self.preprocessor.load(file_path, num_points=self.num_points)

        if not self._model_loaded or self._classifier is None:
            return self._fallback_classification()

        tensor = self.preprocessor.to_tensor(points).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits, _, _ = self._classifier(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        idx = int(np.argmax(probs))
        label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
        return {
            "label": label,
            "confidence": float(probs[idx]),
            "probabilities": {
                self.labels[i] if i < len(self.labels) else f"class_{i}": float(p)
                for i, p in enumerate(probs)
            },
            "status": "ok",
        }

    def extract_features(self, file_path: str) -> Dict[str, Any]:
        """Extract a feature vector from a 3D file.

        Args:
            file_path: Path to STL / OBJ / PLY / XYZ file.

        Returns:
            Dict with keys:
            - vector (list[float]): feature vector
            - dimension (int): vector dimensionality
            - status (str): "ok" or "model_unavailable"
        """
        points = self.preprocessor.load(file_path, num_points=self.num_points)

        if not self._model_loaded or self._feature_extractor is None:
            return self._fallback_features(points)

        tensor = self.preprocessor.to_tensor(points).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._feature_extractor(tensor).squeeze(0).cpu().numpy()

        return {
            "vector": features.tolist(),
            "dimension": len(features),
            "status": "ok",
        }

    def find_similar(
        self, file_path: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """Find similar parts by feature vector comparison.

        Extracts a feature vector and searches a vector store. When no
        vector store is configured, returns a placeholder result.

        Args:
            file_path: Path to STL / OBJ / PLY / XYZ file.
            top_k: Number of similar parts to return.

        Returns:
            Dict with keys:
            - query_vector (list[float]): the query feature vector
            - results (list[dict]): list of similar parts (may be empty)
            - status (str): "ok" or "model_unavailable"
        """
        feat_result = self.extract_features(file_path)
        query_vector = feat_result["vector"]

        # Placeholder -- integration with the vector store is handled at
        # the API layer; here we just return the extracted features.
        return {
            "query_vector": query_vector,
            "dimension": feat_result["dimension"],
            "top_k": top_k,
            "results": [],
            "status": feat_result["status"],
        }

    @staticmethod
    def supported_formats() -> List[str]:
        """Return list of supported 3D file extensions."""
        return [".stl", ".obj", ".ply", ".xyz"]

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------

    def _fallback_classification(self) -> Dict[str, Any]:
        """Return a deterministic fallback classification result."""
        uniform = 1.0 / self.num_classes
        return {
            "label": "unknown",
            "confidence": uniform,
            "probabilities": {
                self.labels[i] if i < len(self.labels) else f"class_{i}": uniform
                for i in range(self.num_classes)
            },
            "status": "model_unavailable",
        }

    def _fallback_features(self, points: np.ndarray) -> Dict[str, Any]:
        """Return a simple statistical feature vector as a fallback.

        Computes basic statistics (mean, std, min, max per axis plus
        bounding box dimensions) and zero-pads to the requested
        feature_dim.
        """
        stats = []
        if points is not None and len(points) > 0:
            stats.extend(points.mean(axis=0).tolist())  # 3 values
            stats.extend(points.std(axis=0).tolist())  # 3 values
            stats.extend(points.min(axis=0).tolist())  # 3 values
            stats.extend(points.max(axis=0).tolist())  # 3 values
            bbox = points.max(axis=0) - points.min(axis=0)
            stats.extend(bbox.tolist())  # 3 values  (15 total)

        # Pad or truncate to feature_dim
        if len(stats) < self.feature_dim:
            stats.extend([0.0] * (self.feature_dim - len(stats)))
        else:
            stats = stats[: self.feature_dim]

        return {
            "vector": stats,
            "dimension": self.feature_dim,
            "status": "model_unavailable",
        }
