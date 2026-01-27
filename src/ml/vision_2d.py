"""
2D Vision Module (DXF graph classification).

Loads a lightweight GNN model trained on DXF entity graphs.
Supports ensemble voting with multiple models.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch

    from src.ml.train.dataset_2d import DXFDataset, DXF_EDGE_DIM, DXF_NODE_DIM
    from src.ml.train.model_2d import EdgeGraphSageClassifier, SimpleGraphClassifier

    HAS_TORCH = True
except Exception:
    logger.warning("Torch not found. 2D vision module disabled.")
    DXF_NODE_DIM = 0
    DXF_EDGE_DIM = 0
    DXFDataset = None
    EdgeGraphSageClassifier = None
    SimpleGraphClassifier = None


class Graph2DClassifier:
    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or os.getenv(
            "GRAPH2D_MODEL_PATH", "models/graph2d_parts_upsampled_20260122.pth"
        )
        self.model: Optional[torch.nn.Module] = None
        self.label_map: Dict[str, int] = {}
        self.node_dim: int = DXF_NODE_DIM
        self.edge_dim: int = DXF_EDGE_DIM
        self.model_type: str = "gcn"
        self.device = "cpu"
        self.temperature: float = 1.0
        self.temperature_source: Optional[str] = None
        self._loaded = False

        self._load_temperature()

        if HAS_TORCH and os.path.exists(self.model_path):
            self._load_model()

    def _load_temperature(self) -> None:
        temp_raw = os.getenv("GRAPH2D_TEMPERATURE")
        if temp_raw:
            try:
                temp = float(temp_raw)
                if temp > 0:
                    self.temperature = temp
                    self.temperature_source = "env"
                else:
                    logger.warning("GRAPH2D_TEMPERATURE must be > 0; got %s", temp_raw)
            except ValueError:
                logger.warning("Invalid GRAPH2D_TEMPERATURE=%s", temp_raw)
            return

        calibration_path = os.getenv("GRAPH2D_TEMPERATURE_CALIBRATION_PATH")
        if not calibration_path:
            return
        path = Path(calibration_path)
        if not path.exists():
            logger.warning("Graph2D calibration file not found: %s", path)
            return
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse Graph2D calibration file: %s", exc)
            return
        temp = payload.get("temperature")
        try:
            temp_val = float(temp)
        except (TypeError, ValueError):
            logger.warning("Invalid temperature in calibration file: %s", temp)
            return
        if temp_val <= 0:
            logger.warning("Calibration temperature must be > 0; got %s", temp_val)
            return
        self.temperature = temp_val
        self.temperature_source = str(path)

    def _load_model(self) -> None:
        if not HAS_TORCH:
            return
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.label_map = checkpoint.get("label_map", {})
        node_dim = int(checkpoint.get("node_dim", DXF_NODE_DIM))
        self.node_dim = node_dim
        hidden_dim = int(checkpoint.get("hidden_dim", 64))
        self.model_type = checkpoint.get("model_type", "gcn")
        self.edge_dim = int(checkpoint.get("edge_dim", DXF_EDGE_DIM))
        num_classes = max(1, len(self.label_map))
        if self.model_type == "edge_sage":
            self.model = EdgeGraphSageClassifier(
                node_dim, self.edge_dim, hidden_dim, num_classes
            )
        else:
            self.model = SimpleGraphClassifier(node_dim, hidden_dim, num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self._loaded = True

    def predict_from_bytes(self, data: bytes, file_name: str) -> Dict[str, Any]:
        if not HAS_TORCH or not self._loaded or self.model is None:
            return {"status": "model_unavailable"}
        if not data:
            return {"status": "empty_input"}

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            import ezdxf  # type: ignore

            doc = ezdxf.readfile(tmp_path)
        except Exception as exc:
            return {"status": "parse_error", "error": str(exc)}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        msp = doc.modelspace()
        dataset = DXFDataset(
            root_dir=".",
            node_dim=self.node_dim,
            return_edge_attr=self.model_type == "edge_sage",
        )
        if self.model_type == "edge_sage":
            x, edge_index, edge_attr = dataset._dxf_to_graph(
                msp, self.node_dim, return_edge_attr=True
            )
        else:
            x, edge_index = dataset._dxf_to_graph(msp, self.node_dim)
        if x.numel() == 0:
            return {"status": "empty_graph"}

        with torch.no_grad():
            if self.model_type == "edge_sage":
                logits = self.model(x, edge_index, edge_attr)
            else:
                logits = self.model(x, edge_index)
            if self.temperature != 1.0:
                logits = logits / self.temperature
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            label = None
            for name, idx in self.label_map.items():
                if idx == pred_idx:
                    label = name
                    break
            return {
                "label": label,
                "confidence": float(probs[pred_idx].item()),
                "temperature": float(self.temperature),
                "temperature_source": self.temperature_source,
                "status": "ok",
            }


_graph2d = Graph2DClassifier()


def get_2d_classifier() -> Graph2DClassifier:
    return _graph2d


class EnsembleGraph2DClassifier:
    """Ensemble classifier that combines predictions from multiple Graph2D models."""

    def __init__(
        self,
        model_paths: Optional[List[str]] = None,
        voting: str = "soft",
    ) -> None:
        """
        Initialize ensemble classifier.

        Args:
            model_paths: List of model checkpoint paths. If None, uses env var
                GRAPH2D_ENSEMBLE_MODELS (comma-separated) or defaults to v3+v4.
            voting: Voting strategy - "soft" (average probabilities) or "hard" (majority vote).
        """
        if model_paths is None:
            env_paths = os.getenv("GRAPH2D_ENSEMBLE_MODELS", "")
            if env_paths:
                model_paths = [p.strip() for p in env_paths.split(",") if p.strip()]
            else:
                model_paths = [
                    "models/graph2d_edge_sage_v3.pth",
                    "models/graph2d_edge_sage_v4_best.pth",
                ]

        self.model_paths = model_paths
        self.voting = voting
        self.classifiers: List[Graph2DClassifier] = []
        self._loaded = False

        if HAS_TORCH:
            self._load_models()

    def _load_models(self) -> None:
        for path in self.model_paths:
            if os.path.exists(path):
                clf = Graph2DClassifier(model_path=path)
                if clf._loaded:
                    self.classifiers.append(clf)
                    logger.info("Loaded ensemble model: %s", path)
                else:
                    logger.warning("Failed to load ensemble model: %s", path)
            else:
                logger.warning("Ensemble model not found: %s", path)

        self._loaded = len(self.classifiers) > 0
        if self._loaded:
            logger.info("Ensemble initialized with %d models", len(self.classifiers))

    def predict_from_bytes(self, data: bytes, file_name: str) -> Dict[str, Any]:
        """Ensemble prediction combining multiple models."""
        if not HAS_TORCH or not self._loaded:
            return {"status": "model_unavailable"}
        if not data:
            return {"status": "empty_input"}

        predictions = []
        all_probs = []

        for clf in self.classifiers:
            result = clf.predict_from_bytes(data, file_name)
            if result.get("status") == "ok":
                predictions.append(result)

        if not predictions:
            return {"status": "all_models_failed"}

        if len(predictions) == 1:
            return predictions[0]

        # Get label map from first classifier
        label_map = self.classifiers[0].label_map
        idx_to_label = {v: k for k, v in label_map.items()}
        num_classes = len(label_map)

        if self.voting == "soft":
            # Average probabilities across models
            import torch

            avg_probs = torch.zeros(num_classes)
            for pred in predictions:
                # Reconstruct probability vector
                pred_label = pred.get("label")
                pred_conf = pred.get("confidence", 0.0)
                if pred_label in label_map:
                    idx = label_map[pred_label]
                    # Simple approximation: assign confidence to predicted class
                    # and distribute remaining across others
                    probs = torch.ones(num_classes) * (1 - pred_conf) / (num_classes - 1)
                    probs[idx] = pred_conf
                    avg_probs += probs

            avg_probs /= len(predictions)
            pred_idx = int(torch.argmax(avg_probs).item())
            final_label = idx_to_label.get(pred_idx)
            final_conf = float(avg_probs[pred_idx].item())

        else:  # hard voting
            from collections import Counter

            votes = Counter(p.get("label") for p in predictions)
            final_label, vote_count = votes.most_common(1)[0]
            # Confidence is proportion of votes
            final_conf = vote_count / len(predictions)

        return {
            "label": final_label,
            "confidence": final_conf,
            "status": "ok",
            "ensemble_size": len(predictions),
            "voting": self.voting,
            "individual_predictions": [
                {"label": p.get("label"), "confidence": p.get("confidence")}
                for p in predictions
            ],
        }


_ensemble_graph2d: Optional[EnsembleGraph2DClassifier] = None


def get_ensemble_2d_classifier() -> EnsembleGraph2DClassifier:
    """Get or create the ensemble 2D classifier singleton."""
    global _ensemble_graph2d
    if _ensemble_graph2d is None:
        _ensemble_graph2d = EnsembleGraph2DClassifier()
    return _ensemble_graph2d
