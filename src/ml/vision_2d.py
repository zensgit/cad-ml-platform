"""
2D Vision Module (DXF graph classification).

Loads a lightweight GNN model trained on DXF entity graphs.
Supports ensemble voting with multiple models.
"""

from __future__ import annotations

import json
import logging
import os
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

    def _predict_probs(self, data: bytes, file_name: str) -> Dict[str, Any]:
        """Return the full probability vector for internal consumers (e.g. ensembles).

        This keeps the public ``predict_from_bytes`` payload small while still
        enabling proper soft-voting ensembling.
        """
        if not HAS_TORCH or not self._loaded or self.model is None:
            return {"status": "model_unavailable"}
        if not data:
            return {"status": "empty_input"}

        try:
            from src.utils.dxf_io import read_dxf_document_from_bytes

            doc = read_dxf_document_from_bytes(data)
        except Exception as exc:  # noqa: BLE001
            return {"status": "parse_error", "error": str(exc)}

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
            edge_attr = None
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

        return {
            "status": "ok",
            "probs": probs,
            "label_map_size": len(self.label_map),
            "temperature": float(self.temperature),
            "temperature_source": self.temperature_source,
        }

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
        payload = self._predict_probs(data, file_name)
        if payload.get("status") != "ok":
            return {k: v for k, v in payload.items() if k != "probs"}

        probs = payload["probs"]
        topk = min(2, int(probs.numel()))
        top_vals, top_idx = torch.topk(probs, k=topk)
        pred_idx = int(top_idx[0].item())
        top2_conf = float(top_vals[1].item()) if topk > 1 else 0.0
        margin = float(top_vals[0].item() - top_vals[1].item()) if topk > 1 else 1.0
        label = None
        for name, idx in self.label_map.items():
            if idx == pred_idx:
                label = name
                break

        return {
            "label": label,
            "confidence": float(probs[pred_idx].item()),
            "top2_confidence": top2_conf,
            "margin": margin,
            "temperature": payload.get("temperature", float(self.temperature)),
            "temperature_source": payload.get(
                "temperature_source", self.temperature_source
            ),
            "label_map_size": payload.get("label_map_size", len(self.label_map)),
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

        predictions: List[Dict[str, Any]] = []
        prob_vectors: List[List[float]] = []
        master_labels: Optional[List[str]] = None
        label_map_mismatch = False

        def _labels_from_map(label_map: Dict[str, int]) -> List[str]:
            return [
                name for name, _idx in sorted(label_map.items(), key=lambda kv: kv[1])
            ]

        def _top2(probs: List[float]) -> Dict[str, Any]:
            if not probs:
                return {"top1_idx": None, "top1": 0.0, "top2": 0.0}
            pairs = sorted(enumerate(probs), key=lambda p: p[1], reverse=True)
            top1_idx, top1 = pairs[0]
            top2 = pairs[1][1] if len(pairs) > 1 else 0.0
            return {"top1_idx": int(top1_idx), "top1": float(top1), "top2": float(top2)}

        def _align_probs(
            label_map: Dict[str, int], probs: List[float], labels: List[str]
        ) -> Optional[List[float]]:
            if len(label_map) != len(labels):
                return None
            aligned: List[float] = []
            for label in labels:
                idx = label_map.get(label)
                if idx is None or idx >= len(probs):
                    return None
                aligned.append(float(probs[idx]))
            s = float(sum(aligned))
            if s > 0:
                aligned = [p / s for p in aligned]
            return aligned

        for clf in self.classifiers:
            label_map = getattr(clf, "label_map", {}) or {}
            if not isinstance(label_map, dict) or not label_map:
                continue

            probs_payload = (
                clf._predict_probs(data, file_name)  # type: ignore[attr-defined]
                if hasattr(clf, "_predict_probs")
                else None
            )
            probs_obj = None
            if isinstance(probs_payload, dict) and probs_payload.get("status") == "ok":
                probs_obj = probs_payload.get("probs")

            probs: Optional[List[float]] = None
            if probs_obj is not None:
                # Support both torch tensors and plain lists (for tests/stubs).
                try:
                    probs = (
                        [float(v) for v in probs_obj.tolist()]
                        if hasattr(probs_obj, "tolist")
                        else [float(v) for v in probs_obj]
                    )
                except Exception:
                    probs = None

            if probs is None:
                # Fallback: use the public interface if available.
                result = clf.predict_from_bytes(data, file_name)
                if result.get("status") != "ok":
                    continue
                predictions.append(result)
                continue

            if master_labels is None:
                master_labels = _labels_from_map(label_map)
            aligned = _align_probs(label_map, probs, master_labels)
            if aligned is None:
                label_map_mismatch = True
                # Still record per-model top-1 for hard-vote fallback.
                stats = _top2(probs)
                top1_idx = stats["top1_idx"]
                top1_label = None
                if top1_idx is not None:
                    for name, idx in label_map.items():
                        if idx == top1_idx:
                            top1_label = name
                            break
                predictions.append(
                    {
                        "label": top1_label,
                        "confidence": float(stats["top1"]),
                        "status": "ok",
                    }
                )
                continue

            stats = _top2(aligned)
            top1_idx = stats["top1_idx"]
            top1_label = (
                master_labels[top1_idx] if top1_idx is not None else None
            )
            predictions.append(
                {
                    "label": top1_label,
                    "confidence": float(stats["top1"]),
                    "top2_confidence": float(stats["top2"]),
                    "margin": float(stats["top1"] - stats["top2"]),
                    "status": "ok",
                }
            )
            prob_vectors.append(aligned)

        if not predictions:
            return {"status": "all_models_failed"}

        if len(predictions) == 1:
            return predictions[0]

        if self.voting == "soft" and prob_vectors and master_labels and not label_map_mismatch:
            # Proper soft-voting: average aligned probability vectors.
            size = len(master_labels)
            avg = [0.0] * size
            for vec in prob_vectors:
                for i, p in enumerate(vec):
                    avg[i] += float(p)
            denom = float(len(prob_vectors))
            avg = [p / denom for p in avg]

            stats = _top2(avg)
            pred_idx = stats["top1_idx"]
            final_label = master_labels[pred_idx] if pred_idx is not None else None
            final_conf = float(stats["top1"])
            top2_conf = float(stats["top2"])
            margin = float(final_conf - top2_conf)
            voting = "soft"
            label_map_size = len(master_labels)
        else:
            # Hard voting (and fallback when label maps mismatch / no prob vectors).
            from collections import Counter

            votes = Counter(p.get("label") for p in predictions)
            final_label, vote_count = votes.most_common(1)[0]
            # Confidence is proportion of votes
            final_conf = vote_count / len(predictions)
            top2_conf = None
            margin = None
            voting = (
                "hard_fallback_label_map_mismatch"
                if label_map_mismatch and self.voting == "soft"
                else "hard"
            )
            label_map_size = len(master_labels) if master_labels else None

        return {
            "label": final_label,
            "confidence": final_conf,
            "status": "ok",
            "ensemble_size": len(predictions),
            "voting": voting,
            "label_map_mismatch": bool(label_map_mismatch),
            "label_map_size": label_map_size,
            "top2_confidence": top2_conf,
            "margin": margin,
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
