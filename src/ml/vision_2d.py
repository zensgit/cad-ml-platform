"""
2D Vision Module (DXF graph classification).

Loads a lightweight GNN model trained on DXF entity graphs.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

HAS_TORCH = False
try:
    import torch

    from src.ml.train.dataset_2d import DXFDataset
    from src.ml.train.model_2d import SimpleGraphClassifier

    HAS_TORCH = True
except Exception:
    logger.warning("Torch not found. 2D vision module disabled.")


class Graph2DClassifier:
    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path or os.getenv(
            "GRAPH2D_MODEL_PATH", "models/graph2d_parts_upsampled_20260122.pth"
        )
        self.model: Optional[SimpleGraphClassifier] = None
        self.label_map: Dict[str, int] = {}
        self.node_dim: int = 7
        self.device = "cpu"
        self._loaded = False

        if HAS_TORCH and os.path.exists(self.model_path):
            self._load_model()

    def _load_model(self) -> None:
        if not HAS_TORCH:
            return
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.label_map = checkpoint.get("label_map", {})
        node_dim = int(checkpoint.get("node_dim", 7))
        self.node_dim = node_dim
        hidden_dim = int(checkpoint.get("hidden_dim", 64))
        num_classes = max(1, len(self.label_map))
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
        dataset = DXFDataset(root_dir=".", node_dim=self.node_dim)
        x, edge_index = dataset._dxf_to_graph(msp, self.node_dim)
        if x.numel() == 0:
            return {"status": "empty_graph"}

        with torch.no_grad():
            logits = self.model(x, edge_index)
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
                "status": "ok",
            }


_graph2d = Graph2DClassifier()


def get_2d_classifier() -> Graph2DClassifier:
    return _graph2d
