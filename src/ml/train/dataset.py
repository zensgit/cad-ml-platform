"""
ABC Dataset Loader.

Handles loading of STEP files from the ABC Dataset structure for training.
"""

import bisect
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
SURFACE_COUNT_BUCKETS = (10, 20, 40, 80)
DEFAULT_NUM_CLASSES = 10


class ABCDataset(Dataset):
    """
    PyTorch Dataset for ABC CAD models.
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        output_format: str = "numeric",
        graph_backend: str = "auto",
        label_strategy: str = "random",
    ):
        """
        Args:
            root_dir (str): Directory with STEP files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.output_format = output_format
        self.graph_backend = graph_backend
        self.label_strategy = label_strategy
        self.num_classes = self._num_classes_for_strategy(label_strategy)
        self.file_list = []

        try:
            from torch_geometric.data import Data as PygData
        except ImportError:
            PygData = None
        self._pyg_data = PygData

        if os.path.exists(root_dir):
            self.file_list = [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.lower().endswith((".step", ".stp"))
            ]
        else:
            logger.warning(f"ABC Dataset root {root_dir} not found. Using empty set.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        step_path = self.file_list[idx]
        node_dim = 0
        edge_dim = 0

        # Real Feature Extraction
        try:
            # Lazy import to avoid circular dependencies
            from src.core.geometry.engine import get_geometry_engine
            from src.core.geometry.engine import (
                BREP_GRAPH_EDGE_FEATURES,
                BREP_GRAPH_NODE_FEATURES,
            )

            node_dim = len(BREP_GRAPH_NODE_FEATURES)
            edge_dim = len(BREP_GRAPH_EDGE_FEATURES)
            brep_features: Optional[Dict[str, Any]] = None

            # Note: In a high-performance training loop, you might want to pre-process
            # these features offline and save them as .pt or .npy files to avoid
            # re-parsing STEP files (which is slow) every epoch.
            # For this MVP, we parse on-the-fly or assume a cached version exists.

            geo = get_geometry_engine()
            with open(step_path, "rb") as f:
                content = f.read()

            shape = geo.load_step(content, os.path.basename(step_path))
            if self.output_format == "graph":
                sample = self._build_graph_sample(
                    geo,
                    shape,
                    node_dim=len(BREP_GRAPH_NODE_FEATURES),
                    edge_dim=len(BREP_GRAPH_EDGE_FEATURES),
                )
            else:
                sample = self._build_numeric_sample(geo, shape)

            if self.label_strategy != "random":
                brep_features = geo.extract_brep_features(shape) if shape else {}

            label = self._label_from_brep_features(brep_features)

        except Exception as e:
            logger.error(f"Error processing {step_path}: {e}")
            if self.output_format == "graph":
                sample = self._empty_graph_sample(node_dim=node_dim, edge_dim=edge_dim)
            else:
                sample = torch.zeros(12, 1024)
            label = 0

        if self.transform:
            sample = self.transform(sample)

        if (
            self.output_format == "graph"
            and self._pyg_data is not None
            and self.graph_backend in {"pyg", "auto"}
            and not isinstance(sample, dict)
        ):
            sample.y = torch.tensor([label], dtype=torch.long)
            return sample

        return sample, label

    def _num_classes_for_strategy(self, strategy: str) -> int:
        if strategy == "random":
            return DEFAULT_NUM_CLASSES
        if strategy == "surface_bucket":
            return len(SURFACE_COUNT_BUCKETS) + 1
        raise ValueError(f"Unknown label strategy: {strategy}")

    def _label_from_brep_features(self, brep_features: Optional[Dict[str, Any]]) -> int:
        if self.label_strategy == "random":
            return int(torch.randint(0, self.num_classes, (1,)).item())
        if not brep_features:
            return 0
        if self.label_strategy == "surface_bucket":
            faces = int(brep_features.get("faces", 0))
            return int(bisect.bisect_left(SURFACE_COUNT_BUCKETS, faces))
        return 0

    def _build_numeric_sample(self, geo: Any, shape: Any) -> torch.Tensor:
        if not shape:
            return torch.zeros(12, 1024)

        feats = geo.extract_brep_features(shape)
        surfaces = feats.get("surface_types", {})
        vector = [
            float(feats.get("faces", 0)),
            float(feats.get("edges", 0)),
            float(feats.get("vertices", 0)),
            float(feats.get("volume", 0)),
            float(feats.get("surface_area", 0)),
            float(surfaces.get("plane", 0)),
            float(surfaces.get("cylinder", 0)),
            float(surfaces.get("cone", 0)),
            float(surfaces.get("sphere", 0)),
            float(surfaces.get("torus", 0)),
            float(surfaces.get("bspline", 0)),
            float(feats.get("solids", 0)),
        ]

        vector = [torch.tensor(v).float() for v in vector]
        return torch.stack(vector).unsqueeze(1).repeat(1, 1024)

    def _build_graph_sample(
        self,
        geo: Any,
        shape: Any,
        node_dim: int,
        edge_dim: int,
    ) -> Any:
        if not shape:
            return self._empty_graph_sample(node_dim=node_dim, edge_dim=edge_dim)

        graph = geo.extract_brep_graph(shape)
        node_features = graph.get("node_features", [])
        edge_index = graph.get("edge_index", [])
        edge_features = graph.get("edge_features", [])

        x = (
            torch.tensor(node_features, dtype=torch.float32)
            if node_features
            else torch.zeros((0, node_dim), dtype=torch.float32)
        )
        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = (
            torch.tensor(edge_features, dtype=torch.float32)
            if edge_features
            else torch.zeros((0, edge_dim), dtype=torch.float32)
        )

        return self._graph_container(
            x=x,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr,
            graph_schema_version=graph.get("graph_schema_version", "v1"),
            node_schema=graph.get("node_schema"),
            edge_schema=graph.get("edge_schema"),
        )

    def _graph_container(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        graph_schema_version: str,
        node_schema: Optional[Tuple[str, ...]],
        edge_schema: Optional[Tuple[str, ...]],
    ) -> Any:
        use_pyg = self.graph_backend in {"pyg", "auto"} and self._pyg_data is not None
        if self.graph_backend == "pyg" and self._pyg_data is None:
            logger.warning("torch_geometric not available; falling back to dict output.")
        if use_pyg:
            return self._pyg_data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                graph_schema_version=graph_schema_version,
                node_schema=node_schema,
                edge_schema=edge_schema,
            )
        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "graph_schema_version": graph_schema_version,
            "node_schema": node_schema,
            "edge_schema": edge_schema,
        }

    def _empty_graph_sample(self, node_dim: int, edge_dim: int) -> Any:
        return self._graph_container(
            x=torch.zeros((0, node_dim), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, edge_dim), dtype=torch.float32),
            graph_schema_version="v1",
            node_schema=None,
            edge_schema=None,
        )


def get_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    output_format: str = "numeric",
    graph_backend: str = "auto",
    label_strategy: str = "random",
):
    dataset = ABCDataset(
        data_dir,
        output_format=output_format,
        graph_backend=graph_backend,
        label_strategy=label_strategy,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
