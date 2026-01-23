#!/usr/bin/env python3
"""
Dry-run UV-Net graph data loading and forward pass.
"""

import argparse
import os
import sys

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment dependent
    print(f"ERROR: torch not available: {exc}")
    sys.exit(1)

sys.path.append(".")

from src.core.geometry.engine import HAS_OCC, BREP_GRAPH_EDGE_FEATURES, BREP_GRAPH_NODE_FEATURES
from src.ml.train.dataset import ABCDataset
from src.ml.train.model import UVNetGraphModel
from src.ml.train.trainer import get_graph_dataloader


def main() -> int:
    parser = argparse.ArgumentParser(description="UV-Net graph dry-run.")
    parser.add_argument("--data-dir", default="data/abc_subset", help="STEP data directory.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for dry-run.")
    parser.add_argument("--limit", type=int, default=1, help="Limit number of files to load.")
    parser.add_argument(
        "--graph-backend",
        default="auto",
        choices=("auto", "pyg", "dict"),
        help="Graph backend selection.",
    )
    args = parser.parse_args()

    if not HAS_OCC:
        print("pythonocc-core not available; dry-run skipped.")
        return 0

    if not os.path.exists(args.data_dir):
        print(f"Data directory not found: {args.data_dir}")
        return 1

    dataset = ABCDataset(
        args.data_dir,
        output_format="graph",
        graph_backend=args.graph_backend,
    )

    if args.limit > 0:
        dataset.file_list = dataset.file_list[: args.limit]

    if len(dataset) == 0:
        print("No STEP files found for dry-run.")
        return 1

    dataloader = get_graph_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
    batch_data = next(iter(dataloader))

    if isinstance(batch_data, tuple):
        inputs, _targets = batch_data
        x = inputs["x"]
        edge_index = inputs["edge_index"]
        edge_attr = inputs.get("edge_attr")
        batch_idx = inputs["batch"]
    else:
        x = batch_data.x
        edge_index = batch_data.edge_index
        edge_attr = getattr(batch_data, "edge_attr", None)
        batch_idx = batch_data.batch

    node_count = x.size(0)
    edge_count = edge_index.size(1)
    if node_count == 0 or edge_count == 0:
        print(
            f"Dry-run produced empty graph (nodes={node_count}, edges={edge_count}); "
            "verify input STEP files."
        )
        return 1

    model = UVNetGraphModel(
        node_input_dim=len(BREP_GRAPH_NODE_FEATURES),
        num_classes=10,
        node_schema=BREP_GRAPH_NODE_FEATURES,
        edge_schema=BREP_GRAPH_EDGE_FEATURES,
    )
    model.eval()

    with torch.no_grad():
        logits, embedding = model(x, edge_index, batch_idx, edge_attr=edge_attr)

    print("UV-Net Graph Dry-Run")
    print(f"Data dir: {args.data_dir}")
    print(f"Batch nodes: {node_count}")
    print(f"Batch edges: {edge_count}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
