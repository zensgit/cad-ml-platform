#!/usr/bin/env python3
"""
Inspect a UV-Net checkpoint and run a minimal forward pass.
"""

import argparse
import os
import sys
from typing import Any, Dict

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment dependent
    print(f"ERROR: torch not available: {exc}")
    sys.exit(1)

sys.path.append(".")

from src.ml.train.model import UVNetGraphModel


def _load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def _build_model(config: Dict[str, Any]) -> UVNetGraphModel:
    return UVNetGraphModel(
        node_input_dim=config.get("node_input_dim", 12),
        hidden_dim=config.get("hidden_dim", 64),
        embedding_dim=config.get("embedding_dim", 1024),
        num_classes=config.get("num_classes", 11),
        dropout_rate=config.get("dropout_rate", 0.3),
    )


def _build_chain_edges(node_count: int) -> torch.Tensor:
    if node_count < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    src = torch.arange(0, node_count - 1, dtype=torch.long)
    dst = torch.arange(1, node_count, dtype=torch.long)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect UV-Net checkpoint config.")
    parser.add_argument(
        "--path",
        default=os.getenv("UVNET_MODEL_PATH", "models/uvnet_v1.pth"),
        help="Path to checkpoint file.",
    )
    parser.add_argument("--nodes", type=int, default=5, help="Number of nodes to test.")
    args = parser.parse_args()

    try:
        checkpoint = _load_checkpoint(args.path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    config = checkpoint.get("config", {})
    if not config:
        print("WARNING: checkpoint config missing; using defaults.")

    model = _build_model(config)
    state_dict = checkpoint.get("model_state_dict")
    if not state_dict:
        print("ERROR: checkpoint missing model_state_dict.")
        return 1

    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        print(f"ERROR: failed to load state_dict: {exc}")
        return 1

    model.eval()

    node_dim = config.get("node_input_dim", model.node_input_dim)
    node_count = max(1, args.nodes)
    x = torch.randn(node_count, node_dim)
    edge_index = _build_chain_edges(node_count)
    batch = torch.zeros(node_count, dtype=torch.long)

    with torch.no_grad():
        logits, embedding = model(x, edge_index, batch)

    print("UV-Net Checkpoint Inspect")
    print(f"Path: {args.path}")
    print(f"Config: {config}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
