#!/usr/bin/env python3
"""
Inspect a UV-Net checkpoint and run a minimal forward pass.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment dependent
    print(f"ERROR: torch not available: {exc}")
    sys.exit(1)

sys.path.append(".")

from src.ml.train.model import (  # noqa: E402
    UVNetGraphModel,
    load_uvnet_state_dict_compatibly,
    resolve_uvnet_grid_branch_surface_kind,
    resolve_uvnet_grid_tower_topology_kind,
)


def _load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def _build_model(config: Dict[str, Any]) -> UVNetGraphModel:
    return UVNetGraphModel(
        node_input_dim=config.get("node_input_dim", 12),
        edge_input_dim=config.get("edge_input_dim", 2),
        hidden_dim=config.get("hidden_dim", 64),
        embedding_dim=config.get("embedding_dim", 1024),
        num_classes=config.get("num_classes", 11),
        dropout_rate=config.get("dropout_rate", 0.3),
        node_schema=config.get("node_schema"),
        edge_schema=config.get("edge_schema"),
        use_edge_attr=bool(config.get("use_edge_attr", True)),
        use_face_grid_features=bool(config.get("use_face_grid_features", False)),
        use_edge_grid_features=bool(config.get("use_edge_grid_features", False)),
        face_grid_channels=int(config.get("face_grid_channels", 8)),
        edge_grid_channels=int(config.get("edge_grid_channels", 13)),
        grid_fusion_mode=str(config.get("grid_fusion_mode", "residual")),
        grid_encoder_kind=str(config.get("grid_encoder_kind", "summary_projection")),
    )


def _build_chain_edges(node_count: int) -> torch.Tensor:
    if node_count < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    src = torch.arange(0, node_count - 1, dtype=torch.long)
    dst = torch.arange(1, node_count, dtype=torch.long)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index


def _build_edge_attr(
    *,
    edge_index: torch.Tensor,
    edge_input_dim: int,
    use_edge_attr: bool,
) -> Optional[torch.Tensor]:
    edge_count = int(edge_index.size(1)) if edge_index.dim() == 2 else 0
    if not use_edge_attr or edge_input_dim <= 0 or edge_count <= 0:
        return None
    return torch.zeros((edge_count, edge_input_dim), dtype=torch.float32)


def _build_summary_payload(
    *,
    path: str,
    config: Dict[str, Any],
    configured_grid_branch_surface_kind: str,
    resolved_grid_branch_surface_kind: str,
    configured_grid_tower_topology_kind: str,
    resolved_grid_tower_topology_kind: str,
    load_result: Dict[str, Any],
    logits: torch.Tensor,
    embedding: torch.Tensor,
) -> Dict[str, Any]:
    return {
        "status": "ok",
        "surface_kind": "uvnet_checkpoint_inspect_summary",
        "path": str(path),
        "config": dict(config),
        "model_surface_contract": {
            "configured_grid_branch_surface_kind": str(configured_grid_branch_surface_kind),
            "resolved_grid_branch_surface_kind": str(resolved_grid_branch_surface_kind),
            "configured_grid_tower_topology_kind": str(configured_grid_tower_topology_kind),
            "resolved_grid_tower_topology_kind": str(resolved_grid_tower_topology_kind),
            "checkpoint_load_mode": str(load_result.get("load_mode") or ""),
            "missing_keys": list(load_result.get("missing_keys") or []),
            "unexpected_keys": list(load_result.get("unexpected_keys") or []),
        },
        "forward_shapes": {
            "logits": [int(dim) for dim in logits.shape],
            "embedding": [int(dim) for dim in embedding.shape],
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect UV-Net checkpoint config.")
    parser.add_argument(
        "--path",
        default=os.getenv("UVNET_MODEL_PATH", "models/uvnet_v1.pth"),
        help="Path to checkpoint file.",
    )
    parser.add_argument("--nodes", type=int, default=5, help="Number of nodes to test.")
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional JSON path for structured checkpoint inspect summary output.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        checkpoint = _load_checkpoint(args.path)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    config = checkpoint.get("config", {})
    if not config:
        print("WARNING: checkpoint config missing; using defaults.")
    configured_grid_branch_surface_kind = str(config.get("grid_branch_surface_kind") or "")
    configured_grid_tower_topology_kind = str(config.get("grid_tower_topology_kind") or "")
    resolved_grid_branch_surface_kind = resolve_uvnet_grid_branch_surface_kind(
        use_face_grid_features=bool(config.get("use_face_grid_features", False)),
        use_edge_grid_features=bool(config.get("use_edge_grid_features", False)),
        grid_encoder_kind=str(config.get("grid_encoder_kind", "summary_projection")),
        grid_fusion_mode=str(config.get("grid_fusion_mode", "residual")),
    )
    resolved_grid_tower_topology_kind = resolve_uvnet_grid_tower_topology_kind(
        use_face_grid_features=bool(config.get("use_face_grid_features", False)),
        use_edge_grid_features=bool(config.get("use_edge_grid_features", False)),
        grid_fusion_mode=str(config.get("grid_fusion_mode", "residual")),
    )

    model = _build_model(config)
    state_dict = checkpoint.get("model_state_dict")
    if not state_dict:
        print("ERROR: checkpoint missing model_state_dict.")
        return 1

    try:
        load_result = load_uvnet_state_dict_compatibly(model, state_dict)
    except Exception as exc:
        print(f"ERROR: failed to load state_dict: {exc}")
        return 1

    model.eval()

    node_dim = config.get("node_input_dim", model.node_input_dim)
    node_count = max(1, args.nodes)
    x = torch.randn(node_count, node_dim)
    edge_index = _build_chain_edges(node_count)
    edge_attr = _build_edge_attr(
        edge_index=edge_index,
        edge_input_dim=int(config.get("edge_input_dim", model.edge_input_dim)),
        use_edge_attr=bool(config.get("use_edge_attr", model.use_edge_attr)),
    )
    batch = torch.zeros(node_count, dtype=torch.long)

    with torch.no_grad():
        logits, embedding = model(x, edge_index, batch, edge_attr=edge_attr)

    print("UV-Net Checkpoint Inspect")
    print(f"Path: {args.path}")
    print(f"Config: {config}")
    print(f"Configured grid branch surface: {configured_grid_branch_surface_kind}")
    print(f"Resolved grid branch surface: {resolved_grid_branch_surface_kind}")
    print(f"Configured grid tower topology: {configured_grid_tower_topology_kind}")
    print(f"Resolved grid tower topology: {resolved_grid_tower_topology_kind}")
    print(f"Checkpoint load mode: {load_result.get('load_mode')}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
    if str(args.summary_json).strip():
        summary_path = Path(str(args.summary_json)).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = _build_summary_payload(
            path=args.path,
            config=config,
            configured_grid_branch_surface_kind=configured_grid_branch_surface_kind,
            resolved_grid_branch_surface_kind=resolved_grid_branch_surface_kind,
            configured_grid_tower_topology_kind=configured_grid_tower_topology_kind,
            resolved_grid_tower_topology_kind=resolved_grid_tower_topology_kind,
            load_result=load_result,
            logits=logits,
            embedding=embedding,
        )
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
