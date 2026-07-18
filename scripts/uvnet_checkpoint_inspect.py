#!/usr/bin/env python3
"""
Inspect a UV-Net checkpoint and run a minimal forward pass.

Contract: this tool reports ONLY what the current ``UVNetGraphModel`` actually
is — the real constructor config, the result of a STRICT ``load_state_dict``,
and the observed forward shapes. It must not claim model capabilities that do
not exist in ``src/ml/train/model.py`` (see PR #523 diagnosis: commit 33cb0f65
added imports of never-implemented grid APIs, false-green wherever torch was
absent). A checkpoint that does not strictly match the current architecture is
reported as an ERROR with a non-zero exit — that mismatch is the finding, not
something to paper over with a compat shim.
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

from src.ml.train.model import UVNetGraphModel  # noqa: E402


def _load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def _build_model(config: Dict[str, Any]) -> UVNetGraphModel:
    # Exactly the current UVNetGraphModel constructor surface — nothing else.
    return UVNetGraphModel(
        node_input_dim=config.get("node_input_dim"),
        edge_input_dim=config.get("edge_input_dim"),
        hidden_dim=config.get("hidden_dim", 64),
        embedding_dim=config.get("embedding_dim", 1024),
        num_classes=config.get("num_classes", 11),
        dropout_rate=config.get("dropout_rate", 0.3),
        node_schema=config.get("node_schema"),
        edge_schema=config.get("edge_schema"),
        use_edge_attr=bool(config.get("use_edge_attr", True)),
    )


def _build_chain_edges(node_count: int) -> torch.Tensor:
    if node_count < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    src = torch.arange(0, node_count - 1, dtype=torch.long)
    dst = torch.arange(1, node_count, dtype=torch.long)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


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
    logits: torch.Tensor,
    embedding: torch.Tensor,
) -> Dict[str, Any]:
    # Only real, observed facts: config as stored, strict-load outcome (writing
    # this payload at all means strict load succeeded), forward shapes.
    return {
        "status": "ok",
        "path": str(path),
        "config": {key: config[key] for key in sorted(config)},
        "strict_load": {"mode": "strict", "ok": True},
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

    model = _build_model(config)
    state_dict = checkpoint.get("model_state_dict")
    if not state_dict:
        print("ERROR: checkpoint missing model_state_dict.")
        return 1

    try:
        model.load_state_dict(state_dict)  # strict by default — mismatches are findings
    except Exception as exc:
        print(f"ERROR: failed to load state_dict strictly: {exc}")
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
    print("Strict load: ok")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Embedding shape: {tuple(embedding.shape)}")
    if str(args.summary_json).strip():
        summary_path = Path(str(args.summary_json)).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                _build_summary_payload(
                    path=args.path, config=config, logits=logits, embedding=embedding
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
