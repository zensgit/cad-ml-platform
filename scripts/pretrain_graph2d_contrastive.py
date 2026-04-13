#!/usr/bin/env python3
"""Graph2D contrastive pretraining (GraphCL / SimCLR style).

Learns useful graph representations from UNLABELED DXF files by training
a GNN encoder to produce similar embeddings for two augmented views of the
same graph.

Usage:
    python scripts/pretrain_graph2d_contrastive.py --dxf-dir /path/to/dxf
    python scripts/pretrain_graph2d_contrastive.py --dry-run  # synthetic data
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ml.graph_augmentations import random_augmentation  # noqa: E402
from src.ml.train.model_2d import (  # noqa: E402
    GraphEncoder,
    ProjectionHead,
    nt_xent_loss,
)
from src.ml.train.dataset_2d import DXF_EDGE_DIM, DXF_NODE_DIM  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Unlabeled DXF dataset
# --------------------------------------------------------------------------- #

class UnlabeledDXFGraphDataset(Dataset):
    """Load DXF files as graphs without requiring labels.

    When ``dry_run=True``, generates synthetic random graphs instead of
    reading real DXF files.
    """

    def __init__(
        self,
        dxf_dir: Optional[str] = None,
        node_dim: int = DXF_NODE_DIM,
        edge_dim: int = DXF_EDGE_DIM,
        max_nodes: int = 200,
        dry_run: bool = False,
        dry_run_size: int = 64,
    ) -> None:
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.max_nodes = max_nodes
        self.dry_run = dry_run
        self.dry_run_size = dry_run_size
        self.dxf_files: List[Path] = []

        if dry_run:
            return

        if dxf_dir is None:
            raise ValueError("dxf_dir is required when not in dry-run mode")

        dxf_path = Path(dxf_dir)
        for ext in ("*.dxf", "*.DXF"):
            self.dxf_files.extend(dxf_path.rglob(ext))
        self.dxf_files.sort()
        if not self.dxf_files:
            logger.warning("No DXF files found in %s", dxf_dir)

    def __len__(self) -> int:
        if self.dry_run:
            return self.dry_run_size
        return len(self.dxf_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.dry_run:
            return self._synthetic_graph()
        return self._load_dxf(self.dxf_files[idx])

    def _synthetic_graph(self) -> Dict[str, torch.Tensor]:
        """Generate a random graph for testing."""
        n = torch.randint(5, min(30, self.max_nodes), (1,)).item()
        x = torch.randn(n, self.node_dim)
        # Random edges (about 2*n)
        num_edges = min(2 * n, n * (n - 1))
        if num_edges > 0:
            src = torch.randint(0, n, (num_edges,))
            dst = torch.randint(0, n, (num_edges,))
            # Remove self-loops
            mask = src != dst
            src, dst = src[mask], dst[mask]
            edge_index = torch.stack([src, dst], dim=0)
            edge_attr = torch.randn(edge_index.size(1), self.edge_dim)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, self.edge_dim)
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}

    def _load_dxf(self, path: Path) -> Dict[str, torch.Tensor]:
        """Load a DXF file and convert to graph tensors."""
        try:
            import ezdxf
            from src.ml.train.dataset_2d import DXFDataset

            doc = ezdxf.readfile(str(path))
            msp = doc.modelspace()
            ds = DXFDataset(root_dir=".", node_dim=self.node_dim, return_edge_attr=True)
            x, edge_index, edge_attr = ds._dxf_to_graph(
                msp, self.node_dim, return_edge_attr=True
            )
            return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
        except Exception as e:
            logger.warning("Failed to load %s: %s — using empty graph", path, e)
            return {
                "x": torch.zeros(0, self.node_dim),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "edge_attr": torch.zeros(0, self.edge_dim),
            }


def collate_graphs(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate variable-size graphs into a single batched graph.

    Returns a dict with ``x``, ``edge_index``, ``edge_attr``, and ``batch``
    (node-to-graph assignment vector).
    """
    xs, edge_indices, edge_attrs, batch_ids = [], [], [], []
    offset = 0
    for i, g in enumerate(batch):
        n = g["x"].size(0)
        xs.append(g["x"])
        if g["edge_index"].numel() > 0:
            edge_indices.append(g["edge_index"] + offset)
        else:
            edge_indices.append(g["edge_index"])
        if "edge_attr" in g and g["edge_attr"] is not None:
            edge_attrs.append(g["edge_attr"])
        batch_ids.append(torch.full((n,), i, dtype=torch.long))
        offset += n

    return {
        "x": torch.cat(xs, dim=0) if xs else torch.zeros(0, batch[0]["x"].size(1) if batch else 1),
        "edge_index": torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros(2, 0, dtype=torch.long),
        "edge_attr": torch.cat(edge_attrs, dim=0) if edge_attrs else None,
        "batch": torch.cat(batch_ids, dim=0) if batch_ids else torch.zeros(0, dtype=torch.long),
    }


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def pretrain(
    encoder: GraphEncoder,
    proj_head: ProjectionHead,
    dataloader: DataLoader,
    *,
    epochs: int = 100,
    lr: float = 0.001,
    temperature: float = 0.5,
    device: str = "cpu",
    output_path: str = "models/graph2d_pretrained_contrastive.pth",
    node_dim: int = DXF_NODE_DIM,
    edge_dim: int = DXF_EDGE_DIM,
    hidden_dim: int = 64,
) -> Dict:
    """Run contrastive pretraining loop."""
    encoder = encoder.to(device)
    proj_head = proj_head.to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()), lr=lr
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        encoder.train()
        proj_head.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_graphs in dataloader:
            # Each graph in the batch gets two augmented views
            graphs_list = _unbatch(batch_graphs)
            if len(graphs_list) < 2:
                continue

            # Create two augmented views per graph
            view1_list = [random_augmentation(g) for g in graphs_list]
            view2_list = [random_augmentation(g) for g in graphs_list]

            # Batch augmented views
            batch1 = collate_graphs(view1_list)
            batch2 = collate_graphs(view2_list)

            # Move to device
            batch1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch1.items()}
            batch2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch2.items()}

            # Forward pass through encoder
            h1 = encoder(
                batch1["x"], batch1["edge_index"],
                edge_attr=batch1.get("edge_attr"),
                batch=batch1["batch"],
            )
            h2 = encoder(
                batch2["x"], batch2["edge_index"],
                edge_attr=batch2.get("edge_attr"),
                batch=batch2["batch"],
            )

            # Project
            z1 = proj_head(h1)
            z2 = proj_head(h2)

            # Contrastive loss
            loss = nt_xent_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]})

        if avg_loss < best_loss:
            best_loss = avg_loss

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4d} | loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    checkpoint = {
        "encoder_state_dict": encoder.state_dict(),
        "proj_head_state_dict": proj_head.state_dict(),
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "hidden_dim": hidden_dim,
        "model_type": encoder.model_type,
        "best_loss": best_loss,
        "epochs": epochs,
        "temperature": temperature,
    }
    torch.save(checkpoint, output_path)
    print(f"Saved pretrained checkpoint to {output_path}")

    return {"best_loss": best_loss, "history": history, "checkpoint": output_path}


def _unbatch(batched: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    """Split a batched graph back into individual graphs."""
    batch_ids = batched["batch"]
    if batch_ids.numel() == 0:
        return []

    num_graphs = int(batch_ids.max().item()) + 1
    graphs = []

    for i in range(num_graphs):
        node_mask = batch_ids == i
        node_indices = torch.where(node_mask)[0]
        if node_indices.numel() == 0:
            continue

        x = batched["x"][node_mask]
        offset = int(node_indices[0].item())

        # Filter edges belonging to this subgraph
        edge_index = batched["edge_index"]
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            edge_mask = node_mask[src] & node_mask[dst]
            sub_edge_index = edge_index[:, edge_mask] - offset
        else:
            sub_edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_mask = torch.zeros(0, dtype=torch.bool)

        g: Dict[str, torch.Tensor] = {
            "x": x,
            "edge_index": sub_edge_index,
        }

        if batched.get("edge_attr") is not None and batched["edge_attr"].numel() > 0:
            g["edge_attr"] = batched["edge_attr"][edge_mask]
        else:
            g["edge_attr"] = torch.zeros(0, DXF_EDGE_DIM)

        graphs.append(g)

    return graphs


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Graph2D contrastive pretraining (self-supervised)."
    )
    parser.add_argument("--dxf-dir", default=None, help="Directory of unlabeled DXF files.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--node-dim", type=int, default=DXF_NODE_DIM)
    parser.add_argument("--edge-dim", type=int, default=DXF_EDGE_DIM)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--model-type", choices=["gcn", "edge_sage"], default="edge_sage")
    parser.add_argument("--output", default="models/graph2d_pretrained_contrastive.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic random graphs (no DXF files needed).",
    )
    parser.add_argument("--dry-run-size", type=int, default=64, help="Synthetic dataset size.")
    args = parser.parse_args()

    if not args.dry_run and args.dxf_dir is None:
        parser.error("--dxf-dir is required unless --dry-run is set")

    logging.basicConfig(level=logging.INFO)

    print(f"Contrastive pretraining: model={args.model_type} hidden={args.hidden_dim} "
          f"epochs={args.epochs} lr={args.lr} temp={args.temperature}")

    dataset = UnlabeledDXFGraphDataset(
        dxf_dir=args.dxf_dir,
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        max_nodes=args.max_nodes,
        dry_run=args.dry_run,
        dry_run_size=args.dry_run_size,
    )
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        drop_last=False,
    )

    encoder = GraphEncoder(
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        model_type=args.model_type,
    )
    proj_head = ProjectionHead(
        input_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        output_dim=128,
    )

    result = pretrain(
        encoder,
        proj_head,
        dataloader,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        device=args.device,
        output_path=args.output,
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
    )

    print(f"Done. Best loss: {result['best_loss']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
