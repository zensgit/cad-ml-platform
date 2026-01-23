#!/usr/bin/env python3
"""
Train UV-Net graph model on STEP B-Rep graphs.
"""

import argparse
import os
import random
import sys
import time
from typing import Optional, Tuple

sys.path.append(".")

try:
    import torch
except ImportError as exc:  # pragma: no cover - environment dependent
    print(f"ERROR: torch not available: {exc}")
    raise SystemExit(1)

from torch.utils.data import DataLoader, Dataset, random_split

from src.core.geometry.engine import (
    BREP_GRAPH_EDGE_FEATURES,
    BREP_GRAPH_NODE_FEATURES,
    HAS_OCC,
)
from src.ml.train.dataset import ABCDataset
from src.ml.train.model import UVNetGraphModel
from src.ml.train.trainer import GraphBatchCollate, UVNetTrainer


class SyntheticGraphDataset(Dataset):
    def __init__(
        self,
        samples: int,
        node_dim: int,
        edge_dim: int,
        num_classes: int,
        seed: int,
    ) -> None:
        self.samples = samples
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.random = random.Random(seed)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Tuple[dict, int]:
        num_nodes = self.random.randint(6, 18)
        num_edges = self.random.randint(max(1, num_nodes - 1), num_nodes * 3)
        x = torch.randn(num_nodes, self.node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, self.edge_dim)
        label = self.random.randint(0, self.num_classes - 1)
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}, label


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataloaders(
    dataset: Dataset,
    batch_size: int,
    val_split: float,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0.0, 1.0).")

    if val_split == 0.0 or len(dataset) < 2:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=GraphBatchCollate(),
        )
        return train_loader, None

    val_size = max(1, int(len(dataset) * val_split))
    train_size = max(1, len(dataset) - val_size)
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=GraphBatchCollate(),
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=GraphBatchCollate()
    )
    return train_loader, val_loader


def main() -> int:
    parser = argparse.ArgumentParser(description="Train UV-Net graph model.")
    parser.add_argument("--data-dir", default="data/abc_subset", help="STEP data directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument(
        "--label-strategy",
        default="surface_bucket",
        choices=("random", "surface_bucket"),
        help="Label strategy for pseudo-supervision.",
    )
    parser.add_argument("--num-classes", type=int, default=0, help="Override class count.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples.")
    parser.add_argument(
        "--output",
        default="models/uvnet_graph_latest.pth",
        help="Checkpoint output path.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic graph data instead of STEP parsing.",
    )
    parser.add_argument(
        "--synthetic-samples", type=int, default=200, help="Synthetic dataset size."
    )
    args = parser.parse_args()

    _set_seed(args.seed)

    node_dim = len(BREP_GRAPH_NODE_FEATURES)
    edge_dim = len(BREP_GRAPH_EDGE_FEATURES)

    if args.synthetic:
        dataset: Dataset = SyntheticGraphDataset(
            samples=args.synthetic_samples,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_classes=args.num_classes or 5,
            seed=args.seed,
        )
        num_classes = args.num_classes or 5
    else:
        if not HAS_OCC:
            print("pythonocc-core not available; training skipped.")
            return 0
        if not os.path.exists(args.data_dir):
            print(f"Data directory not found: {args.data_dir}")
            return 1
        dataset = ABCDataset(
            args.data_dir,
            output_format="graph",
            graph_backend="dict",
            label_strategy=args.label_strategy,
        )
        if args.limit > 0:
            dataset.file_list = dataset.file_list[: args.limit]
        if len(dataset) == 0:
            print("No STEP files found for training.")
            return 1
        num_classes = args.num_classes or dataset.num_classes

    train_loader, val_loader = _build_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    model = UVNetGraphModel(
        node_input_dim=node_dim,
        num_classes=num_classes,
        node_schema=BREP_GRAPH_NODE_FEATURES,
        edge_schema=BREP_GRAPH_EDGE_FEATURES,
    )
    trainer = UVNetTrainer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        start = time.time()
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader) if val_loader else {}
        dur = time.time() - start
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics.get('val_loss', 0.0):.4f} "
            f"val_acc={val_metrics.get('val_accuracy', 0.0):.4f} "
            f"time={dur:.2f}s"
        )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    trainer.save_checkpoint(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
