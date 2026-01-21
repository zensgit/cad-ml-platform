#!/usr/bin/env python3
"""Train a lightweight 2D graph classifier on DXF manifest data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def _require_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception as exc:  # pragma: no cover - runtime check
        print(f"Torch not available: {exc}")
        return False


def _collate(batch: List[Tuple[Dict[str, Any], Any]]) -> Tuple[List[Any], List[Any], List[Any]]:
    xs, edges, labels = [], [], []
    for graph, label in batch:
        xs.append(graph["x"])
        edges.append(graph["edge_index"])
        labels.append(label)
    return xs, edges, labels


def main() -> int:
    if not _require_torch():
        return 1

    import torch
    from torch.utils.data import DataLoader, Subset

    from src.ml.train.dataset_2d import DXFManifestDataset, DXF_NODE_DIM
    from src.ml.train.model_2d import SimpleGraphClassifier

    parser = argparse.ArgumentParser(description="Train 2D DXF graph classifier.")
    parser.add_argument(
        "--manifest",
        default="reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv",
        help="Manifest CSV path",
    )
    parser.add_argument(
        "--dxf-dir",
        default="/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf",
        help="DXF directory",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--downweight-label",
        default="",
        help="Optional label name to downweight in the loss function.",
    )
    parser.add_argument(
        "--downweight-factor",
        type=float,
        default=0.3,
        help="Multiplier applied to the downweighted label (0.0-1.0).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="models/graph2d_merged_latest.pth")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = DXFManifestDataset(args.manifest, args.dxf_dir)
    if args.max_samples and args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]
    if len(dataset) == 0:
        print("Empty dataset; aborting.")
        return 1

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = max(1, int(len(indices) * 0.8))
    train_idx = indices[:split]
    val_idx = indices[split:] or indices[:1]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    label_map = dataset.get_label_map()
    num_classes = len(label_map)
    model = SimpleGraphClassifier(DXF_NODE_DIM, args.hidden_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    weight = torch.ones(num_classes, dtype=torch.float)
    if args.downweight_label and args.downweight_label in label_map:
        factor = max(0.05, min(1.0, float(args.downweight_factor)))
        label_idx = label_map[args.downweight_label]
        weight[label_idx] = factor
        print(
            f"Downweighting label {args.downweight_label!r} (idx={label_idx}) "
            f"with factor {factor:.2f}"
        )
    criterion = torch.nn.CrossEntropyLoss(weight=weight)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for xs, edges, labels in train_loader:
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_count = 0
            for x, edge_index, label in zip(xs, edges, labels):
                if x.numel() == 0:
                    continue
                logits = model(x, edge_index)
                loss = criterion(logits, label.view(1))
                batch_loss += loss
                batch_count += 1
            if batch_count == 0:
                continue
            batch_loss = batch_loss / batch_count
            batch_loss.backward()
            optimizer.step()
            total_loss += float(batch_loss.detach()) * batch_count
            total_seen += batch_count
        avg_loss = total_loss / max(1, total_seen)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xs, edges, labels in val_loader:
                for x, edge_index, label in zip(xs, edges, labels):
                    if x.numel() == 0:
                        continue
                    logits = model(x, edge_index)
                    pred = int(torch.argmax(logits, dim=1)[0])
                    if pred == int(label):
                        correct += 1
                    total += 1
        acc = correct / max(1, total)
        print(f"Epoch {epoch}/{args.epochs} loss={avg_loss:.4f} val_acc={acc:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_map": label_map,
            "node_dim": DXF_NODE_DIM,
            "hidden_dim": args.hidden_dim,
        },
        out_path,
    )
    print(f"Saved checkpoint: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
