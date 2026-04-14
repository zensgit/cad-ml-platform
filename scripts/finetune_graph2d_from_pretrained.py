#!/usr/bin/env python3
"""Finetune a Graph2D model from a contrastive-pretrained encoder checkpoint.

Loads pretrained GNN encoder weights, attaches a new classification head,
and finetunes on labeled data.  Uses differential learning rates: lower for
the encoder (preserving pretrained representations) and higher for the new
classification head.

Usage:
    python scripts/finetune_graph2d_from_pretrained.py \
        --pretrained models/graph2d_pretrained_contrastive.pth \
        --manifest manifest.csv --dxf-dir /path/to/dxf

    python scripts/finetune_graph2d_from_pretrained.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ml.train.model_2d import (  # noqa: E402
    EdgeGraphSageClassifier,
    GraphEncoder,
    SimpleGraphClassifier,
)
from src.ml.train.dataset_2d import DXF_EDGE_DIM, DXF_NODE_DIM  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cached graph dataset (fast: loads pre-processed .pt files)
# --------------------------------------------------------------------------- #

class CachedGraphDataset(Dataset):
    """Load pre-processed graph tensors from .pt cache files.

    Eliminates per-epoch ezdxf parse cost: each __getitem__ is a single
    torch.load() instead of a full DXF → graph conversion.

    Cache files are produced by scripts/preprocess_dxf_to_graphs.py.
    Each .pt file contains keys: x, edge_index, edge_attr, label.
    """

    def __init__(self, cache_manifest_csv: str) -> None:
        self.samples: List[Tuple[str, int]] = []
        self.label_map: Dict[str, int] = {}

        with open(cache_manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = (
                    row.get("taxonomy_v2_class")
                    or row.get("label_cn")
                    or row.get("label")
                    or ""
                ).strip()
                cache_path = row.get("cache_path", "").strip()
                if not label or not cache_path:
                    continue
                if label not in self.label_map:
                    self.label_map[label] = len(self.label_map)
                self.samples.append((cache_path, self.label_map[label]))

        logger.info(
            "CachedGraphDataset: %d samples, %d classes", len(self.samples), len(self.label_map)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        cache_path, label_idx = self.samples[idx]
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=True)
            return {
                "x": data["x"],
                "edge_index": data["edge_index"],
                "edge_attr": data.get("edge_attr"),
            }, label_idx
        except Exception as e:
            logger.warning("Failed to load cache %s: %s", cache_path, e)
            return {
                "x": torch.zeros(1, DXF_NODE_DIM),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "edge_attr": torch.zeros(0, DXF_EDGE_DIM),
            }, label_idx


# --------------------------------------------------------------------------- #
# Finetuning dataset
# --------------------------------------------------------------------------- #

class FinetuneDataset(Dataset):
    """Minimal dataset for finetuning from a CSV manifest + DXF directory.

    In dry-run mode, generates synthetic data with random labels.
    """

    def __init__(
        self,
        manifest_csv: Optional[str] = None,
        dxf_dir: Optional[str] = None,
        node_dim: int = DXF_NODE_DIM,
        edge_dim: int = DXF_EDGE_DIM,
        dry_run: bool = False,
        dry_run_size: int = 64,
        dry_run_num_classes: int = 10,
    ) -> None:
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.dry_run = dry_run
        self.dry_run_size = dry_run_size
        self.dry_run_num_classes = dry_run_num_classes
        self.label_map: Dict[str, int] = {}
        self.samples: List[Tuple[str, int]] = []  # (file_path, label_idx)

        if dry_run:
            for i in range(dry_run_num_classes):
                self.label_map[f"class_{i}"] = i
            return

        if manifest_csv is None or dxf_dir is None:
            raise ValueError("manifest_csv and dxf_dir required when not in dry-run mode")

        self._load_manifest(manifest_csv, dxf_dir)

    def _load_manifest(self, manifest_csv: str, dxf_dir: str) -> None:
        dxf_path = Path(dxf_dir)
        labels_seen: Dict[str, int] = {}
        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = (row.get("taxonomy_v2_class") or row.get("label_cn") or row.get("label") or "").strip()
                # Support both absolute file_path and relative file_name
                fp = (row.get("file_path") or "").strip()
                file_name = (row.get("file_name") or row.get("file") or "").strip()
                if not label:
                    continue
                if fp and Path(fp).exists():
                    file_path = Path(fp)
                elif file_name:
                    file_path = dxf_path / file_name
                    if not file_path.exists():
                        continue
                else:
                    continue
                if label not in labels_seen:
                    labels_seen[label] = len(labels_seen)
                self.samples.append((str(file_path), labels_seen[label]))

        self.label_map = labels_seen
        logger.info(
            "Loaded %d samples, %d classes from %s",
            len(self.samples), len(self.label_map), manifest_csv,
        )

    def __len__(self) -> int:
        if self.dry_run:
            return self.dry_run_size
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        if self.dry_run:
            return self._synthetic_sample()
        return self._load_sample(idx)

    def _synthetic_sample(self) -> Tuple[Dict[str, torch.Tensor], int]:
        n = torch.randint(5, 30, (1,)).item()
        x = torch.randn(n, self.node_dim)
        num_edges = 2 * n
        src = torch.randint(0, n, (num_edges,))
        dst = torch.randint(0, n, (num_edges,))
        mask = src != dst
        src, dst = src[mask], dst[mask]
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.randn(edge_index.size(1), self.edge_dim)
        label = int(torch.randint(0, self.dry_run_num_classes, (1,)).item())
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}, label

    def _load_sample(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        file_path, label_idx = self.samples[idx]
        try:
            import ezdxf
            from src.ml.train.dataset_2d import DXFDataset

            doc = ezdxf.readfile(file_path)
            msp = doc.modelspace()
            ds = DXFDataset(root_dir=".", node_dim=self.node_dim, return_edge_attr=True)
            x, edge_index, edge_attr = ds._dxf_to_graph(
                msp, self.node_dim, return_edge_attr=True
            )
            return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}, label_idx
        except Exception as e:
            logger.warning("Failed to load %s: %s", file_path, e)
            return {
                "x": torch.zeros(1, self.node_dim),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "edge_attr": torch.zeros(0, self.edge_dim),
            }, label_idx


def collate_finetune(
    batch: List[Tuple[Dict[str, torch.Tensor], int]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Collate graphs and labels into batched tensors."""
    xs, edge_indices, edge_attrs, batch_ids, labels = [], [], [], [], []
    offset = 0
    graph_idx = 0
    for i, (g, label) in enumerate(batch):
        n = g["x"].size(0)
        if n == 0:
            continue  # skip empty/failed graphs
        xs.append(g["x"])
        if g["edge_index"].numel() > 0:
            edge_indices.append(g["edge_index"] + offset)
        else:
            edge_indices.append(g["edge_index"])
        if "edge_attr" in g and g["edge_attr"] is not None:
            edge_attrs.append(g["edge_attr"])
        batch_ids.append(torch.full((n,), graph_idx, dtype=torch.long))
        labels.append(label)
        offset += n
        graph_idx += 1

    batched = {
        "x": torch.cat(xs, dim=0) if xs else torch.zeros(0, 1),
        "edge_index": torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros(2, 0, dtype=torch.long),
        "edge_attr": torch.cat(edge_attrs, dim=0) if edge_attrs else None,
        "batch": torch.cat(batch_ids, dim=0) if batch_ids else torch.zeros(0, dtype=torch.long),
    }
    return batched, torch.tensor(labels, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Finetuning
# --------------------------------------------------------------------------- #

def finetune(
    pretrained_path: Optional[str],
    dataset: Dataset,
    *,
    epochs: int = 50,
    encoder_lr: float = 0.0001,
    head_lr: float = 0.001,
    batch_size: int = 16,
    patience: int = 5,
    device: str = "cpu",
    output_path: str = "models/graph2d_finetuned.pth",
    node_dim: int = DXF_NODE_DIM,
    edge_dim: int = DXF_EDGE_DIM,
    hidden_dim: int = 64,
    model_type: str = "edge_sage",
    val_split: float = 0.2,
    val_manifest: Optional[str] = None,
    sampler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Finetune a pretrained encoder on labeled data."""

    num_classes = len(dataset.label_map)
    if num_classes < 2:
        print(f"Warning: only {num_classes} class(es) found, need at least 2 for classification")
        num_classes = max(num_classes, 2)

    # Build encoder + classifier
    encoder = GraphEncoder(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        model_type=model_type,
    )

    # Load pretrained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        ckpt = torch.load(pretrained_path, map_location="cpu")
        encoder_state = ckpt.get("encoder_state_dict", {})
        encoder.load_state_dict(encoder_state, strict=False)
        print(f"Loaded pretrained encoder from {pretrained_path}")
    else:
        print("Training from scratch (no pretrained weights loaded)")

    classifier = nn.Linear(hidden_dim, num_classes)

    encoder = encoder.to(device)
    classifier = classifier.to(device)

    # Differential learning rates
    optimizer = torch.optim.Adam([
        {"params": encoder.parameters(), "lr": encoder_lr},
        {"params": classifier.parameters(), "lr": head_lr},
    ])

    criterion = nn.CrossEntropyLoss()

    # Train/val split
    if val_manifest:
        # Golden validation set: use fixed manifest and exclude val from training
        import csv as _csv
        val_ds = CachedGraphDataset(val_manifest)
        val_paths = set()
        with open(val_manifest, "r", encoding="utf-8") as _f:
            for _row in _csv.DictReader(_f):
                _p = _row.get("cache_path", "").strip()
                if _p:
                    val_paths.add(_p)
        # Exclude val samples from training to prevent leakage
        if hasattr(dataset, "samples"):
            train_indices = [
                i for i, (cp, _) in enumerate(dataset.samples) if cp not in val_paths
            ]
            overlap = len(dataset) - len(train_indices)
            if overlap > 0:
                logger.info("Leakage prevention: removed %d val samples from training set", overlap)
            train_ds = torch.utils.data.Subset(dataset, train_indices)
        else:
            train_ds = dataset
            logger.warning("Cannot check leakage: dataset has no .samples attribute")
        logger.info("Golden val: %d samples, train (excl val): %d samples",
                     len(val_ds), len(train_ds))
    else:
        total = len(dataset)
        val_size = max(1, int(total * val_split))
        train_size = total - val_size
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    # If a sampler is provided it applies only to the train split; sampler
    # indices reference train_ds, not the full dataset, so we remap weights.
    train_sampler = None
    if sampler is not None:
        # sampler was built on the full dataset indices; rebuild for train subset
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        if hasattr(dataset, "samples"):
            train_indices = train_ds.indices  # type: ignore[attr-defined]
            class_counts: Dict[int, int] = Counter(
                dataset.samples[i][1] for i in train_indices  # type: ignore[union-attr]
            )
            threshold = max(class_counts.values()) // 4  # heuristic
            w = [
                max(1.0, threshold / class_counts[dataset.samples[i][1]])  # type: ignore[union-attr]
                for i in train_indices
            ]
            train_sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

    use_shuffle = train_sampler is None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=use_shuffle,
        sampler=train_sampler, collate_fn=collate_finetune,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_finetune)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        # Train
        encoder.train()
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_data, batch_labels in train_loader:
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
            batch_labels = batch_labels.to(device)

            embeddings = encoder(
                batch_data["x"], batch_data["edge_index"],
                edge_attr=batch_data.get("edge_attr"),
                batch=batch_data["batch"],
            )
            logits = classifier(embeddings)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_labels.size(0)
            train_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        # Validate
        encoder.eval()
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
                batch_labels = batch_labels.to(device)

                embeddings = encoder(
                    batch_data["x"], batch_data["edge_index"],
                    edge_attr=batch_data.get("edge_attr"),
                    batch=batch_data["batch"],
                )
                logits = classifier(embeddings)
                loss = criterion(logits, batch_labels)

                val_loss += loss.item() * batch_labels.size(0)
                val_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>3d} | train_loss={avg_train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.3f}"
            )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {
                "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                "classifier": {k: v.cpu().clone() for k, v in classifier.state_dict().items()},
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Restore best model and save
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        classifier.load_state_dict(best_state["classifier"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save in a format compatible with existing Graph2D loading
    if model_type == "edge_sage":
        full_model = EdgeGraphSageClassifier(node_dim, edge_dim, hidden_dim, num_classes)
        full_model.sage1.load_state_dict(encoder.sage1.state_dict())
        full_model.sage2.load_state_dict(encoder.sage2.state_dict())
        full_model.classifier.load_state_dict(classifier.state_dict())
    else:
        full_model = SimpleGraphClassifier(node_dim, hidden_dim, num_classes)
        full_model.gcn1.load_state_dict(encoder.gcn1.state_dict())
        full_model.gcn2.load_state_dict(encoder.gcn2.state_dict())
        full_model.classifier.load_state_dict(classifier.state_dict())

    checkpoint = {
        "model_state_dict": full_model.state_dict(),
        "label_map": dataset.label_map,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "hidden_dim": hidden_dim,
        "model_type": model_type,
        "best_val_loss": best_val_loss,
        "pretrained_from": pretrained_path or "scratch",
    }
    torch.save(checkpoint, output_path)
    print(f"Saved finetuned model to {output_path}")

    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoint": output_path,
        "num_classes": num_classes,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finetune Graph2D from contrastive-pretrained encoder."
    )
    parser.add_argument("--pretrained", default=None, help="Path to pretrained contrastive checkpoint.")
    parser.add_argument("--manifest", default=None, help="CSV manifest with labels.")
    parser.add_argument("--dxf-dir", default=None, help="DXF files directory.")
    parser.add_argument("--use-cache", action="store_true",
                        help="Load from pre-processed .pt cache (manifest must be cache_manifest.csv).")
    parser.add_argument("--tail-oversample-threshold", type=int, default=0,
                        help="Apply WeightedRandomSampler to classes with fewer than N samples. "
                             "0 = disabled (default). Recommended: 50 for 24-class training.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=0.0001)
    parser.add_argument("--head-lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--node-dim", type=int, default=DXF_NODE_DIM)
    parser.add_argument("--edge-dim", type=int, default=DXF_EDGE_DIM)
    parser.add_argument("--model-type", choices=["gcn", "edge_sage"], default="edge_sage")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--val-manifest", default=None,
                        help="Fixed validation manifest CSV. Overrides --val-split. "
                             "Recommended: data/manifests/golden_val_set.csv")
    parser.add_argument("--output", default="models/graph2d_finetuned.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data.")
    parser.add_argument("--dry-run-size", type=int, default=64)
    parser.add_argument("--dry-run-num-classes", type=int, default=10)
    args = parser.parse_args()

    if not args.dry_run and not args.use_cache:
        if args.manifest is None or args.dxf_dir is None:
            parser.error("--manifest and --dxf-dir required unless --dry-run or --use-cache is set")

    if args.use_cache and args.manifest is None:
        parser.error("--manifest (cache_manifest.csv path) required with --use-cache")

    logging.basicConfig(level=logging.INFO)

    if args.use_cache:
        dataset: Dataset = CachedGraphDataset(cache_manifest_csv=args.manifest)
    else:
        dataset = FinetuneDataset(
            manifest_csv=args.manifest,
            dxf_dir=args.dxf_dir,
            node_dim=args.node_dim,
            edge_dim=args.edge_dim,
            dry_run=args.dry_run,
            dry_run_size=args.dry_run_size,
            dry_run_num_classes=args.dry_run_num_classes,
        )

    # Optional: long-tail over-sampling
    sampler = None
    if args.tail_oversample_threshold > 0 and hasattr(dataset, "samples"):
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        class_counts: Dict[int, int] = Counter(label for _, label in dataset.samples)
        threshold = args.tail_oversample_threshold
        weights = [
            max(1.0, threshold / class_counts[label])
            for _, label in dataset.samples
        ]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        logger.info(
            "Long-tail sampler: threshold=%d, boosted %d classes",
            threshold,
            sum(1 for c, n in class_counts.items() if n < threshold),
        )

    result = finetune(
        pretrained_path=args.pretrained,
        dataset=dataset,
        epochs=args.epochs,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        output_path=args.output,
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        hidden_dim=args.hidden_dim,
        model_type=args.model_type,
        val_split=args.val_split,
        val_manifest=getattr(args, "val_manifest", None),
        sampler=sampler,
    )

    print(f"Done. Best val loss: {result['best_val_loss']:.4f}, classes: {result['num_classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
