#!/usr/bin/env python3
"""Train a lightweight 2D graph classifier on DXF manifest data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
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


def _collate(
    batch: List[Tuple[Dict[str, Any], Any]]
) -> Tuple[List[Any], List[Any], List[Any], List[Any], List[str]]:
    xs, edges, edge_attrs, labels, filenames = [], [], [], [], []
    for graph, label in batch:
        xs.append(graph["x"])
        edges.append(graph["edge_index"])
        edge_attrs.append(graph.get("edge_attr"))
        labels.append(label)
        filenames.append(graph.get("file_name", ""))
    return xs, edges, edge_attrs, labels, filenames


def _compute_class_weights(labels: List[int], num_classes: int, mode: str):
    import torch

    weights = torch.ones(num_classes, dtype=torch.float)
    if mode == "none":
        return weights
    counts = Counter(labels)
    total = sum(counts.values())
    for idx in range(num_classes):
        count = counts.get(idx, 0)
        if count <= 0:
            weights[idx] = 0.0
            continue
        base = total / (num_classes * count)
        weights[idx] = base if mode == "inverse" else base**0.5
    return weights


def _compute_log_prior(labels: List[int], num_classes: int):
    import torch

    counts = Counter(labels)
    total = sum(counts.values())
    priors = []
    for idx in range(num_classes):
        count = counts.get(idx, 0)
        prior = count / total if total else 0.0
        priors.append(max(prior, 1e-6))
    prior_tensor = torch.tensor(priors, dtype=torch.float)
    return torch.log(prior_tensor)


def _stratified_split(indices: List[int], labels: List[int], train_ratio: float):
    label_to_indices: Dict[int, List[int]] = {}
    for idx, label in zip(indices, labels):
        label_to_indices.setdefault(label, []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for label, idxs in label_to_indices.items():
        random.shuffle(idxs)
        train_count = max(1, int(len(idxs) * train_ratio))
        if len(idxs) > 1 and train_count == len(idxs):
            train_count -= 1
        train_idx.extend(idxs[:train_count])
        val_idx.extend(idxs[train_count:])

    if not val_idx:
        val_idx = train_idx[:1]
        train_idx = train_idx[1:] if len(train_idx) > 1 else train_idx

    return train_idx, val_idx


def _build_balanced_sampler(labels: List[int]):
    import torch
    from torch.utils.data import WeightedRandomSampler

    counts = Counter(labels)
    if not counts:
        return None
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def main() -> int:
    if not _require_torch():
        return 1

    import torch
    from torch.utils.data import DataLoader, Subset

    from src.ml.train.dataset_2d import (
        DXFManifestDataset,
        DXF_EDGE_DIM,
        DXF_NODE_DIM,
        DXF_NODE_FEATURES,
        DXF_NODE_FEATURES_LEGACY,
    )
    from src.ml.train.model_2d import EdgeGraphSageClassifier, SimpleGraphClassifier

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
    parser.add_argument("--node-dim", type=int, default=DXF_NODE_DIM)
    parser.add_argument(
        "--model",
        choices=["gcn", "edge_sage"],
        default="gcn",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--edge-dim",
        type=int,
        default=DXF_EDGE_DIM,
        help="Edge feature dimension for edge-aware models.",
    )
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
    parser.add_argument(
        "--class-weighting",
        choices=["none", "inverse", "sqrt"],
        default="none",
        help="Optional class weighting strategy.",
    )
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "focal", "logit_adjusted"],
        default="cross_entropy",
        help="Loss function to use for training.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha parameter.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter.",
    )
    parser.add_argument(
        "--logit-adjustment-tau",
        type=float,
        default=1.0,
        help="Logit adjustment tau parameter.",
    )
    parser.add_argument(
        "--sampler",
        choices=["none", "balanced"],
        default="none",
        help="Optional sampling strategy for the training split.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["random", "stratified"],
        default="stratified",
        help="Train/validation split strategy.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable simple feature jitter augmentation for training graphs.",
    )
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=0.5,
        help="Probability of applying augmentation to each training sample.",
    )
    parser.add_argument(
        "--augment-scale",
        type=float,
        default=0.05,
        help="Noise scale for feature jitter augmentation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="models/graph2d_merged_latest.pth")
    parser.add_argument(
        "--distill", action="store_true", help="Enable knowledge distillation training"
    )
    parser.add_argument(
        "--teacher",
        choices=["filename", "hybrid"],
        default="hybrid",
        help="Teacher model type for distillation",
    )
    parser.add_argument(
        "--distill-alpha",
        type=float,
        default=0.3,
        help="Distillation loss weight (0.0-1.0)",
    )
    parser.add_argument(
        "--distill-temp",
        type=float,
        default=3.0,
        help="Distillation temperature",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Early stopping patience (0 to disable). Stop if val_acc doesn't improve for N epochs.",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save the model with best validation accuracy instead of final epoch.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        device = torch.device("cpu")
    print(f"Using device: {device}")

    use_edge_attr = args.model == "edge_sage"
    dataset = DXFManifestDataset(
        args.manifest,
        args.dxf_dir,
        node_dim=args.node_dim,
        return_edge_attr=use_edge_attr,
    )
    if args.max_samples and args.max_samples > 0:
        dataset.samples = dataset.samples[: args.max_samples]
    if len(dataset) == 0:
        print("Empty dataset; aborting.")
        return 1

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    if args.split_strategy == "stratified":
        labels = [dataset.samples[idx]["label_id"] for idx in indices]
        train_idx, val_idx = _stratified_split(indices, labels, 0.8)
    else:
        split = max(1, int(len(indices) * 0.8))
        train_idx = indices[:split]
        val_idx = indices[split:] or indices[:1]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=args.sampler == "none",
        sampler=None,
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
    if args.model == "edge_sage":
        model = EdgeGraphSageClassifier(
            args.node_dim, args.edge_dim, args.hidden_dim, num_classes
        )
    else:
        model = SimpleGraphClassifier(args.node_dim, args.hidden_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_labels = [dataset.samples[idx]["label_id"] for idx in train_idx]
    sampler = None
    if args.sampler == "balanced":
        sampler = _build_balanced_sampler(train_labels)
        if sampler is not None:
            train_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=args.batch_size,
                shuffle=False,
                sampler=sampler,
                collate_fn=_collate,
            )
    from src.ml.class_balancer import ClassBalancer
    from src.ml.knowledge_distillation import DistillationLoss, TeacherModel

    # Initialize Teacher if needed
    teacher_model = None
    distill_loss_fn = None
    if args.distill:
        print(f"Initializing teacher model: {args.teacher}")
        teacher_model = TeacherModel(
            teacher_type=args.teacher,
            label_to_idx=label_map,
            num_classes=num_classes,
        )
        distill_loss_fn = DistillationLoss(
            alpha=args.distill_alpha,
            temperature=args.distill_temp,
        )

    balance_strategy = "none"
    if args.loss == "focal":
        balance_strategy = "focal"
    elif args.loss == "logit_adjusted":
        balance_strategy = "logit_adj"
    elif args.class_weighting != "none":
        balance_strategy = "weights"

    weight_mode = args.class_weighting if args.class_weighting != "none" else "sqrt"
    balancer = ClassBalancer(
        strategy=balance_strategy,
        weight_mode=weight_mode,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        logit_adj_tau=args.logit_adjustment_tau,
    )

    if args.downweight_label and args.downweight_label in label_map:
        factor = max(0.05, min(1.0, float(args.downweight_factor)))
        label_idx = label_map[args.downweight_label]
        print(
            f"Downweighting label {args.downweight_label!r} (idx={label_idx}) "
            f"with factor {factor:.2f}"
        )

    class_counts = None
    if balance_strategy == "logit_adj":
        counts = Counter(train_labels)
        class_counts = [counts.get(i, 1) for i in range(num_classes)]

    labels_for_loss = None
    if balance_strategy in {"weights", "focal"} and args.class_weighting != "none":
        labels_for_loss = train_labels
    elif balance_strategy == "weights":
        labels_for_loss = train_labels

    criterion = balancer.get_loss_function(
        labels=labels_for_loss,
        num_classes=num_classes,
        class_counts=class_counts,
    )

    class_stats = balancer.get_class_distribution(train_labels)
    print(
        "Class balance:",
        f"classes={class_stats['num_classes']} "
        f"min={class_stats['min_count']} max={class_stats['max_count']} "
        f"ratio={class_stats['imbalance_ratio']:.2f} strategy={balance_strategy}",
    )

    feature_idx = {name: i for i, name in enumerate(DXF_NODE_FEATURES)}

    def _augment_features(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or args.node_dim < len(DXF_NODE_FEATURES_LEGACY):
            return x
        x = x.clone()
        scale = float(args.augment_scale)
        for name in ("length_norm", "radius_norm", "center_x_norm", "center_y_norm"):
            idx = feature_idx.get(name)
            if idx is None or idx >= args.node_dim:
                continue
            x[:, idx] = (x[:, idx] + torch.randn_like(x[:, idx]) * scale).clamp(0.0, 1.0)
        for name in ("dir_x", "dir_y"):
            idx = feature_idx.get(name)
            if idx is None or idx >= args.node_dim:
                continue
            x[:, idx] = (x[:, idx] + torch.randn_like(x[:, idx]) * scale).clamp(-1.0, 1.0)
        return x

    model.to(device)

    # Early stopping and best model tracking
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch_data in train_loader:
            # Unpack batch (custom collate returns 5 items now including filenames)
            xs, edges, edge_attrs, labels, filenames = batch_data

            optimizer.zero_grad()
            batch_loss = 0.0
            batch_count = 0

            for i, (x, edge_index, edge_attr, label, filename) in enumerate(
                zip(xs, edges, edge_attrs, labels, filenames)
            ):
                if x.numel() == 0:
                    continue

                # Move to device
                x = x.to(device)
                edge_index = edge_index.to(device)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(device)
                label = label.to(device)

                if args.augment and random.random() < args.augment_prob:
                    x = _augment_features(x)

                if use_edge_attr:
                    logits = model(x, edge_index, edge_attr)
                else:
                    logits = model(x, edge_index)

                # Calculate loss
                if args.distill and teacher_model and distill_loss_fn:
                    # Generate teacher soft labels for this sample
                    teacher_logits = teacher_model.generate_soft_labels([filename])
                    teacher_logits = teacher_logits.to(device)
                    loss, _ = distill_loss_fn(logits, teacher_logits, label.view(1))
                else:
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
            for xs, edges, edge_attrs, labels, _ in val_loader:
                for x, edge_index, edge_attr, label in zip(xs, edges, edge_attrs, labels):
                    if x.numel() == 0:
                        continue
                    x = x.to(device)
                    edge_index = edge_index.to(device)
                    if edge_attr is not None:
                        edge_attr = edge_attr.to(device)
                    if use_edge_attr:
                        logits = model(x, edge_index, edge_attr)
                    else:
                        logits = model(x, edge_index)
                    pred = int(torch.argmax(logits, dim=1)[0])
                    if pred == int(label):
                        correct += 1
                    total += 1
        acc = correct / max(1, total)

        # Track best model and early stopping
        improved = acc > best_val_acc
        if improved:
            best_val_acc = acc
            epochs_without_improvement = 0
            if args.save_best:
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        status = "← best" if improved else ""
        print(f"Epoch {epoch}/{args.epochs} loss={avg_loss:.4f} val_acc={acc:.3f} {status}")

        # Early stopping check
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
            print(f"Best validation accuracy: {best_val_acc:.3f}")
            break

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use best model state if --save-best was enabled and we have one
    save_state = best_model_state if (args.save_best and best_model_state is not None) else model.state_dict()

    torch.save(
        {
            "model_state_dict": save_state,
            "label_map": label_map,
            "node_dim": args.node_dim,
            "hidden_dim": args.hidden_dim,
            "loss_type": args.loss,
            "model_type": args.model,
            "edge_dim": args.edge_dim if use_edge_attr else 0,
            "best_val_acc": best_val_acc,
        },
        out_path,
    )
    if args.save_best and best_model_state is not None:
        print(f"Saved best checkpoint (val_acc={best_val_acc:.3f}): {out_path}")
    else:
        print(f"Saved checkpoint: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
