#!/usr/bin/env python3
"""Fine-tune GraphEncoderV2 from a B4.4 checkpoint on augmented data.

Loads the existing B4.4 model weights and continues training on the
augmented cache manifest. Uses focal loss to handle residual imbalance.

Usage:
    python scripts/finetune_graph2d_v2_augmented.py \
        --checkpoint models/graph2d_finetuned_24class_v2.pth \
        --manifest data/graph_cache_aug/cache_manifest_aug.csv \
        --output models/graph2d_finetuned_24class_v3.pth \
        --epochs 60 --lr 5e-5
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ml.train.model_2d import GraphEncoderV2WithHead
from scripts.finetune_graph2d_from_pretrained import CachedGraphDataset, collate_finetune


# --------------------------------------------------------------------------- #
# Focal Loss
# --------------------------------------------------------------------------- #

class FocalLoss(nn.Module):
    """Focal loss for multi-class classification with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 1.5, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # per-class weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()

        # Gather log-prob of true class
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - pt) ** self.gamma
        loss = -focal_factor * log_pt

        if self.weight is not None:
            w = self.weight.to(logits.device)
            loss = loss * w[targets]

        return loss.mean()


# --------------------------------------------------------------------------- #
# Weighted sampler
# --------------------------------------------------------------------------- #

def build_sampler(dataset: CachedGraphDataset, train_indices: list[int]) -> WeightedRandomSampler:
    """WeightedRandomSampler that up-weights rare classes in training set."""
    from collections import Counter
    class_counts = Counter(dataset.samples[i][1] for i in train_indices)
    max_count = max(class_counts.values())
    sample_weights = [max_count / class_counts[dataset.samples[i][1]] for i in train_indices]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    # Load dataset
    print(f"Loading dataset from {args.manifest} ...", flush=True)
    ds = CachedGraphDataset(args.manifest)
    num_classes = len(ds.label_map)
    print(f"  {len(ds)} samples, {num_classes} classes", flush=True)

    # Train / val split
    if args.val_manifest:
        # Phase 2: Use fixed golden validation set
        val_ds = CachedGraphDataset(args.val_manifest)
        train_ds = ds  # Full manifest used for training (golden val is a separate file)
        print(f"  Using golden val manifest: {args.val_manifest} ({len(val_ds)} samples)")
    else:
        val_size = int(args.val_split * len(ds))
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(
            ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    # Weighted sampler for training
    sampler = build_sampler(ds, list(train_ds.indices))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collate_finetune, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_finetune,
    )

    # Load or initialise model
    device = torch.device(args.device)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = GraphEncoderV2WithHead.from_checkpoint(ckpt)

        # If checkpoint label_map doesn't match augmented label_map, reinit head
        ckpt_label_map = ckpt.get("label_map", {})
        if set(ckpt_label_map.keys()) != set(ds.label_map.keys()):
            print("  Label map changed — reinitialising classifier head.", flush=True)
            hidden_dim = ckpt.get("hidden_dim", 256)
            model.classifier = nn.Linear(hidden_dim, num_classes)
    else:
        print("No checkpoint found — training from scratch.", flush=True)
        model = GraphEncoderV2WithHead(
            node_dim=19, edge_dim=7, hidden_dim=256, num_classes=num_classes
        )

    model = model.to(device)

    # Focal loss (gamma=1.5 for moderate focus on hard samples)
    criterion = FocalLoss(gamma=args.focal_gamma)

    # Differential LRs: lower for encoder body, higher for classifier head
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr, "weight_decay": 1e-4},
        {"params": model.classifier.parameters(), "lr": args.head_lr, "weight_decay": 0.0},
    ])

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.encoder_lr * 0.1
    )

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...\n", flush=True)
    print(f"{'Epoch':>6} {'train_loss':>10} {'train_acc':>10} {'val_acc':>8} {'best':>6}", flush=True)
    print("-" * 45, flush=True)

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = train_correct = train_total = 0

        for bd, bl in train_loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in bd.items()}
            bl = bl.to(device)

            logits = model(bd["x"], bd["edge_index"],
                           edge_attr=bd.get("edge_attr"), batch=bd.get("batch"))
            loss = criterion(logits, bl)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * bl.size(0)
            train_correct += (logits.argmax(1) == bl).sum().item()
            train_total += bl.size(0)

        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_correct = val_total = 0

        with torch.no_grad():
            for bd, bl in val_loader:
                bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in bd.items()}
                bl = bl.to(device)
                logits = model(bd["x"], bd["edge_index"],
                               edge_attr=bd.get("edge_attr"), batch=bd.get("batch"))
                val_correct += (logits.argmax(1) == bl).sum().item()
                val_total += bl.size(0)

        t_loss = train_loss / max(train_total, 1)
        t_acc = train_correct / max(train_total, 1)
        v_acc = val_correct / max(val_total, 1)
        is_best = v_acc > best_val_acc
        marker = " ✓" if is_best else ""

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"{epoch:>6d} {t_loss:>10.4f} {t_acc:>10.3f} {v_acc:>8.3f}{marker}", flush=True)

        if is_best:
            best_val_acc = v_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    # ---- Save best ----
    if best_state is not None:
        model.load_state_dict(best_state)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "arch": "GraphEncoderV2",
        "encoder": model.encoder.state_dict(),
        "classifier": model.classifier.state_dict(),
        "label_map": ds.label_map,
        "hidden_dim": model.encoder.lin_in.out_features,
        "node_dim": 19,
        "edge_dim": 7,
        "best_val_acc": best_val_acc,
        "trained_from": args.checkpoint or "scratch",
    }
    torch.save(checkpoint, args.output)
    print(f"\nBest val acc: {best_val_acc:.1%}")
    print(f"Saved to: {args.output}", flush=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GraphEncoderV2 on augmented data.")
    parser.add_argument("--checkpoint", default="models/graph2d_finetuned_24class_v2.pth",
                        help="B4.4 checkpoint to start from.")
    parser.add_argument("--manifest", default="data/graph_cache_aug/cache_manifest_aug.csv",
                        help="Augmented cache manifest CSV.")
    parser.add_argument("--output", default="models/graph2d_finetuned_24class_v3.pth")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-lr", type=float, default=5e-5,
                        help="Lower LR for pretrained encoder body.")
    parser.add_argument("--head-lr", type=float, default=5e-4,
                        help="Higher LR for classifier head.")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--val-manifest", default=None,
                        help="Fixed validation manifest CSV. Overrides --val-split. "
                             "Recommended: data/manifests/golden_val_set.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
