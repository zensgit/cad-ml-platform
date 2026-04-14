#!/usr/bin/env python3
"""Evaluate Graph2D v2 (B4.4) model on cached graph dataset.

Produces:
  - Overall accuracy, Top-3 accuracy, Macro F1
  - Per-class precision / recall / F1
  - Confusion matrix summary (top misclassifications)
  - Comparison table vs baseline

Usage:
    # Evaluate B4.4 model on full cache
    python scripts/evaluate_graph2d_v2.py \
        --model models/graph2d_finetuned_24class_v2.pth \
        --manifest data/graph_cache/cache_manifest.csv

    # Compare v1 vs v2
    python scripts/evaluate_graph2d_v2.py \
        --model models/graph2d_finetuned_24class_v2.pth \
        --baseline models/graph2d_finetuned_24class_v1.pth \
        --manifest data/graph_cache/cache_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ml.train.model_2d import GraphEncoderV2WithHead, EdgeGraphSageClassifier, GraphEncoder
from scripts.finetune_graph2d_from_pretrained import CachedGraphDataset, collate_finetune
from torch.utils.data import DataLoader, random_split


def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    label_map = ckpt["label_map"]
    arch = ckpt.get("arch", "")
    if arch == "GraphEncoderV2":
        model = GraphEncoderV2WithHead.from_checkpoint(ckpt)
    else:
        # Legacy: encoder + classifier separate
        hidden_dim = ckpt.get("hidden_dim", 128)
        num_classes = len(label_map)
        encoder = GraphEncoder(node_dim=19, edge_dim=7, hidden_dim=hidden_dim, model_type="edge_sage")
        encoder.load_state_dict(ckpt["encoder"])
        classifier = torch.nn.Linear(hidden_dim, num_classes)
        classifier.load_state_dict(ckpt["classifier"])

        class _LegacyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = encoder
                self.clf = classifier
            def forward(self, x, ei, edge_attr=None, batch=None):
                emb = self.enc(x, ei, edge_attr=edge_attr, batch=batch)
                return self.clf(emb)

        model = _LegacyModel()

    model.eval()
    return model, label_map


def evaluate(model, loader, label_map):
    inv = {v: k for k, v in label_map.items()}
    correct = 0
    total = 0
    top3_correct = 0
    per_class_tp = Counter()
    per_class_total = Counter()
    per_class_fp = Counter()
    confusion = defaultdict(Counter)  # confusion[true][pred] = count

    with torch.no_grad():
        for bd, bl in loader:
            emb_out = model(bd["x"], bd["edge_index"],
                            edge_attr=bd.get("edge_attr"), batch=bd["batch"])
            if emb_out.size(0) != bl.size(0):
                continue
            probs = F.softmax(emb_out, dim=1)
            preds = emb_out.argmax(dim=1)
            top3 = torch.topk(probs, k=min(3, probs.size(1)), dim=1).indices

            for pred, true, t3 in zip(preds.tolist(), bl.tolist(), top3.tolist()):
                per_class_total[true] += 1
                total += 1
                if pred == true:
                    correct += 1
                    per_class_tp[true] += 1
                else:
                    per_class_fp[pred] += 1
                if true in t3:
                    top3_correct += 1
                confusion[true][pred] += 1

    acc = correct / max(total, 1)
    top3_acc = top3_correct / max(total, 1)

    # Per-class metrics
    per_class = {}
    f1_scores = []
    for cls_idx in sorted(per_class_total.keys()):
        tp = per_class_tp[cls_idx]
        fp = per_class_fp[cls_idx]
        fn = per_class_total[cls_idx] - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(per_class_total[cls_idx], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[cls_idx] = {"name": inv.get(cls_idx, str(cls_idx)),
                               "tp": tp, "total": per_class_total[cls_idx],
                               "precision": precision, "recall": recall, "f1": f1}
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / max(len(f1_scores), 1)

    return {
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": confusion,
        "total": total,
    }


def print_report(results: dict, title: str = "Evaluation Report"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Overall Accuracy : {results['accuracy']:.1%}")
    print(f"  Top-3 Accuracy   : {results['top3_accuracy']:.1%}")
    print(f"  Macro F1         : {results['macro_f1']:.3f}")
    print(f"  Total samples    : {results['total']}")
    print(f"\n  Per-class Recall:")
    for cls_idx, m in sorted(results["per_class"].items()):
        bar = "█" * int(m["recall"] * 20)
        print(f"  {m['name']:16s} {m['tp']:3d}/{m['total']:3d} "
              f"rec={m['recall']:.0%}  prec={m['precision']:.0%}  f1={m['f1']:.2f}  {bar}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Graph2D V2 model.")
    parser.add_argument("--model", required=True, help="Path to model checkpoint.")
    parser.add_argument("--baseline", default=None, help="Optional baseline checkpoint for comparison.")
    parser.add_argument("--manifest", required=True, help="Cache manifest CSV.")
    parser.add_argument("--golden-val-manifest", default=None,
                        help="Fixed golden validation manifest CSV. When provided, "
                             "evaluation uses this set instead of random_split. "
                             "Recommended: data/manifests/golden_val_set.csv")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.golden_val_manifest:
        # Phase 2: Use fixed golden validation set (no random_split)
        val_ds = CachedGraphDataset(args.golden_val_manifest)
        logger.info("Using golden validation set: %s (%d samples)",
                     args.golden_val_manifest, len(val_ds))
    else:
        ds = CachedGraphDataset(args.manifest)
        val_size = int(args.val_split * len(ds))
        train_size = len(ds) - val_size
        _, val_ds = random_split(ds, [train_size, val_size],
                                 generator=torch.Generator().manual_seed(args.seed))
        logger.info("Using random_split val set: %d samples (seed=%d). "
                     "Consider --golden-val-manifest for stable evaluation.",
                     len(val_ds), args.seed)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)

    model, label_map = load_model(args.model)
    results = evaluate(model, val_loader, label_map)
    print_report(results, f"B4.4 Model: {Path(args.model).name}")

    if args.baseline:
        base_model, base_label_map = load_model(args.baseline)
        # Use same val split for fair comparison (same seed)
        base_results = evaluate(base_model, val_loader, base_label_map)
        print_report(base_results, f"Baseline: {Path(args.baseline).name}")

        # Delta table
        print("  DELTA (new - baseline):")
        print(f"  Accuracy : {results['accuracy'] - base_results['accuracy']:+.1%}")
        print(f"  Top-3    : {results['top3_accuracy'] - base_results['top3_accuracy']:+.1%}")
        print(f"  Macro F1 : {results['macro_f1'] - base_results['macro_f1']:+.3f}")


if __name__ == "__main__":
    main()
