#!/usr/bin/env python3
"""Knowledge distillation: heterogeneous ensemble → single GNN student (B6.2b).

Uses the GNN+StatMLP+TextMLP ensemble (95.8%) as a teacher to train
a single GraphEncoderV2 student that inherits the ensemble's knowledge
without the runtime cost of multiple models.

Usage:
    python scripts/distill_ensemble.py \
        --gnn-model models/graph2d_finetuned_24class_v4.pth \
        --stat-model models/stat_mlp_24class.pth \
        --text-model models/text_classifier_tfidf.pth \
        --manifest data/graph_cache/cache_manifest.csv \
        --output models/graph2d_distilled_v5.pth \
        --epochs 50 --temperature 3.0 --alpha 0.7
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_graph2d_v2 import load_model
from scripts.finetune_graph2d_from_pretrained import CachedGraphDataset, collate_finetune
from scripts.train_stat_mlp import StatMLP, extract_stat_features, STAT_FEAT_DIM
from scripts.train_text_classifier_ml import TextMLP, SimpleVectorizer
from src.ml.text_extractor import extract_text_from_path
from src.ml.train.model_2d import GraphEncoderV2WithHead
from torch.utils.data import DataLoader, random_split

logger = logging.getLogger(__name__)


def build_teacher_soft_labels(
    gnn_model, stat_model, stat_ckpt, text_model, text_vectorizer, text_label_map,
    dataset, manifest_rows, label_keys,
    gnn_weight=0.60, stat_weight=0.25, text_weight=0.15,
):
    """Pre-compute ensemble soft labels for the full dataset."""
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_finetune)
    ds_inv = {v: k for k, v in dataset.label_map.items()}

    stat_inv = {v: k for k, v in stat_ckpt["label_map"].items()}
    s_mean = stat_ckpt.get("feat_mean")
    s_std = stat_ckpt.get("feat_std")

    txt_inv = {v: k for k, v in text_label_map.items()}

    soft_labels = []  # List of (num_classes,) tensors
    num_classes = len(label_keys)

    sample_idx = 0
    with torch.no_grad():
        for batched, labels in loader:
            x = batched["x"]; ei = batched["edge_index"]
            ea = batched.get("edge_attr"); b = batched["batch"]
            gnn_probs = F.softmax(gnn_model(x, ei, ea, b), dim=1)

            for i in range(len(labels)):
                # GNN contribution
                teacher = torch.zeros(num_classes)
                for j, cls in enumerate(label_keys):
                    teacher[j] += float(gnn_probs[i][j]) * gnn_weight

                # StatMLP contribution
                cache_path = manifest_rows[sample_idx]["cache_path"] if sample_idx < len(manifest_rows) else ""
                if cache_path and stat_model is not None:
                    try:
                        data = torch.load(cache_path, map_location="cpu", weights_only=True)
                        feats = extract_stat_features(data).unsqueeze(0)
                        if s_mean is not None and s_std is not None:
                            feats = (feats - s_mean) / s_std.clamp(min=1e-6)
                        sp = F.softmax(stat_model(feats), dim=1)[0]
                        for j_s in range(len(stat_inv)):
                            cls_s = stat_inv.get(j_s, "?")
                            if cls_s in label_keys:
                                teacher[label_keys.index(cls_s)] += float(sp[j_s]) * stat_weight
                    except Exception:
                        pass

                # TextMLP contribution
                fp = manifest_rows[sample_idx]["file_path"] if sample_idx < len(manifest_rows) else ""
                if fp and text_model is not None:
                    try:
                        text = extract_text_from_path(fp)
                        if text and len(text.strip()) >= 4:
                            vec = text_vectorizer.transform(text).unsqueeze(0)
                            tp = F.softmax(text_model(vec), dim=1)[0]
                            for j_t in range(len(txt_inv)):
                                cls_t = txt_inv.get(j_t, "?")
                                if cls_t in label_keys:
                                    teacher[label_keys.index(cls_t)] += float(tp[j_t]) * text_weight
                    except Exception:
                        pass

                # Normalize to probability distribution
                teacher = teacher / teacher.sum().clamp(min=1e-8)
                soft_labels.append(teacher)
                sample_idx += 1

    logger.info("Generated %d soft labels", len(soft_labels))
    return soft_labels


class DistillDataset(torch.utils.data.Dataset):
    """Wraps CachedGraphDataset with pre-computed soft labels."""
    def __init__(self, base_dataset, soft_labels):
        self.base = base_dataset
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        graph, hard_label = self.base[idx]
        soft_label = self.soft_labels[idx]
        return graph, hard_label, soft_label


def collate_distill(batch):
    graphs, hard_labels, soft_labels = zip(*batch)
    # Reuse existing collate for graphs + hard labels
    batched, hard_tensor = collate_finetune(list(zip(graphs, hard_labels)))
    soft_tensor = torch.stack(soft_labels)
    return batched, hard_tensor, soft_tensor


def distillation_loss(student_logits, hard_labels, soft_labels, temperature, alpha):
    """Combined distillation loss: α × CE(hard) + (1-α) × KL(soft)."""
    # Hard label cross-entropy
    ce_loss = F.cross_entropy(student_logits, hard_labels)

    # Soft label KL divergence
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(soft_labels / temperature, dim=1) if soft_labels.dim() > 1 else soft_labels
    # For pre-computed soft labels (already probabilities), just use them directly
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    return alpha * ce_loss + (1 - alpha) * kl_loss


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation (B6.2b)")
    parser.add_argument("--gnn-model", default="models/graph2d_finetuned_24class_v4.pth")
    parser.add_argument("--stat-model", default="models/stat_mlp_24class.pth")
    parser.add_argument("--text-model", default="models/text_classifier_tfidf.pth")
    parser.add_argument("--manifest", default="data/graph_cache/cache_manifest.csv")
    parser.add_argument("--output", default="models/graph2d_distilled_v5.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Weight for hard label CE (1-alpha = soft label KL)")
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    # Load teacher models
    logger.info("Loading teacher ensemble...")
    gnn_teacher, gnn_lm = load_model(args.gnn_model)
    gnn_teacher.eval()
    inv_gnn = {v: k for k, v in gnn_lm.items()}
    label_keys = [inv_gnn[i] for i in range(len(inv_gnn))]

    stat_model = None
    stat_ckpt = {}
    if Path(args.stat_model).exists():
        stat_ckpt = torch.load(args.stat_model, map_location="cpu", weights_only=False)
        if stat_ckpt.get("best_val_acc", 0) > 0.5:
            stat_model = StatMLP(input_dim=stat_ckpt["input_dim"], num_classes=stat_ckpt["num_classes"])
            stat_model.load_state_dict(stat_ckpt["model_state"])
            stat_model.eval()

    text_model = None
    text_vectorizer = None
    text_lm = {}
    if Path(args.text_model).exists():
        txt_ckpt = torch.load(args.text_model, map_location="cpu", weights_only=False)
        text_model = TextMLP(input_dim=txt_ckpt["input_dim"], num_classes=txt_ckpt["num_classes"])
        text_model.load_state_dict(txt_ckpt["model_state"])
        text_model.eval()
        text_vectorizer = SimpleVectorizer(max_features=txt_ckpt.get("max_features", 500))
        text_vectorizer.vocab = txt_ckpt["vectorizer_vocab"]
        text_vectorizer.idf = txt_ckpt["vectorizer_idf"]
        text_lm = txt_ckpt["label_map"]

    # Dataset
    dataset = CachedGraphDataset(args.manifest)
    manifest_rows = list(csv.DictReader(open(args.manifest)))

    # Pre-compute soft labels
    logger.info("Computing ensemble soft labels (this takes a few minutes)...")
    soft_labels = build_teacher_soft_labels(
        gnn_teacher, stat_model, stat_ckpt, text_model, text_vectorizer, text_lm,
        dataset, manifest_rows, label_keys,
    )

    # Train/val split
    distill_ds = DistillDataset(dataset, soft_labels)
    n = len(distill_ds)
    n_val = max(1, int(0.2 * n))
    train_ds, val_ds = random_split(distill_ds, [n - n_val, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_distill)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_distill)

    # Student model (fresh GraphEncoderV2, initialized from v4 checkpoint)
    logger.info("Initializing student from %s...", args.gnn_model)
    student, _ = load_model(args.gnn_model)
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        tc = tt = 0
        for batched, hard_labels, soft_labels_batch in train_loader:
            x = batched["x"]; ei = batched["edge_index"]
            ea = batched.get("edge_attr"); b = batched["batch"]

            optimizer.zero_grad()
            logits = student(x, ei, ea, b)

            # Align soft labels size with logits
            if soft_labels_batch.size(0) != logits.size(0):
                min_n = min(soft_labels_batch.size(0), logits.size(0))
                soft_labels_batch = soft_labels_batch[:min_n]
                logits = logits[:min_n]
                hard_labels = hard_labels[:min_n]

            loss = distillation_loss(logits, hard_labels, soft_labels_batch, args.temperature, args.alpha)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            tc += (preds == hard_labels).sum().item()
            tt += len(hard_labels)
        scheduler.step()

        # Val
        student.eval()
        vc = vt = 0
        with torch.no_grad():
            for batched, hard_labels, _ in val_loader:
                x = batched["x"]; ei = batched["edge_index"]
                ea = batched.get("edge_attr"); b = batched["batch"]
                logits = student(x, ei, ea, b)
                if logits.size(0) != hard_labels.size(0):
                    min_n = min(logits.size(0), hard_labels.size(0))
                    logits = logits[:min_n]; hard_labels = hard_labels[:min_n]
                preds = logits.argmax(dim=1)
                vc += (preds == hard_labels).sum().item()
                vt += len(hard_labels)

        train_acc = tc / max(tt, 1)
        val_acc = vc / max(vt, 1)
        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            improved = " ✓"
            torch.save({
                "arch": "GraphEncoderV2",
                "model_state": student.state_dict(),
                "label_map": gnn_lm,
                "best_val_acc": best_val_acc,
                "distilled_from": "gnn+stat+text ensemble",
                "temperature": args.temperature,
                "alpha": args.alpha,
            }, args.output)
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch <= 3 or improved:
            print(f"  {epoch:>3d}  train={train_acc:.3f}  val={val_acc:.3f}  loss={loss.item():.4f}{improved}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest val acc: {best_val_acc*100:.1f}%")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
