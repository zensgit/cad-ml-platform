#!/usr/bin/env python3
"""Evaluate heterogeneous ensemble: GNN + StatMLP + TextMLP (B6.0c).

Combines three fundamentally different model types for true diversity:
  1. GNN (v4): Graph topology features → 91.9%
  2. StatMLP: Hand-crafted statistical features → ~X%
  3. TextMLP: TF-IDF text features → 73.7%

Usage:
    python scripts/evaluate_hetero_ensemble.py \
        --gnn-model models/graph2d_finetuned_24class_v4.pth \
        --stat-model models/stat_mlp_24class.pth \
        --text-model models/text_classifier_tfidf.pth \
        --manifest data/graph_cache/cache_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Heterogeneous ensemble evaluation")
    parser.add_argument("--gnn-model", default="models/graph2d_finetuned_24class_v4.pth")
    parser.add_argument("--stat-model", default="models/stat_mlp_24class.pth")
    parser.add_argument("--text-model", default="models/text_classifier_tfidf.pth")
    parser.add_argument("--manifest", default="data/graph_cache/cache_manifest.csv")
    parser.add_argument("--gnn-weight", type=float, default=0.70)
    parser.add_argument("--stat-weight", type=float, default=0.15)
    parser.add_argument("--text-weight", type=float, default=0.15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    from scripts.evaluate_graph2d_v2 import load_model
    from scripts.finetune_graph2d_from_pretrained import CachedGraphDataset, collate_finetune
    from scripts.train_stat_mlp import StatMLP, extract_stat_features, STAT_FEAT_DIM
    from scripts.train_text_classifier_ml import TextMLP, SimpleVectorizer
    from src.ml.text_extractor import extract_text_from_path
    from torch.utils.data import DataLoader

    # Load GNN
    gnn_model, gnn_lm = load_model(args.gnn_model)
    gnn_model.eval()
    inv_gnn = {v: k for k, v in gnn_lm.items()}
    label_keys = [inv_gnn[i] for i in range(len(inv_gnn))]
    logger.info("GNN loaded: %d classes", len(gnn_lm))

    # Load StatMLP
    stat_model = None
    stat_ckpt = None
    if Path(args.stat_model).exists():
        stat_ckpt = torch.load(args.stat_model, map_location="cpu", weights_only=False)
        if stat_ckpt.get("best_val_acc", 0) > 0.01:  # Skip if barely trained
            stat_model = StatMLP(
                input_dim=stat_ckpt.get("input_dim", STAT_FEAT_DIM),
                num_classes=stat_ckpt.get("num_classes", 24),
            )
            stat_model.load_state_dict(stat_ckpt["model_state"])
            stat_model.eval()
            logger.info("StatMLP loaded: val_acc=%.1f%%", stat_ckpt["best_val_acc"] * 100)
        else:
            logger.info("StatMLP skipped (val_acc=%.1f%%, too low)", stat_ckpt.get("best_val_acc", 0) * 100)

    # Load TextMLP
    text_model = None
    text_vectorizer = None
    if Path(args.text_model).exists():
        text_ckpt = torch.load(args.text_model, map_location="cpu", weights_only=False)
        text_model = TextMLP(
            input_dim=text_ckpt.get("input_dim", 500),
            num_classes=text_ckpt.get("num_classes", 24),
        )
        text_model.load_state_dict(text_ckpt["model_state"])
        text_model.eval()
        text_vectorizer = SimpleVectorizer(max_features=text_ckpt.get("max_features", 500))
        text_vectorizer.vocab = text_ckpt["vectorizer_vocab"]
        text_vectorizer.idf = text_ckpt["vectorizer_idf"]
        logger.info("TextMLP loaded: val_acc=%.1f%%", text_ckpt["best_val_acc"] * 100)

    # Load data
    dataset = CachedGraphDataset(args.manifest)
    ds_inv = {v: k for k, v in dataset.label_map.items()}
    rows = list(csv.DictReader(open(args.manifest)))
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_finetune)

    # Evaluate
    correct_gnn = correct_ens = total = 0
    sample_idx = 0

    stat_mean = stat_ckpt.get("feat_mean") if stat_ckpt else None
    stat_std = stat_ckpt.get("feat_std") if stat_ckpt else None

    with torch.no_grad():
        for batched, labels in loader:
            x = batched["x"]; ei = batched["edge_index"]
            ea = batched.get("edge_attr"); b = batched["batch"]

            # GNN predictions
            gnn_logits = gnn_model(x, ei, ea, b)
            gnn_probs = F.softmax(gnn_logits, dim=1)

            for i, l in enumerate(labels):
                true_cls = ds_inv.get(l.item(), "?")

                # GNN
                gnn_pred = inv_gnn.get(gnn_probs[i].argmax().item(), "?")
                if gnn_pred == true_cls:
                    correct_gnn += 1

                # Build ensemble score
                scores = defaultdict(float)
                for j, cls in enumerate(label_keys):
                    scores[cls] += float(gnn_probs[i][j]) * args.gnn_weight

                # StatMLP (if available and trained)
                if stat_model and stat_mean is not None:
                    # Extract features for this sample
                    cache_path = rows[sample_idx]["cache_path"] if sample_idx < len(rows) else ""
                    if cache_path:
                        try:
                            data = torch.load(cache_path, map_location="cpu", weights_only=True)
                            feats = extract_stat_features(data).unsqueeze(0)
                            feats = (feats - stat_mean) / stat_std.clamp(min=1e-6)
                            stat_probs = F.softmax(stat_model(feats), dim=1)[0]
                            stat_lm = stat_ckpt["label_map"]
                            stat_inv = {v: k for k, v in stat_lm.items()}
                            for j in range(len(stat_inv)):
                                cls = stat_inv.get(j, "?")
                                if cls in scores:
                                    scores[cls] += float(stat_probs[j]) * args.stat_weight
                        except Exception:
                            pass

                # TextMLP (if available)
                if text_model and text_vectorizer:
                    fp = rows[sample_idx]["file_path"] if sample_idx < len(rows) else ""
                    if fp:
                        try:
                            text = extract_text_from_path(fp)
                            if text and len(text.strip()) >= 4:
                                vec = text_vectorizer.transform(text).unsqueeze(0)
                                text_probs = F.softmax(text_model(vec), dim=1)[0]
                                text_lm = text_ckpt["label_map"]
                                text_inv = {v: k for k, v in text_lm.items()}
                                for j in range(len(text_inv)):
                                    cls = text_inv.get(j, "?")
                                    if cls in scores:
                                        scores[cls] += float(text_probs[j]) * args.text_weight
                        except Exception:
                            pass

                ens_pred = max(scores, key=scores.get) if scores else "?"
                if ens_pred == true_cls:
                    correct_ens += 1

                total += 1
                sample_idx += 1

        if (sample_idx) % 500 == 0:
            logger.info("  [%d/%d]", sample_idx, len(rows))

    print(f"\n{'='*60}")
    print(f"Heterogeneous Ensemble Evaluation ({total} samples)")
    print(f"{'='*60}")
    print(f"  GNN only:  {correct_gnn}/{total} = {correct_gnn/total*100:.1f}%")
    print(f"  Ensemble:  {correct_ens}/{total} = {correct_ens/total*100:.1f}%")
    print(f"  Delta:     {(correct_ens-correct_gnn)/total*100:+.1f}pp")
    print(f"\n  Weights: GNN={args.gnn_weight} Stat={args.stat_weight} Text={args.text_weight}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
