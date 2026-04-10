"""Ensemble voting for Graph2D models.

Loads multiple trained Graph2D checkpoints and evaluates them individually
and as an ensemble (majority voting) on the validation set.

Usage:
    python scripts/ensemble_graph2d.py \
        --manifest data/training_merged_v2/manifest_4class.csv \
        --dxf-dir data/training_merged_v2/by_class \
        --models models/graph2d_4class_sage_v2.pth models/graph2d_4class_gcn_focal.pth models/graph2d_4class_sage.pth
"""

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def load_val_data(manifest_path: str, dxf_dir: str, val_ratio: float = 0.15, seed: int = 42):
    """Load and split data, return validation set graphs."""
    try:
        from src.ml.train.dataset_2d import DXFGraphDataset
    except ImportError:
        logger.error("Cannot import DXFGraphDataset")
        return None, None

    rows = list(csv.DictReader(open(manifest_path)))
    labels = sorted(set(r["label_cn"] for r in rows))
    label_to_idx = {l: i for i, l in enumerate(labels)}

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(rows))
    val_size = int(len(rows) * val_ratio)
    val_indices = set(indices[:val_size])

    val_files = []
    val_labels = []
    for i, r in enumerate(rows):
        if i in val_indices:
            fpath = Path(dxf_dir) / r["label_cn"] / r["file_name"]
            if not fpath.exists():
                fpath = Path(r.get("full_path", ""))
            val_files.append(str(fpath))
            val_labels.append(label_to_idx[r["label_cn"]])

    return val_files, val_labels, labels


def predict_single_model(model_path: str, val_files: list, num_classes: int) -> List[int]:
    """Run inference with a single model on all val files."""
    try:
        import torch
        from src.ml.train.model_2d import SimpleGraphClassifier, EdgeGraphSageClassifier
        from src.ml.train.dataset_2d import DXFGraphDataset
    except ImportError:
        logger.error("Cannot import torch/models")
        return []

    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Detect model type from checkpoint
    state = checkpoint if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint else checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    node_dim = config.get("node_dim", 19)
    edge_dim = config.get("edge_dim", 7)
    hidden_dim = config.get("hidden_dim", 64)

    # Try EdgeSAGE first, fall back to GCN
    try:
        model = EdgeGraphSageClassifier(
            node_input_dim=node_dim, edge_input_dim=edge_dim,
            hidden_dim=hidden_dim, num_classes=num_classes
        )
        model.load_state_dict(state, strict=False)
    except Exception:
        try:
            model = SimpleGraphClassifier(
                node_input_dim=node_dim, hidden_dim=hidden_dim, num_classes=num_classes
            )
            model.load_state_dict(state, strict=False)
        except Exception as e:
            logger.warning("Cannot load model %s: %s", model_path, e)
            return [-1] * len(val_files)

    model.eval()
    predictions = []

    dataset = DXFGraphDataset.__new__(DXFGraphDataset)
    dataset.max_nodes = config.get("dxf_max_nodes", 300)
    dataset.sampling_strategy = config.get("dxf_sampling_strategy", "importance")
    dataset.sampling_seed = config.get("dxf_sampling_seed", 42)

    for fpath in val_files:
        try:
            graph = dataset._build_graph_from_dxf(fpath)
            if graph is None:
                predictions.append(-1)
                continue
            with torch.no_grad():
                x = graph["x"].unsqueeze(0) if graph["x"].dim() == 2 else graph["x"]
                logits = model(x)
                pred = logits.argmax(dim=-1).item()
                predictions.append(pred)
        except Exception:
            predictions.append(-1)

    return predictions


def ensemble_vote(all_predictions: List[List[int]], num_samples: int) -> List[int]:
    """Majority voting across models."""
    ensemble_preds = []
    for i in range(num_samples):
        votes = [preds[i] for preds in all_predictions if preds[i] >= 0]
        if votes:
            ensemble_preds.append(Counter(votes).most_common(1)[0][0])
        else:
            ensemble_preds.append(-1)
    return ensemble_preds


def accuracy(predictions: List[int], labels: List[int]) -> float:
    """Compute accuracy, ignoring -1 predictions."""
    correct = sum(1 for p, l in zip(predictions, labels) if p == l and p >= 0)
    valid = sum(1 for p in predictions if p >= 0)
    return correct / valid if valid > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Ensemble Graph2D models")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dxf-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    print("Loading validation data...")
    val_files, val_labels, label_names = load_val_data(args.manifest, args.dxf_dir)
    if not val_files:
        print("No validation data loaded.")
        return

    print(f"Validation set: {len(val_files)} files, {len(label_names)} classes")
    num_classes = len(label_names)

    all_predictions = []
    for model_path in args.models:
        print(f"\nRunning {Path(model_path).name}...")
        preds = predict_single_model(model_path, val_files, num_classes)
        acc = accuracy(preds, val_labels)
        print(f"  Individual accuracy: {acc:.1%}")
        all_predictions.append(preds)

    print("\n=== Ensemble (majority vote) ===")
    ensemble_preds = ensemble_vote(all_predictions, len(val_files))
    ens_acc = accuracy(ensemble_preds, val_labels)
    print(f"  Ensemble accuracy: {ens_acc:.1%}")

    print(f"\n{'Model':<45} {'Accuracy':>10}")
    print("-" * 57)
    for model_path, preds in zip(args.models, all_predictions):
        acc = accuracy(preds, val_labels)
        print(f"{Path(model_path).name:<45} {acc:>9.1%}")
    print("-" * 57)
    print(f"{'ENSEMBLE (majority vote)':<45} {ens_acc:>9.1%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
