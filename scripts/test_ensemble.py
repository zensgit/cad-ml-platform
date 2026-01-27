#!/usr/bin/env python3
"""Test ensemble classifier on validation set."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import random
from collections import Counter

import torch

from src.ml.train.dataset_2d import DXFManifestDataset
from src.ml.vision_2d import Graph2DClassifier, EnsembleGraph2DClassifier


def main():
    # Load dataset
    manifest = "reports/experiments/20260120/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260120.csv"
    dxf_dir = "/Users/huazhou/Downloads/4000例CAD及三维机械零件练习图纸/机械CAD图纸_dxf"

    dataset = DXFManifestDataset(manifest, dxf_dir, return_edge_attr=True)
    print(f"Dataset size: {len(dataset)}")

    # Split same as training
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Stratified split
    labels = [dataset.samples[idx]["label_id"] for idx in indices]
    label_to_indices = {}
    for idx, label in zip(indices, labels):
        label_to_indices.setdefault(label, []).append(idx)

    val_idx = []
    for label, idxs in label_to_indices.items():
        train_count = max(1, int(len(idxs) * 0.8))
        if len(idxs) > 1 and train_count == len(idxs):
            train_count -= 1
        val_idx.extend(idxs[train_count:])

    print(f"Validation set size: {len(val_idx)}")

    # Load classifiers
    v3 = Graph2DClassifier(model_path="models/graph2d_edge_sage_v3.pth")
    v4 = Graph2DClassifier(model_path="models/graph2d_edge_sage_v4_best.pth")
    ensemble_soft = EnsembleGraph2DClassifier(voting="soft")
    ensemble_hard = EnsembleGraph2DClassifier(voting="hard")

    print(f"\nModels loaded:")
    print(f"  v3: {v3._loaded}")
    print(f"  v4: {v4._loaded}")
    print(f"  ensemble_soft: {ensemble_soft._loaded} ({len(ensemble_soft.classifiers)} models)")
    print(f"  ensemble_hard: {ensemble_hard._loaded} ({len(ensemble_hard.classifiers)} models)")

    # Evaluate
    results = {
        "v3": {"correct": 0, "total": 0},
        "v4": {"correct": 0, "total": 0},
        "ensemble_soft": {"correct": 0, "total": 0},
        "ensemble_hard": {"correct": 0, "total": 0},
    }

    label_map = dataset.get_label_map()
    idx_to_label = {v: k for k, v in label_map.items()}

    print("\nEvaluating on validation set...")
    for i, idx in enumerate(val_idx):
        sample = dataset.samples[idx]
        # Convert .dwg to .dxf filename
        file_name = sample["file_name"]
        if file_name.endswith(".dwg"):
            file_name = file_name[:-4] + ".dxf"
        dxf_path = Path(dxf_dir) / file_name
        if not dxf_path.exists():
            continue

        try:
            dxf_bytes = dxf_path.read_bytes()
        except Exception:
            continue

        true_label = idx_to_label.get(sample["label_id"])

        # Test each classifier
        for name, clf in [("v3", v3), ("v4", v4), ("ensemble_soft", ensemble_soft), ("ensemble_hard", ensemble_hard)]:
            pred = clf.predict_from_bytes(dxf_bytes, sample["file_name"])
            if pred.get("status") == "ok":
                results[name]["total"] += 1
                if pred.get("label") == true_label:
                    results[name]["correct"] += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(val_idx)}")

    # Print results
    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    for name, r in results.items():
        acc = r["correct"] / max(1, r["total"]) * 100
        print(f"{name:15s}: {r['correct']:3d}/{r['total']:3d} = {acc:.1f}%")


if __name__ == "__main__":
    main()
