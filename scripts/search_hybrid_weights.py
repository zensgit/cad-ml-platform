#!/usr/bin/env python3
"""Grid search for optimal HybridClassifier weights using Graph2D predictions.

Simulates two scenarios:
  A) filename_available=True  : filename conf=high + graph2d conf=model_prob
  B) filename_available=False : filename conf=0    + graph2d conf=model_prob

Finds the (fn_w, g2d_w) pair that maximises accuracy in both scenarios.

Usage:
    # Basic grid search
    python scripts/search_hybrid_weights.py \
        --model models/graph2d_finetuned_24class_v2.pth \
        --manifest data/graph_cache/cache_manifest.csv

    # With output report
    python scripts/search_hybrid_weights.py \
        --model models/graph2d_finetuned_24class_v2.pth \
        --manifest data/graph_cache/cache_manifest.csv \
        --output docs/design/B4_5_WEIGHT_SEARCH_RESULT.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.finetune_graph2d_from_pretrained import CachedGraphDataset, collate_finetune
from scripts.evaluate_graph2d_v2 import load_model
from torch.utils.data import DataLoader, random_split


def collect_predictions(model, loader, label_map):
    """Return list of (true_label, prob_vector) for each sample."""
    inv = {v: k for k, v in label_map.items()}
    records = []
    with torch.no_grad():
        for bd, bl in loader:
            out = model(bd["x"], bd["edge_index"],
                        edge_attr=bd.get("edge_attr"), batch=bd["batch"])
            if out.size(0) != bl.size(0):
                continue
            probs = F.softmax(out, dim=1)
            for p, t in zip(probs, bl.tolist()):
                records.append((inv.get(t, str(t)), p.tolist()))
    return records, inv


def eval_weights(records, inv, fn_w, g2d_w, filename_available: bool):
    """Simulate hybrid prediction with given weights.

    When filename_available=True:  filename gives perfect signal with conf=1.0
    When filename_available=False: filename gives 0 signal (no-name scenario)
    """
    correct = 0
    total = 0
    label_keys = list(inv.values())

    for true_label, g2d_probs in records:
        # Graph2D contribution
        g2d_scores = {label_keys[i]: g2d_probs[i] * g2d_w for i in range(len(g2d_probs))}

        if filename_available:
            # Perfect filename: full weight on true class
            fn_scores = {label_keys[i]: (1.0 if label_keys[i] == true_label else 0.0) * fn_w
                         for i in range(len(label_keys))}
        else:
            fn_scores = {k: 0.0 for k in label_keys}

        # Fuse
        fused = {k: g2d_scores.get(k, 0.0) + fn_scores.get(k, 0.0) for k in label_keys}
        pred = max(fused, key=fused.get)
        if pred == true_label:
            correct += 1
        total += 1

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Grid search hybrid weights.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Write MD report to this path.")
    args = parser.parse_args()

    ds = CachedGraphDataset(args.manifest)
    val_size = int(args.val_split * len(ds))
    _, val_ds = random_split(ds, [len(ds) - val_size, val_size],
                             generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)

    model, label_map = load_model(args.model)
    inv = {v: k for k, v in label_map.items()}

    print("Collecting Graph2D predictions...", flush=True)
    records, inv = collect_predictions(model, val_loader, label_map)
    print(f"Collected {len(records)} predictions.", flush=True)

    # Baseline: graph2d only
    g2d_only_acc = eval_weights(records, inv, fn_w=0.0, g2d_w=1.0, filename_available=False)
    print(f"\nGraph2D only (no filename): {g2d_only_acc:.1%}")

    # Grid search
    fn_weights  = [0.3, 0.4, 0.45, 0.5, 0.6, 0.7]
    g2d_weights = [0.3, 0.4, 0.45, 0.5, 0.6]

    best_with_name   = (0, 0, 0)
    best_no_name     = (0, 0, 0)
    best_combined    = (0, 0, 0)

    print(f"\n{'fn_w':>6} {'g2d_w':>6} {'with_name':>10} {'no_name':>10} {'combined':>10}")
    print("-" * 50)

    results = []
    for fn_w, g2d_w in product(fn_weights, g2d_weights):
        acc_with = eval_weights(records, inv, fn_w, g2d_w, filename_available=True)
        acc_no   = eval_weights(records, inv, fn_w, g2d_w, filename_available=False)
        combined = 0.5 * acc_with + 0.5 * acc_no   # equal weighting of scenarios
        results.append((fn_w, g2d_w, acc_with, acc_no, combined))

        if acc_with > best_with_name[0]:
            best_with_name = (acc_with, fn_w, g2d_w)
        if acc_no > best_no_name[0]:
            best_no_name = (acc_no, fn_w, g2d_w)
        if combined > best_combined[0]:
            best_combined = (combined, fn_w, g2d_w)

        print(f"{fn_w:>6.2f} {g2d_w:>6.2f} {acc_with:>10.1%} {acc_no:>10.1%} {combined:>10.1%}")

    print("\n" + "=" * 50)
    print(f"Best for WITH filename : acc={best_with_name[0]:.1%}  fn={best_with_name[1]}  g2d={best_with_name[2]}")
    print(f"Best for NO filename   : acc={best_no_name[0]:.1%}  fn={best_no_name[1]}  g2d={best_no_name[2]}")
    print(f"Best combined (50/50)  : acc={best_combined[0]:.1%}  fn={best_combined[1]}  g2d={best_combined[2]}")

    if args.output:
        _write_report(args.output, results, best_with_name, best_no_name, best_combined,
                      g2d_only_acc, args.model)
        print(f"\nReport written to {args.output}")


def _write_report(path, results, best_with, best_no, best_combined, g2d_only, model_path):
    lines = [
        "# B4.5 Hybrid 权重搜索结果",
        "",
        f"**模型**: `{model_path}`  ",
        f"**日期**: 2026-04-14  ",
        "",
        "## 搜索结果",
        "",
        f"| fn_w | g2d_w | with_name | no_name | combined |",
        f"|------|-------|-----------|---------|---------|",
    ]
    for fn_w, g2d_w, acc_w, acc_n, comb in sorted(results, key=lambda x: -x[4])[:15]:
        lines.append(f"| {fn_w:.2f} | {g2d_w:.2f} | {acc_w:.1%} | {acc_n:.1%} | {comb:.1%} |")

    lines += [
        "",
        "## 最优配置",
        "",
        f"- **有文件名最优**: fn_w={best_with[1]}, g2d_w={best_with[2]} → acc={best_with[0]:.1%}",
        f"- **无文件名最优**: fn_w={best_no[1]}, g2d_w={best_no[2]} → acc={best_no[0]:.1%}",
        f"- **综合最优（推荐）**: fn_w={best_combined[1]}, g2d_w={best_combined[2]} → acc={best_combined[0]:.1%}",
        f"- Graph2D 单独（无文件名基线）: {g2d_only:.1%}",
        "",
        "## 建议更新",
        "",
        "```python",
        f"# src/ml/hybrid_config.py",
        f"FilenameClassifierConfig.fusion_weight = {best_combined[1]}",
        f"Graph2DConfig.fusion_weight = {best_combined[2]}",
        f"Graph2DConfig.enabled = True",
        "```",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
