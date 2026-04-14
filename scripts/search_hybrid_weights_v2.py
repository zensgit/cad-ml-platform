#!/usr/bin/env python3
"""3-way grid search: filename + Graph2D + text-content weights.

Evaluates all (fn_w, g2d_w, txt_w) combinations on the validation set.
Simulates four scenarios:
  A) filename_available=True,  text_available=True   → realistic best case
  B) filename_available=False, text_available=True   → no-name with text
  C) filename_available=False, text_available=False  → pure graph2d
  D) filename_available=True,  text_available=False  → name only (baseline)

Usage:
    python scripts/search_hybrid_weights_v2.py \
        --model models/graph2d_finetuned_24class_v3.pth \
        --manifest data/graph_cache/cache_manifest.csv \
        --output docs/design/B5_1_WEIGHT_SEARCH_RESULT.md
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.finetune_graph2d_from_pretrained import CachedGraphDataset, collate_finetune
from scripts.evaluate_graph2d_v2 import load_model
from src.ml.text_extractor import extract_text_from_path
from src.ml.text_classifier import TextContentClassifier
from torch.utils.data import DataLoader, random_split


def collect_predictions(model, loader, label_map, val_indices, orig_rows):
    """Collect (true_label, g2d_probs, text_probs) for each val sample."""
    inv = {v: k for k, v in label_map.items()}
    label_keys = [inv[i] for i in range(len(inv))]

    clf = TextContentClassifier()
    records = []

    with torch.no_grad():
        sample_idx = 0
        for bd, bl in loader:
            out = model(bd["x"], bd["edge_index"],
                        edge_attr=bd.get("edge_attr"), batch=bd["batch"])
            if out.size(0) != bl.size(0):
                continue
            probs = F.softmax(out, dim=1)
            for p, t in zip(probs, bl.tolist()):
                true_label = inv.get(t, str(t))

                # Get text for this sample
                row_idx = val_indices[sample_idx] if sample_idx < len(val_indices) else 0
                file_path = orig_rows[row_idx].get("file_path", "")
                text = extract_text_from_path(file_path) if file_path else ""
                text_probs = clf.predict_probs(text)

                records.append({
                    "true": true_label,
                    "g2d": p.tolist(),
                    "text": text_probs,
                })
                sample_idx += 1

    return records, label_keys


def eval_scenario(records, label_keys, fn_w, g2d_w, txt_w,
                  fn_available: bool, txt_available: bool) -> float:
    correct = total = 0
    for rec in records:
        true_label = rec["true"]
        g2d = rec["g2d"]
        text_probs = rec["text"] if txt_available else {}

        # Build fused scores
        scores = {}
        for i, lk in enumerate(label_keys):
            s = g2d[i] * g2d_w
            if fn_available:
                fn_score = 1.0 if lk == true_label else 0.0
                s += fn_score * fn_w
            if txt_available and text_probs:
                s += text_probs.get(lk, 0.0) * txt_w
            scores[lk] = s

        pred = max(scores, key=scores.get)
        if pred == true_label:
            correct += 1
        total += 1

    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="3-way hybrid weight search.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import csv
    with open(args.manifest, encoding="utf-8") as f:
        orig_rows = list(csv.DictReader(f))

    ds = CachedGraphDataset(args.manifest)
    val_size = int(args.val_split * len(ds))
    _, val_ds = random_split(ds, [len(ds) - val_size, val_size],
                             generator=torch.Generator().manual_seed(args.seed))
    val_indices = list(val_ds.indices)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            collate_fn=collate_finetune)

    model, label_map = load_model(args.model)
    print(f"Model loaded: {len(label_map)} classes", flush=True)

    print("Collecting Graph2D predictions + text features...", flush=True)
    records, label_keys = collect_predictions(model, val_loader, label_map,
                                              val_indices, orig_rows)
    n_with_text = sum(1 for r in records if r["text"])
    print(f"Collected {len(records)} predictions, "
          f"{n_with_text} with text ({n_with_text/len(records):.1%})", flush=True)

    # Baselines
    acc_g2d_only = eval_scenario(records, label_keys, 0, 1, 0, False, False)
    acc_g2d_text = eval_scenario(records, label_keys, 0, 0.7, 0.3, False, True)
    print(f"\nBaselines:")
    print(f"  Graph2D only (C):       {acc_g2d_only:.1%}")
    print(f"  Graph2D + text (B):     {acc_g2d_text:.1%}")

    # Grid search
    fn_weights  = [0.35, 0.40, 0.45, 0.50]
    g2d_weights = [0.35, 0.40, 0.45, 0.50]
    txt_weights = [0.10, 0.15, 0.20, 0.25]

    results = []
    best_B = best_A = best_combined = (0, 0, 0, 0)

    print(f"\n{'fn_w':>5} {'g2d_w':>6} {'txt_w':>6} "
          f"{'A(fn+txt)':>10} {'B(txt)':>8} {'C(g2d)':>8} {'D(fn)':>7} {'avg':>7}")
    print("-" * 60)

    for fn_w, g2d_w, txt_w in product(fn_weights, g2d_weights, txt_weights):
        acc_A = eval_scenario(records, label_keys, fn_w, g2d_w, txt_w, True,  True)
        acc_B = eval_scenario(records, label_keys, fn_w, g2d_w, txt_w, False, True)
        acc_C = eval_scenario(records, label_keys, fn_w, g2d_w, txt_w, False, False)
        acc_D = eval_scenario(records, label_keys, fn_w, g2d_w, txt_w, True,  False)
        avg = 0.25 * acc_A + 0.50 * acc_B + 0.15 * acc_C + 0.10 * acc_D
        results.append((fn_w, g2d_w, txt_w, acc_A, acc_B, acc_C, acc_D, avg))

        if acc_B > best_B[0]:
            best_B = (acc_B, fn_w, g2d_w, txt_w)
        if acc_A > best_A[0]:
            best_A = (acc_A, fn_w, g2d_w, txt_w)
        if avg > best_combined[0]:
            best_combined = (avg, fn_w, g2d_w, txt_w)

        print(f"{fn_w:>5.2f} {g2d_w:>6.2f} {txt_w:>6.2f} "
              f"{acc_A:>10.1%} {acc_B:>8.1%} {acc_C:>8.1%} {acc_D:>7.1%} {avg:>7.1%}")

    print("\n" + "=" * 60)
    print(f"Best B (no-name+text): acc={best_B[0]:.1%}  fn={best_B[1]} g2d={best_B[2]} txt={best_B[3]}")
    print(f"Best A (fn+txt)      : acc={best_A[0]:.1%}  fn={best_A[1]} g2d={best_A[2]} txt={best_A[3]}")
    print(f"Best combined        : avg={best_combined[0]:.1%}  fn={best_combined[1]} g2d={best_combined[2]} txt={best_combined[3]}")

    if args.output:
        _write_report(args.output, results, best_A, best_B, best_combined,
                      acc_g2d_only, acc_g2d_text, n_with_text, len(records))
        print(f"\nReport written to {args.output}")


def _write_report(path, results, best_A, best_B, best_combined,
                  g2d_only, g2d_text, n_text, n_total):
    top15 = sorted(results, key=lambda x: -x[7])[:15]
    lines = [
        "# B5.1 三路融合权重搜索结果",
        "",
        f"**文字覆盖率**: {n_text}/{n_total} = {n_text/max(n_total,1):.1%}  ",
        f"**Graph2D 基线（无名无文字）**: {g2d_only:.1%}  ",
        f"**Graph2D + 文字（无名有文字）**: {g2d_text:.1%}  ",
        "",
        "## Top-15 组合（按综合得分）",
        "",
        "| fn_w | g2d_w | txt_w | A(有名有文字) | B(无名有文字) | C(纯图形) | avg |",
        "|------|-------|-------|------------|------------|---------|-----|",
    ]
    for r in top15:
        fn_w, g2d_w, txt_w, aA, aB, aC, aD, avg = r
        lines.append(
            f"| {fn_w:.2f} | {g2d_w:.2f} | {txt_w:.2f} | {aA:.1%} | {aB:.1%} | {aC:.1%} | {avg:.1%} |"
        )
    lines += [
        "",
        "## 最优配置",
        "",
        f"- **场景B最优（无名+文字）**: fn={best_B[1]}, g2d={best_B[2]}, txt={best_B[3]} → {best_B[0]:.1%}",
        f"- **场景A最优（有名+文字）**: fn={best_A[1]}, g2d={best_A[2]}, txt={best_A[3]} → {best_A[0]:.1%}",
        f"- **综合最优（推荐）**: fn={best_combined[1]}, g2d={best_combined[2]}, txt={best_combined[3]} → avg={best_combined[0]:.1%}",
        "",
        "## 建议配置",
        "",
        "```python",
        "# src/ml/hybrid_config.py",
        f"FilenameClassifierConfig.fusion_weight = {best_combined[1]}",
        f"Graph2DConfig.fusion_weight = {best_combined[2]}",
        f"# TextContentConfig.fusion_weight = {best_combined[3]}  # B5.1 新增",
        "```",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
