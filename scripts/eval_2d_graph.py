#!/usr/bin/env python3
"""Evaluate a trained 2D graph classifier checkpoint with per-class metrics."""

from __future__ import annotations

import argparse
import csv
import random
import sys
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


def _inverse_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    return {idx: name for name, idx in label_map.items()}


def _bucket_error(confidence: float, margin: float) -> str:
    if confidence < 0.4:
        return "low_confidence"
    if margin < 0.1:
        return "low_margin"
    if confidence >= 0.8:
        return "high_confidence"
    return "mid_confidence"


def _stratified_split(indices: List[int], labels: List[int], val_ratio: float):
    label_to_indices: Dict[int, List[int]] = {}
    for idx, label in zip(indices, labels):
        label_to_indices.setdefault(label, []).append(idx)

    val_idx: List[int] = []
    for label, idxs in label_to_indices.items():
        random.shuffle(idxs)
        val_count = max(1, int(len(idxs) * val_ratio))
        val_count = min(val_count, len(idxs))
        val_idx.extend(idxs[:val_count])

    if not val_idx:
        val_idx = indices[:1]

    return val_idx


def _collate(
    batch: List[Tuple[Dict[str, Any], Any]]
) -> Tuple[List[Any], List[Any], List[Any], List[Any], List[str]]:
    xs, edges, edge_attrs, labels, file_names = [], [], [], [], []
    for graph, label in batch:
        xs.append(graph["x"])
        edges.append(graph["edge_index"])
        edge_attrs.append(graph.get("edge_attr"))
        labels.append(label)
        file_names.append(str(graph.get("file_name", "")))
    return xs, edges, edge_attrs, labels, file_names


def main() -> int:
    if not _require_torch():
        return 1

    import torch
    from torch.utils.data import DataLoader, Subset

    from src.ml.train.dataset_2d import DXFManifestDataset, DXF_EDGE_DIM, DXF_NODE_DIM
    from src.ml.train.model_2d import EdgeGraphSageClassifier, SimpleGraphClassifier

    parser = argparse.ArgumentParser(description="Evaluate 2D DXF graph classifier.")
    parser.add_argument(
        "--manifest",
        default="reports/experiments/20260121/MECH_4000_DWG_LABEL_MANIFEST_MERGED_20260121.csv",
        help="Manifest CSV path",
    )
    parser.add_argument(
        "--dxf-dir",
        required=True,
        help="DXF directory",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/graph2d_merged_latest.pth",
        help="Trained model checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "stratified"],
        default="stratified",
        help="Validation split strategy.",
    )
    parser.add_argument(
        "--output-metrics",
        default="reports/experiments/20260121/MECH_4000_DWG_GRAPH2D_VAL_METRICS_20260121.csv",
        help="Metrics CSV output",
    )
    parser.add_argument(
        "--output-errors",
        default="reports/experiments/20260121/MECH_4000_DWG_GRAPH2D_VAL_ERRORS_20260121.csv",
        help="Error CSV output",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return 1

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_map = checkpoint.get("label_map", {})
    inv_label_map = _inverse_label_map(label_map)
    node_dim = int(checkpoint.get("node_dim", DXF_NODE_DIM))
    hidden_dim = int(checkpoint.get("hidden_dim", 64))
    model_type = checkpoint.get("model_type", "gcn")
    edge_dim = int(checkpoint.get("edge_dim", DXF_EDGE_DIM))
    num_classes = max(1, len(label_map))

    dataset = DXFManifestDataset(
        args.manifest,
        args.dxf_dir,
        label_map=label_map,
        node_dim=node_dim,
        return_edge_attr=model_type == "edge_sage",
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
        val_idx = _stratified_split(indices, labels, args.val_split)
    else:
        split = max(1, int(len(indices) * (1.0 - args.val_split)))
        val_idx = indices[split:] or indices[:1]

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    if model_type == "edge_sage":
        model = EdgeGraphSageClassifier(node_dim, edge_dim, hidden_dim, num_classes)
    else:
        model = SimpleGraphClassifier(node_dim, hidden_dim, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    per_label_total: Dict[int, int] = {idx: 0 for idx in inv_label_map}
    per_label_correct: Dict[int, int] = {idx: 0 for idx in inv_label_map}
    per_label_top2: Dict[int, int] = {idx: 0 for idx in inv_label_map}
    per_label_pred: Dict[int, int] = {idx: 0 for idx in inv_label_map}
    errors: List[Dict[str, Any]] = []

    total = 0
    correct = 0
    top2_correct = 0

    with torch.no_grad():
        for xs, edges, edge_attrs, labels, file_names in val_loader:
            for graph_x, edge_index, edge_attr, label, file_name in zip(
                xs, edges, edge_attrs, labels, file_names
            ):
                if graph_x.numel() == 0:
                    continue
                if model_type == "edge_sage":
                    logits = model(graph_x, edge_index, edge_attr)
                else:
                    logits = model(graph_x, edge_index)
                probs = torch.softmax(logits, dim=1)[0]
                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item())
                topk = min(2, probs.numel())
                top_vals, top_idx = torch.topk(probs, k=topk)
                margin = float(top_vals[0].item() - top_vals[1].item()) if topk > 1 else 1.0

                label_id = int(label)
                per_label_total[label_id] = per_label_total.get(label_id, 0) + 1
                total += 1

                per_label_pred[pred_idx] = per_label_pred.get(pred_idx, 0) + 1

                if pred_idx == label_id:
                    correct += 1
                    per_label_correct[label_id] = per_label_correct.get(label_id, 0) + 1
                else:
                    errors.append(
                        {
                            "file_name": file_name,
                            "true_label": inv_label_map.get(label_id, str(label_id)),
                            "pred_label": inv_label_map.get(pred_idx, str(pred_idx)),
                            "confidence": f"{confidence:.3f}",
                            "margin": f"{margin:.3f}",
                            "bucket": _bucket_error(confidence, margin),
                        }
                    )

                if label_id in [int(idx) for idx in top_idx.tolist()]:
                    top2_correct += 1
                    per_label_top2[label_id] = per_label_top2.get(label_id, 0) + 1

    # Compute macro/weighted F1
    f1_sum = 0.0
    weighted_f1_sum = 0.0
    class_count = 0
    for label_id, total_count in per_label_total.items():
        if total_count == 0:
            continue
        tp = per_label_correct.get(label_id, 0)
        fp = per_label_pred.get(label_id, 0) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(total_count, 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_sum += f1
        weighted_f1_sum += f1 * total_count
        class_count += 1

    macro_f1 = f1_sum / max(class_count, 1)
    weighted_f1 = weighted_f1_sum / max(total, 1)

    output_metrics = Path(args.output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with output_metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label_cn",
                "total",
                "correct",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "top2_accuracy",
                "share",
                "macro_f1",
                "weighted_f1",
            ],
        )
        writer.writeheader()
        for label_id, total_count in sorted(per_label_total.items(), key=lambda item: item[0]):
            if total_count == 0:
                continue
            correct_count = per_label_correct.get(label_id, 0)
            top2_count = per_label_top2.get(label_id, 0)
            pred_count = per_label_pred.get(label_id, 0)
            precision = correct_count / max(pred_count, 1)
            recall = correct_count / max(total_count, 1)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            writer.writerow(
                {
                    "label_cn": inv_label_map.get(label_id, str(label_id)),
                    "total": total_count,
                    "correct": correct_count,
                    "accuracy": f"{correct_count / total_count:.3f}",
                    "precision": f"{precision:.3f}",
                    "recall": f"{recall:.3f}",
                    "f1": f"{f1:.3f}",
                    "top2_accuracy": f"{top2_count / total_count:.3f}",
                    "share": f"{total_count / max(1, total):.3f}",
                    "macro_f1": "",
                    "weighted_f1": "",
                }
            )
        writer.writerow(
            {
                "label_cn": "__overall__",
                "total": total,
                "correct": correct,
                "accuracy": f"{correct / max(1, total):.3f}",
                "precision": "",
                "recall": "",
                "f1": "",
                "top2_accuracy": f"{top2_correct / max(1, total):.3f}",
                "share": "1.000",
                "macro_f1": f"{macro_f1:.3f}",
                "weighted_f1": f"{weighted_f1:.3f}",
            }
        )

    output_errors = Path(args.output_errors)
    output_errors.parent.mkdir(parents=True, exist_ok=True)
    with output_errors.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "true_label",
                "pred_label",
                "confidence",
                "margin",
                "bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(errors)

    print(
        f"Validation samples={total} acc={correct / max(1, total):.3f} "
        f"top2={top2_correct / max(1, total):.3f} "
        f"macro_f1={macro_f1:.3f} weighted_f1={weighted_f1:.3f} errors={len(errors)}"
    )
    print(f"Metrics written to {output_metrics}")
    print(f"Errors written to {output_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
