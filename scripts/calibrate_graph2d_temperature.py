#!/usr/bin/env python3
"""Calibrate a 2D graph classifier via temperature scaling."""

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


def _ece(confidences, predictions, labels, n_bins: int = 10) -> float:
    import torch

    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.tensor(0.0)
    total = len(confidences)
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            acc = (predictions[mask] == labels[mask]).float().mean()
            avg_conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(acc - avg_conf)
    return float(ece.item()) if total else 0.0


def main() -> int:
    if not _require_torch():
        return 1

    import torch
    from torch.utils.data import DataLoader, Subset

    from src.ml.train.dataset_2d import DXFManifestDataset, DXF_EDGE_DIM, DXF_NODE_DIM
    from src.ml.train.model_2d import EdgeGraphSageClassifier, SimpleGraphClassifier

    parser = argparse.ArgumentParser(description="Temperature scaling for 2D graph models.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Manifest CSV path",
    )
    parser.add_argument(
        "--dxf-dir",
        required=True,
        help="DXF directory",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
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
        "--output-calibration",
        required=True,
        help="Calibration JSON output path",
    )
    parser.add_argument(
        "--output-predictions",
        required=True,
        help="Predictions CSV output path",
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

    logits_list: List[torch.Tensor] = []
    labels_list: List[int] = []
    file_names: List[str] = []

    with torch.no_grad():
        for xs, edges, edge_attrs, labels, file_names_batch in val_loader:
            for graph_x, edge_index, edge_attr, label, file_name in zip(
                xs, edges, edge_attrs, labels, file_names_batch
            ):
                if graph_x.numel() == 0:
                    continue
                if model_type == "edge_sage":
                    logits = model(graph_x, edge_index, edge_attr)
                else:
                    logits = model(graph_x, edge_index)
                logits_list.append(logits[0])
                labels_list.append(int(label))
                file_names.append(file_name)

    if not logits_list:
        print("No logits collected; aborting.")
        return 1

    logits_tensor = torch.stack(logits_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    # Temperature scaling
    log_t = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_t], lr=0.01, max_iter=50)

    def _nll_loss(temp: torch.Tensor) -> torch.Tensor:
        scaled = logits_tensor / temp
        return torch.nn.functional.cross_entropy(scaled, labels_tensor)

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_t)
        loss = _nll_loss(temp)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_t).item())

    with torch.no_grad():
        probs_before = torch.softmax(logits_tensor, dim=1)
        probs_after = torch.softmax(logits_tensor / temperature, dim=1)
        conf_before, pred_before = torch.max(probs_before, dim=1)
        conf_after, pred_after = torch.max(probs_after, dim=1)

    nll_before = float(_nll_loss(torch.tensor(1.0)).item())
    nll_after = float(_nll_loss(torch.tensor(temperature)).item())
    ece_before = _ece(conf_before, pred_before, labels_tensor)
    ece_after = _ece(conf_after, pred_after, labels_tensor)

    output_calibration = Path(args.output_calibration)
    output_calibration.parent.mkdir(parents=True, exist_ok=True)
    output_calibration.write_text(
        csv.writer  # type: ignore
        if False
        else (
            "{\n"
            f'  "temperature": {temperature:.6f},\n'
            f'  "nll_before": {nll_before:.6f},\n'
            f'  "nll_after": {nll_after:.6f},\n'
            f'  "ece_before": {ece_before:.6f},\n'
            f'  "ece_after": {ece_after:.6f},\n'
            f'  "val_samples": {len(labels_list)}\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    output_predictions = Path(args.output_predictions)
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    with output_predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "true_label",
                "pred_label",
                "confidence_before",
                "confidence_after",
            ],
        )
        writer.writeheader()
        for file_name, true_idx, pred_idx, cb, ca in zip(
            file_names, labels_tensor.tolist(), pred_before.tolist(), conf_before.tolist(), conf_after.tolist()
        ):
            writer.writerow(
                {
                    "file_name": file_name,
                    "true_label": inv_label_map.get(true_idx, str(true_idx)),
                    "pred_label": inv_label_map.get(pred_idx, str(pred_idx)),
                    "confidence_before": f"{cb:.6f}",
                    "confidence_after": f"{ca:.6f}",
                }
            )

    print(
        f"Calibration samples={len(labels_list)} T={temperature:.4f} "
        f"NLL {nll_before:.4f}->{nll_after:.4f} ECE {ece_before:.4f}->{ece_after:.4f}"
    )
    print(f"Calibration written to {output_calibration}")
    print(f"Predictions written to {output_predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
