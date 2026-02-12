#!/usr/bin/env python3
"""Calibrate Graph2D temperature scaling for many-class DXF classification.

This script performs a simple grid search over temperature values and selects
the best temperature using a chosen objective.

Recommended objective for "anonymous DXF" gating:
- maximize precision (accuracy) on the subset of samples whose max-probability
  exceeds a configurable confidence threshold, with a minimum accepted sample count.

Inputs:
- A manifest CSV with weak labels (label_cn, file_name, relative_path)
- A DXF directory
- A Graph2D checkpoint produced by scripts/train_2d_graph.py

Output:
- A JSON file compatible with Graph2DClassifier's loader via
  GRAPH2D_TEMPERATURE_CALIBRATION_PATH (must include "temperature").

Note:
The output JSON intentionally avoids per-file details to prevent leaking
filename-based weak labels into the repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _parse_temperatures(value: str) -> List[float]:
    temps: List[float] = []
    for raw in (value or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        temps.append(float(raw))
    return temps


def _default_temperature_grid() -> List[float]:
    # Wide range, denser around 1.0. Keep deterministic for report diffs.
    base = [
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.25,
        1.4,
        1.6,
        1.8,
        2.0,
        2.5,
        3.0,
        4.0,
    ]
    return [t for t in base if t > 0]


def _candidate_paths(base_dir: Path, file_name: str, relative_path: str) -> List[Path]:
    candidates: List[Path] = []
    rel = (relative_path or "").strip()
    if rel:
        rel_path = Path(rel)
        candidates.append(rel_path if rel_path.is_absolute() else base_dir / rel_path)
    if file_name:
        candidates.append(base_dir / file_name)
        stem = Path(file_name).stem
        if stem:
            candidates.append(base_dir / f"{stem}.dxf")
            candidates.append(base_dir / f"{stem}.DXF")
    return candidates or [base_dir / file_name]


@dataclass(frozen=True)
class _Sample:
    file_path: Path
    label_id: int


def _load_samples(
    manifest_csv: Path,
    dxf_dir: Path,
    label_map: Dict[str, int],
    max_samples: int,
    seed: int,
) -> Tuple[List[_Sample], Dict[str, int]]:
    stats = {
        "rows_total": 0,
        "rows_missing_file": 0,
        "rows_unknown_label": 0,
        "rows_loaded": 0,
    }
    samples: List[_Sample] = []

    with manifest_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stats["rows_total"] += 1
            if not row:
                continue
            label = (row.get("label_cn") or "").strip()
            file_name = (row.get("file_name") or "").strip()
            relative_path = (row.get("relative_path") or "").strip()
            if not label or not file_name:
                continue
            if label not in label_map:
                stats["rows_unknown_label"] += 1
                continue
            label_id = int(label_map[label])
            candidates = _candidate_paths(dxf_dir, file_name=file_name, relative_path=relative_path)
            file_path = next((p for p in candidates if p.exists()), None)
            if file_path is None:
                stats["rows_missing_file"] += 1
                continue
            samples.append(_Sample(file_path=file_path, label_id=label_id))

    random.Random(seed).shuffle(samples)
    if max_samples > 0:
        samples = samples[:max_samples]
    stats["rows_loaded"] = len(samples)
    return samples, stats


def _load_graph2d_checkpoint(model_path: Path) -> Dict[str, Any]:
    import torch

    checkpoint = torch.load(str(model_path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dict")
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint missing model_state_dict")
    return checkpoint


def _build_model_from_checkpoint(checkpoint: Dict[str, Any]):
    import torch

    from src.ml.train.dataset_2d import DXF_EDGE_DIM, DXF_NODE_DIM
    from src.ml.train.model_2d import EdgeGraphSageClassifier, SimpleGraphClassifier

    label_map = checkpoint.get("label_map") or {}
    if not isinstance(label_map, dict):
        raise ValueError("checkpoint label_map must be a dict")
    num_classes = max(1, len(label_map))

    node_dim = int(checkpoint.get("node_dim", DXF_NODE_DIM))
    hidden_dim = int(checkpoint.get("hidden_dim", 64))
    model_type = str(checkpoint.get("model_type", "gcn"))
    edge_dim = int(checkpoint.get("edge_dim", DXF_EDGE_DIM))

    if model_type == "edge_sage":
        model = EdgeGraphSageClassifier(node_dim, edge_dim, hidden_dim, num_classes)
    else:
        model = SimpleGraphClassifier(node_dim, hidden_dim, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {
        "label_map": dict(label_map),
        "num_classes": num_classes,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "model_type": model_type,
        "hidden_dim": hidden_dim,
    }


def _collect_logits(
    model: Any,
    model_meta: Dict[str, Any],
    samples: Sequence[_Sample],
) -> Tuple["torch.Tensor", "torch.Tensor", Dict[str, int]]:
    import ezdxf
    import torch

    from src.ml.train.dataset_2d import DXFDataset

    return_edge_attr = model_meta["model_type"] == "edge_sage"
    dataset = DXFDataset(
        root_dir=".",
        node_dim=int(model_meta["node_dim"]),
        return_edge_attr=bool(return_edge_attr),
    )

    stats = {
        "samples_total": len(samples),
        "samples_empty_graph": 0,
        "samples_parse_error": 0,
        "samples_used": 0,
    }

    logits_list: List[torch.Tensor] = []
    labels_list: List[int] = []

    with torch.no_grad():
        for sample in samples:
            try:
                doc = ezdxf.readfile(str(sample.file_path))
                msp = doc.modelspace()
            except Exception:
                stats["samples_parse_error"] += 1
                continue

            try:
                if return_edge_attr:
                    x, edge_index, edge_attr = dataset._dxf_to_graph(
                        msp, int(model_meta["node_dim"]), return_edge_attr=True
                    )
                else:
                    x, edge_index = dataset._dxf_to_graph(msp, int(model_meta["node_dim"]))
                    edge_attr = None
            except Exception:
                stats["samples_parse_error"] += 1
                continue

            if getattr(x, "numel", lambda: 0)() == 0:
                stats["samples_empty_graph"] += 1
                continue

            if return_edge_attr:
                assert edge_attr is not None
                out = model(x, edge_index, edge_attr)
            else:
                out = model(x, edge_index)
            if out.dim() != 2 or out.size(0) != 1:
                raise ValueError(f"unexpected logits shape: {tuple(out.shape)}")
            logits_list.append(out[0].cpu())
            labels_list.append(int(sample.label_id))

    if not logits_list:
        raise RuntimeError("No usable samples (all graphs empty or parse errors).")

    logits = torch.stack(logits_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.long)
    stats["samples_used"] = int(labels.numel())
    return logits, labels, stats


def _top2_margin(probs: "torch.Tensor") -> "torch.Tensor":
    import torch

    k = min(2, int(probs.size(1)))
    vals, _idx = torch.topk(probs, k=k, dim=1)
    if k < 2:
        return torch.ones((probs.size(0),), dtype=probs.dtype)
    return vals[:, 0] - vals[:, 1]


def _metrics_for_temperature(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    temperature: float,
    confidence_thresholds: Sequence[float],
    margin_thresholds: Sequence[float],
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    scaled = logits / float(temperature)
    nll = float(F.cross_entropy(scaled, labels).item())
    probs = torch.softmax(scaled, dim=1)
    conf, pred = torch.max(probs, dim=1)
    acc = float((pred == labels).float().mean().item())
    margin = _top2_margin(probs)

    def _subset_stats(mask: "torch.Tensor") -> Dict[str, Any]:
        count = int(mask.sum().item())
        if count == 0:
            return {"count": 0, "rate": 0.0, "accuracy": None}
        subset_acc = float((pred[mask] == labels[mask]).float().mean().item())
        return {
            "count": count,
            "rate": float(count) / float(labels.numel()),
            "accuracy": subset_acc,
        }

    conf_stats: Dict[str, Any] = {}
    for thr in confidence_thresholds:
        mask = conf >= float(thr)
        conf_stats[str(thr)] = _subset_stats(mask)

    margin_stats: Dict[str, Any] = {}
    for thr in margin_thresholds:
        mask = margin >= float(thr)
        margin_stats[str(thr)] = _subset_stats(mask)

    return {
        "temperature": float(temperature),
        "nll": nll,
        "accuracy": acc,
        "mean_confidence": float(conf.mean().item()),
        "median_confidence": float(conf.median().item()),
        "confidence_thresholds": conf_stats,
        "margin_thresholds": margin_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Graph2D temperature scaling.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with label_cn/file_name")
    parser.add_argument("--dxf-dir", required=True, help="DXF directory for manifest rows")
    parser.add_argument(
        "--model-path",
        default=os.getenv(
            "GRAPH2D_MODEL_PATH",
            "models/graph2d_training_dxf_oda_titleblock_distill_20260210.pth",
        ),
        help="Graph2D checkpoint path",
    )
    parser.add_argument(
        "--output-json",
        default="models/calibration/graph2d_temperature_calibration.json",
        help="Output JSON path (must include temperature field)",
    )
    parser.add_argument(
        "--temperatures",
        default="",
        help="Comma-separated temperature grid. If empty, uses a default grid.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--objective",
        choices=["nll", "precision_at_conf"],
        default="precision_at_conf",
        help="Selection objective. "
        "'precision_at_conf' maximizes accuracy among samples with confidence >= "
        "objective-confidence-threshold (with objective-min-count guard).",
    )
    parser.add_argument(
        "--objective-confidence-threshold",
        type=float,
        default=0.15,
        help="Used by objective=precision_at_conf.",
    )
    parser.add_argument(
        "--objective-min-count",
        type=int,
        default=20,
        help="Used by objective=precision_at_conf; avoid selecting temperatures "
        "that only look good on a tiny accepted subset.",
    )
    parser.add_argument(
        "--confidence-thresholds",
        default="0.05,0.1,0.15,0.2",
        help="Comma-separated max-prob thresholds for coverage/precision reporting.",
    )
    parser.add_argument(
        "--margin-thresholds",
        default="0.0,0.01,0.02,0.03,0.05",
        help="Comma-separated top1-top2 margin thresholds for coverage/precision reporting.",
    )

    args = parser.parse_args()

    if not _require_torch():
        return 2

    manifest = Path(args.manifest)
    dxf_dir = Path(args.dxf_dir)
    model_path = Path(args.model_path)
    out_path = Path(args.output_json)

    if not manifest.exists():
        print(f"Manifest not found: {manifest}")
        return 2
    if not dxf_dir.exists():
        print(f"DXF dir not found: {dxf_dir}")
        return 2
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 2

    temps = _parse_temperatures(args.temperatures) if args.temperatures else _default_temperature_grid()
    temps = [t for t in temps if t > 0]
    if not temps:
        print("No valid temperatures provided.")
        return 2

    confidence_thresholds = _parse_temperatures(args.confidence_thresholds)
    obj_conf_thr = float(args.objective_confidence_threshold)
    if obj_conf_thr > 0 and all(abs(obj_conf_thr - t) > 1e-9 for t in confidence_thresholds):
        confidence_thresholds.append(obj_conf_thr)
        confidence_thresholds = sorted(set(confidence_thresholds))
    margin_thresholds = _parse_temperatures(args.margin_thresholds)

    checkpoint = _load_graph2d_checkpoint(model_path)
    model, meta = _build_model_from_checkpoint(checkpoint)
    label_map: Dict[str, int] = meta["label_map"]

    samples, sample_stats = _load_samples(
        manifest_csv=manifest,
        dxf_dir=dxf_dir,
        label_map=label_map,
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    if not samples:
        print("No usable samples after filtering (unknown labels / missing files).")
        print(json.dumps(sample_stats, indent=2, ensure_ascii=False))
        return 2

    logits, labels, logit_stats = _collect_logits(model, meta, samples)

    results: List[Dict[str, Any]] = []
    for t in temps:
        results.append(
            _metrics_for_temperature(
                logits=logits,
                labels=labels,
                temperature=float(t),
                confidence_thresholds=confidence_thresholds,
                margin_thresholds=margin_thresholds,
            )
        )

    # Select best by objective.
    best: Dict[str, Any]
    if args.objective == "nll":
        best = min(results, key=lambda r: float(r.get("nll", math.inf)))
    else:
        # Maximize accuracy for samples above a confidence threshold.
        thr_key = str(obj_conf_thr)
        min_count = int(args.objective_min_count)

        def _score(row: Dict[str, Any]) -> Tuple[float, int, float]:
            bucket = row.get("confidence_thresholds", {}).get(thr_key, {})
            count = int(bucket.get("count") or 0)
            acc = bucket.get("accuracy")
            if acc is None or count < min_count:
                # Treat as unusable: lowest score, then smallest count.
                return (-1.0, -1, float("-inf"))
            nll = float(row.get("nll") or math.inf)
            # Higher is better for all tuple fields:
            # - accuracy higher
            # - count higher
            # - (-nll) higher (i.e. nll lower)
            return (float(acc), count, -nll)

        # Sort by: accuracy desc, count desc, nll asc.
        best = sorted(results, key=_score, reverse=True)[0]
        # If everything is unusable, fall back to NLL.
        if _score(best)[0] < 0:
            best = min(results, key=lambda r: float(r.get("nll", math.inf)))

    payload: Dict[str, Any] = {
        "method": "temperature_scaling_grid",
        "objective": str(args.objective),
        "objective_confidence_threshold": obj_conf_thr,
        "objective_min_count": int(args.objective_min_count),
        "fitted": True,
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": int(labels.numel()),
        "model_path": str(model_path),
        "label_map_size": int(meta["num_classes"]),
        "temperature": float(best["temperature"]),
        "best": best,
        "grid": results,
        "stats": {
            "samples": sample_stats,
            "logits": logit_stats,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Best temperature:")
    print(json.dumps(best, indent=2, ensure_ascii=False))
    print(f"Wrote calibration JSON: {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
