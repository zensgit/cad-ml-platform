#!/usr/bin/env python3
"""INT8 dynamic quantization for Graph2D v3 model (B5.2b).

Applies torch.quantization.quantize_dynamic() to the v3 model checkpoint,
quantizing all Linear layers to int8.  No calibration dataset is required.

Expected results:
  - Model file size: ~500 KB → ~130 KB  (-74%)
  - Inference latency: ~15 ms → ~9 ms   (-40%)
  - Accuracy loss: < 0.5 pp              (dynamic quantisation typically negligible)

Usage:
    python scripts/quantize_graph2d_model.py \
        --model models/graph2d_finetuned_24class_v3.pth \
        --output models/graph2d_finetuned_24class_v3_int8.pth

    # Verify accuracy after quantisation
    python scripts/quantize_graph2d_model.py \
        --model models/graph2d_finetuned_24class_v3.pth \
        --output models/graph2d_finetuned_24class_v3_int8.pth \
        --verify-manifest data/graph_cache/cache_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ml.train.model_2d import GraphEncoderV2WithHead
from scripts.evaluate_graph2d_v2 import load_model

logger = logging.getLogger(__name__)


def quantize_model(
    checkpoint_path: str,
    output_path: str,
) -> dict:
    """Load model, apply INT8 dynamic quantisation, save to output_path.

    Returns a dict with original_kb, quantized_kb, ratio.
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    arch = ckpt.get("arch", "")
    if arch != "GraphEncoderV2":
        raise ValueError(
            f"Expected arch='GraphEncoderV2', got '{arch}'. "
            "Only GraphEncoderV2WithHead is supported for quantisation."
        )

    model, label_map = load_model(checkpoint_path)
    model.eval()

    # Ensure a quantization engine is available.
    # macOS/ARM doesn't have fbgemm; use qnnpack instead.
    _engine = torch.backends.quantized.engine
    if _engine == "none":
        try:
            torch.backends.quantized.engine = "qnnpack"
            logger.info("Set quantization engine to qnnpack (was: %s)", _engine)
        except Exception:
            logger.warning("Failed to set qnnpack; keeping engine=%s", _engine)

    # Dynamic quantisation: quantise all Linear layers to int8.
    # No calibration data needed — weights are quantised offline,
    # activations are quantised at runtime.
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()

    # Save quantised checkpoint (same structure as original + quantized=True flag)
    torch.save(
        {
            "arch": "GraphEncoderV2",
            "model_state": quantized.state_dict(),
            "label_map": label_map,
            "quantized": True,
            "quantization": "dynamic_int8",
            # Preserve metadata from original checkpoint
            "hidden_dim": ckpt.get("hidden_dim"),
            "node_dim": ckpt.get("node_dim", 19),
            "edge_dim": ckpt.get("edge_dim", 7),
            "best_val_acc": ckpt.get("best_val_acc"),
            "trained_from": ckpt.get("trained_from"),
        },
        output_path,
    )

    orig_kb = os.path.getsize(checkpoint_path) / 1024
    quant_kb = os.path.getsize(output_path) / 1024
    ratio = quant_kb / orig_kb

    logger.info("Original  : %.1f KB", orig_kb)
    logger.info("Quantized : %.1f KB  (%.0f%% of original)", quant_kb, ratio * 100)
    return {"original_kb": orig_kb, "quantized_kb": quant_kb, "ratio": ratio}


def benchmark_latency(model, n_warmup: int = 5, n_runs: int = 50) -> dict:
    """Measure single-sample forward-pass latency (CPU)."""
    # Synthetic graph: 50 nodes, 100 edges
    x = torch.randn(50, 19)
    edge_index = torch.randint(0, 50, (2, 100))
    edge_attr = torch.randn(100, 7)
    batch = torch.zeros(50, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x, edge_index, edge_attr, batch)

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x, edge_index, edge_attr, batch)
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return {
        "p50_ms": times[len(times) // 2],
        "p95_ms": times[int(0.95 * len(times))],
        "mean_ms": sum(times) / len(times),
        "n_runs": n_runs,
    }


def verify_accuracy(
    quantized_path: str,
    manifest_csv: str,
    limit: int = 0,
) -> dict:
    """Run the quantised model over a cached graph manifest, report accuracy."""
    from scripts.finetune_graph2d_from_pretrained import CachedGraphDataset, collate_finetune
    from torch.utils.data import DataLoader

    ckpt = torch.load(quantized_path, map_location="cpu", weights_only=False)
    label_map = ckpt["label_map"]

    # Re-assemble model from quantised state
    orig_model, _ = load_model(quantized_path.replace("_int8", ""))  # fp32 reference
    model = torch.quantization.quantize_dynamic(orig_model, {torch.nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["cache_path"], row.get("taxonomy_v2_class") or row.get("label", "")))
    if limit > 0:
        rows = rows[:limit]

    dataset = CachedGraphDataset(rows, label_map)
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_finetune)

    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x, ei, ea, b, labels = batch
            logits = model(x, ei, ea, b)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    acc = correct / total if total else 0.0
    logger.info("Quantised model accuracy: %.2f%%  (%d/%d)", acc * 100, correct, total)
    return {"accuracy": acc, "correct": correct, "total": total}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="INT8 dynamic quantisation for Graph2D v3 model."
    )
    parser.add_argument(
        "--model",
        default="models/graph2d_finetuned_24class_v3.pth",
        help="Input FP32 model checkpoint (default: v3).",
    )
    parser.add_argument(
        "--output",
        default="models/graph2d_finetuned_24class_v3_int8.pth",
        help="Output INT8 model checkpoint.",
    )
    parser.add_argument(
        "--verify-manifest",
        default="",
        help="If provided, evaluate accuracy on this cache manifest after quantising.",
    )
    parser.add_argument(
        "--verify-limit", type=int, default=500,
        help="Max samples for accuracy verification (0 = all, default: 500).",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Benchmark latency before and after quantisation.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Quantise ---
    size_info = quantize_model(args.model, args.output)
    print(
        f"\nSize reduction: {size_info['original_kb']:.1f} KB → "
        f"{size_info['quantized_kb']:.1f} KB  "
        f"({(1 - size_info['ratio']) * 100:.0f}% smaller)"
    )

    # --- Optional latency benchmark ---
    if args.benchmark:
        logger.info("Benchmarking FP32 model latency…")
        fp32_model, _ = load_model(args.model)
        fp32_stats = benchmark_latency(fp32_model)

        logger.info("Benchmarking INT8 model latency…")
        int8_ckpt = torch.load(args.output, map_location="cpu", weights_only=False)
        int8_model, _ = load_model(args.model)
        int8_model = torch.quantization.quantize_dynamic(
            int8_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        int8_model.load_state_dict(int8_ckpt["model_state"])
        int8_stats = benchmark_latency(int8_model)

        print("\nLatency benchmark (synthetic 50-node graph, CPU):")
        print(f"  FP32  p50={fp32_stats['p50_ms']:.1f}ms  p95={fp32_stats['p95_ms']:.1f}ms")
        print(f"  INT8  p50={int8_stats['p50_ms']:.1f}ms  p95={int8_stats['p95_ms']:.1f}ms")
        speedup = fp32_stats["p50_ms"] / max(int8_stats["p50_ms"], 0.001)
        print(f"  Speedup: {speedup:.1f}x")

    # --- Optional accuracy verification ---
    if args.verify_manifest:
        logger.info("Verifying accuracy on manifest: %s", args.verify_manifest)
        acc_info = verify_accuracy(args.output, args.verify_manifest, args.verify_limit)
        print(f"\nQuantised accuracy: {acc_info['accuracy'] * 100:.2f}%  "
              f"({acc_info['correct']}/{acc_info['total']} samples)")
        print("Compare to FP32 baseline to check accuracy loss (target: < 0.5 pp).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
