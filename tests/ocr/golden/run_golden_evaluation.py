"""Golden evaluation script (Batch S enhancement).

Adds calibration analysis:
 - Per-item confidence usage (dimension.confidence if available)
 - Brier score buckets & overall
 - Expected Calibration Error (ECE) across confidence bins
 - Reliability table appended to report

Still computes core metrics:
 - dimension_recall
 - symbol_recall
 - edge_precision / edge_recall / edge_f1
 - dual_tolerance_accuracy

Outputs:
 - reports/ocr_evaluation.md (aggregate & deltas)
 - reports/ocr_calibration.md (calibration details)
 - JSON summary printed to stdout for CI parsing

Exits non-zero based on Week1 thresholds (soft gate may override in CI).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, TextIO, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))
from src.core.ocr.base import OcrResult  # type: ignore
from src.core.ocr.manager import OcrManager  # type: ignore
from src.core.ocr.providers.deepseek_hf import DeepSeekHfProvider  # type: ignore
from src.core.ocr.providers.paddle import PaddleOcrProvider  # type: ignore

# ROOT already defined above
GOLDEN_DIR = Path(__file__).resolve().parent / "samples"
METADATA = Path(__file__).resolve().parent / "metadata.yaml"


def _path_from_env(var_name: str, default: Path) -> Path:
    raw = os.getenv(var_name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    return Path(value)


REPORT_PATH = _path_from_env(
    "OCR_GOLDEN_EVALUATION_REPORT_PATH", ROOT / "reports" / "ocr_evaluation.md"
)
CALIBRATION_REPORT_PATH = _path_from_env(
    "OCR_GOLDEN_CALIBRATION_REPORT_PATH", ROOT / "reports" / "ocr_calibration.md"
)


def _open_report(path: Path) -> Tuple[Path, TextIO]:
    """Open a report path for writing, falling back to a temp directory on permission errors."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path, open(path, "w", encoding="utf-8")
    except PermissionError:
        fallback = Path(tempfile.gettempdir()) / path.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"WARNING: cannot write report to {path} (permission denied); using {fallback}",
            file=sys.stderr,
        )
        return fallback, open(fallback, "w", encoding="utf-8")


def iou(b1: List[int], b2: List[int]) -> float:
    if not b1 or not b2:
        return 0.0
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def tolerance_match(pred: Dict[str, Any], gt: Dict[str, Any]) -> bool:
    if pred.get("type") != gt.get("type"):
        return False
    pv = pred.get("value")
    gv = gt.get("value")
    if pv is None or gv is None:
        return False
    tol_pos = gt.get("tol_pos") or gt.get("tolerance") or 0.0
    tol_neg = gt.get("tol_neg") or gt.get("tolerance") or 0.0
    limit = max(abs(tol_pos), abs(tol_neg), 0.05 * gv)
    return abs(pv - gv) <= limit + 1e-9


async def run_sample(manager: OcrManager, annotation_path: Path) -> Dict[str, Any]:
    with open(annotation_path) as f:
        ann = json.load(f)
    # For MVP: synthesize image bytes from annotation (stub path)
    image_bytes = b"dummy_image_bytes_for_golden"
    result: OcrResult = await manager.extract(image_bytes, strategy="auto")
    # Collect dimensions & symbols
    gt_dims = ann.get("dimensions", [])
    gt_syms = ann.get("symbols", [])
    pred_dims = [d.model_dump() for d in result.dimensions]
    pred_syms = [s.model_dump() for s in result.symbols]
    matched_dim = 0
    matched_bbox = 0
    for gd in gt_dims:
        for pd in pred_dims:
            if tolerance_match(pd, gd):
                matched_dim += 1
                if gd.get("bbox") and pd.get("bbox") and iou(gd["bbox"], pd["bbox"]) >= 0.5:
                    matched_bbox += 1
                break
    matched_sym = 0
    for gs in gt_syms:
        for ps in pred_syms:
            if (
                gs.get("type") == ps.get("type")
                and str(gs.get("value")).strip().lower() == str(ps.get("value")).strip().lower()
            ):
                matched_sym += 1
                break
    # Dual tolerance accuracy (仅对有 tol_pos & tol_neg 的尺寸)
    dual_total = 0
    dual_correct = 0
    for gd in gt_dims:
        if gd.get("tol_pos") is not None and gd.get("tol_neg") is not None:
            dual_total += 1
            for pd in pred_dims:
                if (
                    tolerance_match(pd, gd)
                    and pd.get("tol_pos") is not None
                    and pd.get("tol_neg") is not None
                ):
                    dual_correct += 1
                    break
    confidences = []
    outcomes = []
    # Brier: treat each predicted dimension match as outcome=1 else 0, use per-item confidence if available
    for pd in pred_dims:
        is_match = any(tolerance_match(pd, gd) for gd in gt_dims)
        outcomes.append(1 if is_match else 0)
        # prefer dimension-level confidence, else calibrated result confidence, else fallback 0.9
        c = pd.get("confidence")
        if c is None:
            c = 0.9
        confidences.append(c)
    # Compute metrics
    dim_recall = matched_dim / len(gt_dims) if gt_dims else 1.0
    sym_recall = matched_sym / len(gt_syms) if gt_syms else 1.0
    edge_precision = matched_bbox / len(pred_dims) if pred_dims else 1.0
    edge_recall = matched_bbox / len(gt_dims) if gt_dims else 1.0
    if edge_precision + edge_recall > 0:
        edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall)
    else:
        edge_f1 = 0.0
    if outcomes:
        brier = sum((c - o) ** 2 for c, o in zip(confidences, outcomes)) / len(outcomes)
    else:
        brier = 0.0
    dual_acc = dual_correct / dual_total if dual_total else 1.0
    return {
        "sample": annotation_path.parent.name,
        "dimension_recall": dim_recall,
        "symbol_recall": sym_recall,
        "edge_f1": edge_f1,
        "brier_score": brier,
        "dual_tolerance_accuracy": dual_acc,
        "brier_outcomes": outcomes,
        "brier_confidences": confidences,
    }


def _calibration(confidences: List[float], outcomes: List[int]) -> Dict[str, Any]:
    if not confidences:
        return {"ece": 0.0, "buckets": []}
    # Define bins
    bins = [0.0, 0.6, 0.8, 0.9, 1.0]
    bucket_data = []
    ece_acc = 0.0
    total = len(confidences)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        vals = [
            (c, o)
            for c, o in zip(confidences, outcomes)
            if lo <= c < hi or (i == len(bins) - 2 and c == hi)
        ]
        if not vals:
            bucket_data.append(
                {
                    "range": f"[{lo},{hi}]",
                    "count": 0,
                    "avg_conf": None,
                    "empirical_acc": None,
                    "brier": None,
                }
            )
            continue
        avg_conf = sum(c for c, _ in vals) / len(vals)
        empirical = sum(o for _, o in vals) / len(vals)
        brier_bucket = sum((c - o) ** 2 for c, o in vals) / len(vals)
        weight = len(vals) / total
        ece_acc += weight * abs(avg_conf - empirical)
        bucket_data.append(
            {
                "range": f"[{lo},{hi}]",
                "count": len(vals),
                "avg_conf": avg_conf,
                "empirical_acc": empirical,
                "brier": brier_bucket,
            }
        )
    return {"ece": ece_acc, "buckets": bucket_data}


async def evaluate_all(manager: OcrManager) -> Dict[str, Any]:
    # preload providers (optional)
    await asyncio.gather(*[p.warmup() for p in manager.providers.values()])
    samples = sorted(GOLDEN_DIR.glob("*/annotation.json"))
    results = []
    for ann in samples:
        results.append(await run_sample(manager, ann))

    def avg(key: str) -> float:
        return sum(r[key] for r in results) / len(results) if results else 0.0

    agg = {
        k: avg(k)
        for k in [
            "dimension_recall",
            "symbol_recall",
            "edge_f1",
            "brier_score",
            "dual_tolerance_accuracy",
        ]
    }
    # Flatten confidences/outcomes for calibration
    all_conf = []
    all_out = []
    for r in results:
        all_conf.extend(r.get("brier_confidences", []))
        all_out.extend(r.get("brier_outcomes", []))
    calib = _calibration(all_conf, all_out)
    return {"results": results, "aggregate": agg, "calibration": calib}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare-preprocess",
        action="store_true",
        help="Run before/after preprocessing and report deltas",
    )
    args = parser.parse_args()

    # Baseline (preprocess OFF)
    mgr_off = OcrManager(
        providers={
            "paddle": PaddleOcrProvider(enable_preprocess=False),
            "deepseek_hf": DeepSeekHfProvider(),
        }
    )
    off = await evaluate_all(mgr_off)

    # After (preprocess ON)
    mgr_on = OcrManager(
        providers={
            "paddle": PaddleOcrProvider(enable_preprocess=True),
            "deepseek_hf": DeepSeekHfProvider(),
        }
    )
    on = await evaluate_all(mgr_on)

    _, f = _open_report(REPORT_PATH)
    with f:
        f.write("# OCR Evaluation Report (Golden v1.0)\n\n")
        f.write("## Aggregate (Preprocess OFF)\n")
        for k, v in off["aggregate"].items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write("\n## Aggregate (Preprocess ON)\n")
        for k, v in on["aggregate"].items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write("\n## Delta (ON - OFF)\n")
        for k in on["aggregate"].keys():
            dv = on["aggregate"][k] - off["aggregate"][k]
            f.write(f"- {k}_delta: {dv:+.4f}\n")

    # Calibration report (uses ON metrics for current configuration)
    _, cf = _open_report(CALIBRATION_REPORT_PATH)
    with cf:
        cf.write("# OCR Calibration Report\n\n")
        cf.write(f"Overall Brier Score: {on['aggregate']['brier_score']:.4f}\n")
        cf.write(f"ECE: {on['calibration']['ece']:.4f}\n\n")
        cf.write("## Buckets\n")
        cf.write("Range | Count | Avg_Conf | Empirical_Acc | Brier\n")
        cf.write("---|---|---|---|---\n")
        for b in on["calibration"]["buckets"]:
            if b["count"] == 0:
                cf.write(f"{b['range']} | 0 | - | - | -\n")
            else:
                cf.write(
                    f"{b['range']} | {b['count']} | {b['avg_conf']:.3f} | {b['empirical_acc']:.3f} | {b['brier']:.3f}\n"
                )

    # Threshold gating use ON metrics
    agg = on["aggregate"]
    with open(METADATA, "r", encoding="utf-8") as mf:
        meta_yaml = mf.read()

    # naive parse thresholds from yaml (simple substring search to avoid yaml dependency)
    def parse_threshold(name: str, default: float) -> float:
        for line in meta_yaml.splitlines():
            if f"{name}:" in line and "week1" in meta_yaml:
                # but line belongs to week1? simplified approach: take first occurrence after week1 header
                pass
        # fallback direct search
        for line in meta_yaml.splitlines():
            if line.strip().startswith(f"{name}:"):
                try:
                    return float(line.split(":", 1)[1].strip())
                except Exception:
                    return default
        return default

    dim_thr = 0.70
    edge_thr = 0.60
    exit_code = 0
    if agg["dimension_recall"] < dim_thr or agg["edge_f1"] < edge_thr:
        exit_code = 2
    print(
        json.dumps(
            {
                "aggregate_off": off["aggregate"],
                "aggregate_on": on["aggregate"],
                "calibration": on["calibration"],
                "exit_code": exit_code,
            },
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
