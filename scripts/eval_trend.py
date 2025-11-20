#!/usr/bin/env python3
"""
Plot evaluation trends from reports/eval_history/*.json.

Generates line charts for COMBINED SCORE over time and (if present)
OCR-only metrics (dimension_recall, 1 - brier_score).

Usage:
    python3 scripts/eval_trend.py --out reports/eval_history/plots
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
import argparse
import os

# Ensure matplotlib uses a non-interactive backend and writable config dir
ROOT = Path(__file__).resolve().parents[1]
_mpl_cfg_dir = ROOT / "reports" / "eval_history" / "mplcache"
_mpl_cfg_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cfg_dir))

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    print("ERROR: matplotlib is required. Install with 'pip install matplotlib'.")
    raise
HISTORY_DIR = ROOT / "reports" / "eval_history"


def load_history() -> tuple[list[dict], list[dict]]:
    combined, ocr_only = [], []
    if not HISTORY_DIR.exists():
        return combined, ocr_only

    for f in sorted(HISTORY_DIR.glob("*.json")):
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            data["_file"] = str(f.relative_to(ROOT))
            if data.get("type") == "combined":
                combined.append(data)
            elif "metrics" in data:  # OCR-only format from eval_with_history.sh
                ocr_only.append(data)
        except Exception:
            continue

    return combined, ocr_only


def to_dt(ts: str) -> datetime:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return datetime.fromtimestamp(0)


def plot_combined(combined: list[dict], outdir: Path) -> Path | None:
    if not combined:
        return None
    xs = [to_dt(d.get("timestamp", "1970-01-01T00:00:00Z")) for d in combined]
    ys = [d.get("combined", {}).get("combined_score", 0.0) for d in combined]
    vws = [d.get("combined", {}).get("vision_weight", 0.5) for d in combined]

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, marker="o", label="Combined Score")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.title("Combined Evaluation Trend")
    plt.xlabel("Time")
    plt.ylabel("Score (0-1)")
    # Annotate last point with weights
    if xs:
        plt.annotate(f"w_v={vws[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(8,8))

    out = outdir / "combined_trend.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_ocr_only(ocr_only: list[dict], outdir: Path) -> Path | None:
    if not ocr_only:
        return None
    xs = [to_dt(d.get("timestamp", "1970-01-01T00:00:00Z")) for d in ocr_only]
    dr = [d.get("metrics", {}).get("dimension_recall", 0.0) for d in ocr_only]
    bs = [d.get("metrics", {}).get("brier_score", 1.0) for d in ocr_only]
    one_minus_brier = [max(0.0, 1.0 - b) for b in bs]

    plt.figure(figsize=(8, 4))
    plt.plot(xs, dr, marker="o", label="Dimension Recall")
    plt.plot(xs, one_minus_brier, marker="s", label="1 - Brier")
    plt.ylim(0, 1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.title("OCR Metrics Trend")
    plt.xlabel("Time")
    plt.ylabel("Score (0-1)")
    plt.legend()

    out = outdir / "ocr_trend.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation trends from history")
    parser.add_argument("--out", type=str, default=str(HISTORY_DIR / "plots"), help="Output directory for plots")
    args = parser.parse_args()

    combined, ocr_only = load_history()
    outdir = Path(args.out)

    cpath = plot_combined(combined, outdir)
    opath = plot_ocr_only(ocr_only, outdir)

    if cpath:
        print(f"Combined trend saved: {cpath}")
    else:
        print("No combined history found.")

    if opath:
        print(f"OCR trend saved: {opath}")
    else:
        print("No OCR-only history found.")


if __name__ == "__main__":
    main()
