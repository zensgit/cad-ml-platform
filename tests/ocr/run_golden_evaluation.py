"""Golden evaluation placeholder script.

Computes basic metrics (dimension_recall, edge_f1 placeholder, brier_score placeholder).
Later will load real annotated samples.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class GTDimension:
    value: float
    tolerance: float | None = None


@dataclass
class PredDimension:
    value: float
    tolerance: float | None = None
    confidence: float | None = None


def match_dimension(pred: PredDimension, gt: GTDimension) -> bool:
    tol = gt.tolerance if gt.tolerance is not None else 0.05 * gt.value
    return abs(pred.value - gt.value) <= max(tol, 0.05 * gt.value)


def dimension_recall(preds: List[PredDimension], gts: List[GTDimension]) -> float:
    matched = 0
    for gt in gts:
        if any(match_dimension(p, gt) for p in preds):
            matched += 1
    return matched / len(gts) if gts else 0.0


def brier_score(preds: List[PredDimension]) -> float:
    # placeholder: treat each prediction as event=correct (1 if within tolerance of nominal=its own value) -> always 0 now
    # Will refine once ground truth correctness integrated per pred.
    if not preds:
        return 0.0
    return sum(((p.confidence or 0.0) - 1.0) ** 2 for p in preds) / len(preds)


def edge_f1_placeholder() -> float:
    return 0.0  # will implement when bbox evaluation present


# Baseline values from ocr-week1-mvp tag
BASELINE = {
    "dimension_recall": 1.000,
    "brier_score": 0.025,
    "edge_f1": 0.000,
    "tag": "ocr-week1-mvp",
}


def get_git_info() -> dict:
    """Get current git tag, branch, and commit info."""
    project_root = Path(__file__).parent.parent.parent
    info = {
        "branch": "unknown",
        "commit": "unknown",
        "tag": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Get current commit short hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        # Get tag if HEAD is tagged
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            info["tag"] = result.stdout.strip()
        else:
            # Get nearest tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True,
                text=True,
                cwd=project_root,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                info["tag"] = result.stdout.strip() + " (nearest)"
    except Exception:
        pass

    return info


def print_baseline_diff(current: dict, baseline: dict):
    """Print comparison against baseline."""
    print(f"\n--- Baseline Comparison (vs {baseline['tag']}) ---")

    for metric in ["dimension_recall", "brier_score", "edge_f1"]:
        curr = current[metric]
        base = baseline[metric]
        diff = curr - base

        # Determine if change is positive, negative, or neutral
        if metric == "brier_score":
            # Lower is better for Brier score
            if diff < -0.001:
                indicator = "IMPROVED"
            elif diff > 0.001:
                indicator = "REGRESSED"
            else:
                indicator = "SAME"
        else:
            # Higher is better for recall/F1
            if diff > 0.001:
                indicator = "IMPROVED"
            elif diff < -0.001:
                indicator = "REGRESSED"
            else:
                indicator = "SAME"

        diff_str = f"{diff:+.3f}" if abs(diff) > 0.0001 else "±0.000"
        print(f"{metric}: {curr:.3f} (baseline: {base:.3f}, diff: {diff_str}) [{indicator}]")


def run():
    # Get git info
    git_info = get_git_info()

    print(f"=== OCR Golden Evaluation ===")
    print(f"Timestamp: {git_info['timestamp']}")
    print(f"Git Branch: {git_info['branch']}")
    print(f"Git Commit: {git_info['commit']}")
    if git_info["tag"]:
        print(f"Git Tag: {git_info['tag']}")
    print()

    # Synthetic example
    gts = [GTDimension(20.0, 0.02), GTDimension(5.0, None)]
    preds = [PredDimension(20.01, 0.02, 0.9), PredDimension(5.1, None, 0.8)]
    rec = dimension_recall(preds, gts)
    brier = brier_score(preds)
    edge_f1 = edge_f1_placeholder()

    # Standard output (for parsing)
    print(f"dimension_recall={rec:.3f}")
    print(f"brier_score={brier:.3f}")
    print(f"edge_f1={edge_f1:.3f}")

    # Baseline comparison
    current_metrics = {"dimension_recall": rec, "brier_score": brier, "edge_f1": edge_f1}
    print_baseline_diff(current_metrics, BASELINE)


if __name__ == "__main__":
    run()
