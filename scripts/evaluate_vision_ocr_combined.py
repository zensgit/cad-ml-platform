#!/usr/bin/env python3
"""
Vision + OCR Combined Evaluation Script

Runs both Vision and OCR golden evaluations, calculates a combined score,
and generates a unified report for end-to-end system assessment.

Usage:
    python3 scripts/evaluate_vision_ocr_combined.py
    python3 scripts/evaluate_vision_ocr_combined.py --save-history
    python3 scripts/evaluate_vision_ocr_combined.py --report-only
"""

import argparse
import asyncio
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_git_info() -> Dict[str, str]:
    """Get current git branch, commit, and tag info."""
    info = {
        "branch": "unknown",
        "commit": "unknown",
        "tag": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=project_root
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=project_root
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            capture_output=True, text=True, cwd=project_root, stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            info["tag"] = result.stdout.strip()
        else:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True, text=True, cwd=project_root, stderr=subprocess.DEVNULL
            )
            if result.returncode == 0:
                info["tag"] = result.stdout.strip() + " (nearest)"
    except Exception:
        pass

    return info


def sanitize_filename_component(value: str) -> str:
    """Return a filesystem-safe filename component."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    safe = safe.strip("-.")
    return safe or "unknown"


def run_vision_golden() -> Dict[str, float]:
    """Run Vision golden evaluation and extract metrics."""
    result = subprocess.run(
        ["python3", "scripts/evaluate_vision_golden.py"],
        capture_output=True, text=True, cwd=project_root
    )

    metrics = {
        "avg_hit_rate": 0.0,
        "min_hit_rate": 0.0,
        "max_hit_rate": 0.0,
        "num_samples": 0,
        "success": False
    }

    if result.returncode == 0:
        metrics["success"] = True
        # Parse output
        for line in result.stdout.split("\n"):
            if "AVG_HIT_RATE" in line and "%" in line:
                try:
                    # Extract percentage from line like "AVG_HIT_RATE         66.7%"
                    parts = line.split()
                    for p in parts:
                        if "%" in p:
                            metrics["avg_hit_rate"] = float(p.replace("%", "")) / 100
                            break
                except (ValueError, IndexError):
                    pass
            elif "MIN_HIT_RATE" in line and "%" in line:
                try:
                    parts = line.split()
                    for p in parts:
                        if "%" in p:
                            metrics["min_hit_rate"] = float(p.replace("%", "")) / 100
                            break
                except (ValueError, IndexError):
                    pass
            elif "MAX_HIT_RATE" in line and "%" in line:
                try:
                    parts = line.split()
                    for p in parts:
                        if "%" in p:
                            metrics["max_hit_rate"] = float(p.replace("%", "")) / 100
                            break
                except (ValueError, IndexError):
                    pass
            elif "NUM_SAMPLES" in line:
                try:
                    parts = line.split()
                    metrics["num_samples"] = int(parts[-1])
                except (ValueError, IndexError):
                    pass

    return metrics


def run_ocr_golden() -> Dict[str, float]:
    """Run OCR golden evaluation and extract metrics."""
    result = subprocess.run(
        ["python3", "tests/ocr/run_golden_evaluation.py"],
        capture_output=True, text=True, cwd=project_root
    )

    metrics = {
        "dimension_recall": 0.0,
        "brier_score": 1.0,  # worst case
        "edge_f1": 0.0,
        "success": False
    }

    if result.returncode == 0:
        metrics["success"] = True
        # Parse output
        for line in result.stdout.split("\n"):
            if "dimension_recall=" in line:
                try:
                    value = line.split("=")[1].strip()
                    metrics["dimension_recall"] = float(value)
                except (ValueError, IndexError):
                    pass
            elif "brier_score=" in line:
                try:
                    value = line.split("=")[1].strip()
                    metrics["brier_score"] = float(value)
                except (ValueError, IndexError):
                    pass
            elif "edge_f1=" in line:
                try:
                    value = line.split("=")[1].strip()
                    metrics["edge_f1"] = float(value)
                except (ValueError, IndexError):
                    pass

    return metrics


def validate_and_normalize_weights(
    vision_weight: float,
    ocr_weight: float
) -> tuple[float, float, bool]:
    """
    Validate and normalize weights.
    Returns (vision_weight, ocr_weight, fallback_used).
    """
    fallback_used = False

    # Check for invalid weights (negative or zero sum)
    if vision_weight < 0 or ocr_weight < 0:
        vision_weight = 0.5
        ocr_weight = 0.5
        fallback_used = True
    elif vision_weight + ocr_weight <= 0:
        vision_weight = 0.5
        ocr_weight = 0.5
        fallback_used = True
    else:
        # Normalize weights if they don't sum to 1
        total_weight = vision_weight + ocr_weight
        if abs(total_weight - 1.0) > 0.001:
            vision_weight = vision_weight / total_weight
            ocr_weight = ocr_weight / total_weight

    return vision_weight, ocr_weight, fallback_used


def calculate_combined_score(
    vision: Dict[str, float],
    ocr: Dict[str, float],
    vision_weight: float = 0.5,
    ocr_weight: float = 0.5
) -> Dict[str, float]:
    """
    Calculate combined score from Vision and OCR metrics.

    Combined Score Formula (MVP):
    - Vision component: avg_hit_rate (0-1)
    - OCR component: dimension_recall * (1 - brier_score) (0-1)
    - Weights: configurable (default 50% Vision, 50% OCR)

    Returns score in range [0, 1] where 1 is perfect.
    """
    # Vision score (0-1)
    vision_score = vision.get("avg_hit_rate", 0.0)

    # OCR score: reward high recall, penalize high brier (uncertainty)
    # brier_score in [0, 1], lower is better
    ocr_recall = ocr.get("dimension_recall", 0.0)
    ocr_brier = ocr.get("brier_score", 1.0)
    ocr_score = ocr_recall * (1 - ocr_brier)

    # Weighted combination
    combined = vision_weight * vision_score + ocr_weight * ocr_score

    return {
        "vision_score": vision_score,
        "ocr_score": ocr_score,
        "combined_score": combined,
        "vision_weight": vision_weight,
        "ocr_weight": ocr_weight
    }


# Baselines from milestone tags
BASELINES = {
    "vision": {
        "avg_hit_rate": 0.667,
        "tag": "vision-golden-b1"
    },
    "ocr": {
        "dimension_recall": 1.0,
        "brier_score": 0.025,
        "tag": "ocr-week1-mvp"
    },
    "combined": {
        "vision_score": 0.667,
        "ocr_score": 1.0 * (1 - 0.025),  # 0.975
        "combined_score": 0.5 * 0.667 + 0.5 * 0.975  # 0.821
    }
}


def print_report(
    git_info: Dict[str, str],
    vision: Dict[str, float],
    ocr: Dict[str, float],
    combined: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
    fallback_used: bool = False
):
    """Print comprehensive evaluation report."""
    print("=" * 70)
    print("        VISION + OCR COMBINED EVALUATION REPORT")
    print("=" * 70)
    print()

    # Warning if fallback weights were used
    if fallback_used:
        print("WARNING: Invalid weights provided, fallback to 0.5/0.5")
        print()

    # Effective weights (always show after normalization)
    v_weight = combined.get("vision_weight", 0.5)
    o_weight = combined.get("ocr_weight", 0.5)
    print(f"Effective Weights: vision={v_weight:.2f}, ocr={o_weight:.2f} (normalized)")

    # Thresholds (only if provided)
    if thresholds:
        threshold_parts = []
        if thresholds.get("min_combined") is not None:
            threshold_parts.append(f"combined>={thresholds['min_combined']:.3f}")
        if thresholds.get("min_vision") is not None:
            threshold_parts.append(f"vision>={thresholds['min_vision']:.3f}")
        if thresholds.get("min_ocr") is not None:
            threshold_parts.append(f"ocr>={thresholds['min_ocr']:.3f}")
        if threshold_parts:
            print(f"Thresholds: {', '.join(threshold_parts)}")

    print()

    # Git info
    print(f"Timestamp: {git_info['timestamp']}")
    print(f"Git Branch: {git_info['branch']}")
    print(f"Git Commit: {git_info['commit']}")
    if git_info['tag']:
        print(f"Git Tag: {git_info['tag']}")
    print()

    # Vision metrics
    print("--- Vision Module ---")
    if vision["success"]:
        print(f"  AVG_HIT_RATE:  {vision['avg_hit_rate']:.1%}")
        print(f"  MIN_HIT_RATE:  {vision['min_hit_rate']:.1%}")
        print(f"  MAX_HIT_RATE:  {vision['max_hit_rate']:.1%}")
        print(f"  NUM_SAMPLES:   {vision['num_samples']}")
    else:
        print("  ERROR: Vision evaluation failed")
    print()

    # OCR metrics
    print("--- OCR Module ---")
    if ocr["success"]:
        print(f"  dimension_recall: {ocr['dimension_recall']:.3f}")
        print(f"  brier_score:      {ocr['brier_score']:.3f}")
        print(f"  edge_f1:          {ocr['edge_f1']:.3f}")
    else:
        print("  ERROR: OCR evaluation failed")
    print()

    # Combined score
    print("--- Combined Score ---")
    print(f"  Vision Score:    {combined['vision_score']:.3f} (weight: {combined['vision_weight']:.0%})")
    print(f"  OCR Score:       {combined['ocr_score']:.3f} (weight: {combined['ocr_weight']:.0%})")
    print(f"  COMBINED SCORE:  {combined['combined_score']:.3f}")
    print()

    # Baseline comparison
    print("--- Baseline Comparison ---")

    # Vision
    base_vision = BASELINES["vision"]["avg_hit_rate"]
    diff_vision = combined["vision_score"] - base_vision
    indicator = "SAME" if abs(diff_vision) < 0.001 else ("IMPROVED" if diff_vision > 0 else "REGRESSED")
    print(f"  Vision:   {combined['vision_score']:.3f} vs {base_vision:.3f} (diff: {diff_vision:+.3f}) [{indicator}]")

    # OCR
    base_ocr = BASELINES["combined"]["ocr_score"]
    diff_ocr = combined["ocr_score"] - base_ocr
    indicator = "SAME" if abs(diff_ocr) < 0.001 else ("IMPROVED" if diff_ocr > 0 else "REGRESSED")
    print(f"  OCR:      {combined['ocr_score']:.3f} vs {base_ocr:.3f} (diff: {diff_ocr:+.3f}) [{indicator}]")

    # Combined
    base_combined = BASELINES["combined"]["combined_score"]
    diff_combined = combined["combined_score"] - base_combined
    indicator = "SAME" if abs(diff_combined) < 0.001 else ("IMPROVED" if diff_combined > 0 else "REGRESSED")
    print(f"  Combined: {combined['combined_score']:.3f} vs {base_combined:.3f} (diff: {diff_combined:+.3f}) [{indicator}]")

    print()
    print("=" * 70)

    # Health summary
    all_same = (
        abs(diff_vision) < 0.001 and
        abs(diff_ocr) < 0.001 and
        abs(diff_combined) < 0.001
    )

    if all_same:
        print("STATUS: ALL METRICS STABLE - No regression detected")
    elif diff_combined >= 0:
        print("STATUS: IMPROVED OR STABLE - Combined score maintained or better")
    else:
        print("STATUS: WARNING - Combined score regression detected")

    print("=" * 70)


def save_history(
    git_info: Dict[str, str],
    vision: Dict[str, float],
    ocr: Dict[str, float],
    combined: Dict[str, float]
) -> str:
    """Save combined evaluation results to JSON history file."""
    history_dir = project_root / "reports" / "eval_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch = git_info["branch"]
    commit = git_info["commit"]
    safe_branch = sanitize_filename_component(branch)

    filename = f"{timestamp}_{safe_branch}_{commit}_combined.json"
    filepath = history_dir / filename

    # Get run context information
    start_time = time.time()
    runner = os.environ.get("CI", "false") == "true"
    runner_type = "ci" if runner else "local"

    data = {
        "schema_version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branch": branch,
        "commit": commit,
        "type": "combined",
        "run_context": {
            "runner": runner_type,
            "machine": platform.node(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "start_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ci_job_id": os.environ.get("GITHUB_RUN_ID", None),
            "ci_workflow": os.environ.get("GITHUB_WORKFLOW", None)
        },
        "vision_metrics": {
            "avg_hit_rate": vision["avg_hit_rate"],
            "min_hit_rate": vision["min_hit_rate"],
            "max_hit_rate": vision["max_hit_rate"],
            "num_samples": vision["num_samples"]
        },
        "ocr_metrics": {
            "dimension_recall": ocr["dimension_recall"],
            "brier_score": ocr["brier_score"],
            "edge_f1": ocr["edge_f1"]
        },
        "combined": {
            "vision_score": combined["vision_score"],
            "ocr_score": combined["ocr_score"],
            "combined_score": combined["combined_score"],
            "vision_weight": combined["vision_weight"],
            "ocr_weight": combined["ocr_weight"]
        }
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return str(filepath)


def check_thresholds(
    combined: Dict[str, float],
    min_combined: Optional[float] = None,
    min_vision: Optional[float] = None,
    min_ocr: Optional[float] = None
) -> bool:
    """Check if scores meet minimum thresholds. Returns True if all pass."""
    failures = []

    if min_combined is not None and combined["combined_score"] < min_combined:
        failures.append(f"Combined score {combined['combined_score']:.3f} < {min_combined:.3f}")

    if min_vision is not None and combined["vision_score"] < min_vision:
        failures.append(f"Vision score {combined['vision_score']:.3f} < {min_vision:.3f}")

    if min_ocr is not None and combined["ocr_score"] < min_ocr:
        failures.append(f"OCR score {combined['ocr_score']:.3f} < {min_ocr:.3f}")

    if failures:
        print()
        print("=" * 70)
        print("THRESHOLD ALERT - Scores below minimum requirements:")
        for f in failures:
            print(f"  ❌ {f}")
        print("=" * 70)
        return False

    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vision + OCR Combined Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/evaluate_vision_ocr_combined.py
  python3 scripts/evaluate_vision_ocr_combined.py --save-history
  python3 scripts/evaluate_vision_ocr_combined.py --vision-weight 0.6 --ocr-weight 0.4
  python3 scripts/evaluate_vision_ocr_combined.py --min-combined 0.8
  python3 scripts/evaluate_vision_ocr_combined.py --report-only
        """
    )

    parser.add_argument(
        "--save-history",
        action="store_true",
        help="Save results to JSON history file"
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Show baseline configuration only"
    )

    parser.add_argument(
        "--vision-weight",
        type=float,
        default=0.5,
        help="Weight for Vision score (default: 0.5)"
    )

    parser.add_argument(
        "--ocr-weight",
        type=float,
        default=0.5,
        help="Weight for OCR score (default: 0.5)"
    )

    parser.add_argument(
        "--min-combined",
        type=float,
        default=None,
        help="Minimum combined score threshold (alert if below)"
    )

    parser.add_argument(
        "--min-vision",
        type=float,
        default=None,
        help="Minimum Vision score threshold (alert if below)"
    )

    parser.add_argument(
        "--min-ocr",
        type=float,
        default=None,
        help="Minimum OCR score threshold (alert if below)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.report_only:
        # Just show baselines
        print("Combined Evaluation Baselines")
        print("=" * 50)
        print(f"Vision baseline: {BASELINES['vision']['avg_hit_rate']:.3f} ({BASELINES['vision']['tag']})")
        print(f"OCR baseline:    {BASELINES['combined']['ocr_score']:.3f} ({BASELINES['ocr']['tag']})")
        print(f"Combined:        {BASELINES['combined']['combined_score']:.3f}")
        return 0

    # Validate and normalize weights
    vision_weight, ocr_weight, fallback_used = validate_and_normalize_weights(
        args.vision_weight, args.ocr_weight
    )

    # Read thresholds from environment if not provided via CLI
    def _env_float(name: str) -> Optional[float]:
        val = os.getenv(name)
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None

    if args.min_combined is None:
        args.min_combined = _env_float("MIN_COMBINED")
    if args.min_vision is None:
        args.min_vision = _env_float("MIN_VISION")
    if args.min_ocr is None:
        args.min_ocr = _env_float("MIN_OCR")

    # Get git info
    git_info = get_git_info()

    print("Running Vision+OCR Combined Evaluation...")
    if fallback_used:
        print(f"WARNING: Invalid weights provided, using fallback 0.5/0.5")
    elif args.vision_weight != 0.5 or args.ocr_weight != 0.5:
        print(f"Custom weights: Vision={args.vision_weight:.2f}, OCR={args.ocr_weight:.2f}")
    print()

    # Run evaluations
    print("Step 1/3: Running Vision golden evaluation...")
    vision_metrics = run_vision_golden()
    print(f"  Vision: {'OK' if vision_metrics['success'] else 'FAILED'}")

    print("Step 2/3: Running OCR golden evaluation...")
    ocr_metrics = run_ocr_golden()
    print(f"  OCR: {'OK' if ocr_metrics['success'] else 'FAILED'}")

    print("Step 3/3: Calculating combined score...")
    combined_scores = calculate_combined_score(
        vision_metrics, ocr_metrics,
        vision_weight=vision_weight,
        ocr_weight=ocr_weight
    )
    print(f"  Combined: {combined_scores['combined_score']:.3f}")
    print()

    # Build thresholds dict for reporting
    thresholds = {
        "min_combined": args.min_combined,
        "min_vision": args.min_vision,
        "min_ocr": args.min_ocr
    }

    # Print report
    print_report(
        git_info, vision_metrics, ocr_metrics, combined_scores,
        thresholds=thresholds,
        fallback_used=fallback_used
    )

    # Check thresholds if specified
    threshold_pass = check_thresholds(
        combined_scores,
        min_combined=args.min_combined,
        min_vision=args.min_vision,
        min_ocr=args.min_ocr
    )

    # Save history if requested
    if args.save_history:
        filepath = save_history(git_info, vision_metrics, ocr_metrics, combined_scores)
        # Show relative path and weights for easy retrieval
        rel_path = Path(filepath).relative_to(project_root)
        v_w = combined_scores["vision_weight"]
        o_w = combined_scores["ocr_weight"]
        print()
        print(f"Saved: {rel_path} (weights: v={v_w:.2f}, o={o_w:.2f})")

    # Return non-zero if thresholds failed (useful for CI)
    return 0 if threshold_pass else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
