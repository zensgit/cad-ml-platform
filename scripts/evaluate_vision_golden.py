#!/usr/bin/env python3
"""Vision Golden Evaluation Script - Stage A MVP.

Minimal evaluation pipeline:
1. Load golden annotation(s) from tests/vision/golden/annotations/
2. Use VisionManager with stub provider to generate descriptions
3. Calculate keyword hit rate (simple metric)
4. Print results table

Usage:
    python scripts/evaluate_vision_golden.py
    python scripts/evaluate_vision_golden.py --dry-run
    python scripts/evaluate_vision_golden.py --limit 1
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.vision import VisionAnalyzeRequest, VisionManager, create_stub_provider
from src.core.vision.base import VisionInputError

# Core evaluation functions


# Baseline values from vision-golden-b1 tag
BASELINE = {
    "avg_hit_rate": 0.667,  # 66.7%
    "min_hit_rate": 0.400,  # 40.0% (sample_003_hard)
    "max_hit_rate": 1.000,  # 100.0% (sample_001_easy)
    "num_samples": 3,
    "tag": "vision-golden-b1",
}


def get_git_info() -> Dict[str, str]:
    """Get current git tag, branch, and commit info."""
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
        pass  # Keep defaults if git not available

    return info


def print_baseline_diff(current: Dict[str, float], baseline: Dict[str, Any]):
    """Print comparison against baseline."""
    print(f"\n--- Baseline Comparison (vs {baseline['tag']}) ---")

    metrics = [
        ("avg_hit_rate", "AVG_HIT_RATE"),
        ("min_hit_rate", "MIN_HIT_RATE"),
        ("max_hit_rate", "MAX_HIT_RATE"),
    ]

    for key, label in metrics:
        curr = current.get(key, 0)
        base = baseline.get(key, 0)
        diff = curr - base

        # Higher is better for hit rates
        if diff > 0.001:
            indicator = "IMPROVED"
        elif diff < -0.001:
            indicator = "REGRESSED"
        else:
            indicator = "SAME"

        diff_str = f"{diff:+.1%}" if abs(diff) > 0.0001 else "±0.0%"
        print(f"{label}: {curr:.1%} (baseline: {base:.1%}, diff: {diff_str}) [{indicator}]")


def load_annotation(annotation_path: Path) -> Dict[str, Any]:
    """Load a single annotation JSON file."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_keyword_hits(description_text: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """Calculate keyword hit statistics."""
    text_lower = description_text.lower()

    hits = []
    misses = []

    for keyword in expected_keywords:
        if keyword.lower() in text_lower:
            hits.append(keyword)
        else:
            misses.append(keyword)

    hit_count = len(hits)
    total_keywords = len(expected_keywords)
    hit_rate = hit_count / total_keywords if total_keywords > 0 else 0.0

    return {
        "total_keywords": total_keywords,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "hits": hits,
        "misses": misses,
    }


async def evaluate_sample(
    sample_id: str, expected_keywords: List[str], image_bytes: bytes
) -> Dict[str, Any]:
    """Evaluate a single sample using VisionManager."""
    # Create VisionManager with stub provider
    vision_provider = create_stub_provider(simulate_latency_ms=10)
    manager = VisionManager(vision_provider=vision_provider, ocr_manager=None)

    # Prepare request (using base64 for simplicity in MVP)
    import base64

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    request = VisionAnalyzeRequest(
        image_base64=image_base64,
        include_description=True,
        include_ocr=False,  # Vision-only for Stage A
    )

    # Execute vision analysis
    try:
        response = await manager.analyze(request)
    except VisionInputError as e:
        # Handle input validation errors gracefully
        return {"sample_id": sample_id, "success": False, "error": str(e)}

    if not response.success or not response.description:
        return {
            "sample_id": sample_id,
            "success": False,
            "error": response.error or "No description generated",
        }

    # Combine summary and details into single text for keyword matching
    description_text = response.description.summary
    if response.description.details:
        description_text += " " + " ".join(response.description.details)

    # Calculate keyword hits
    keyword_stats = calculate_keyword_hits(description_text, expected_keywords)

    return {
        "sample_id": sample_id,
        "success": True,
        "description_summary": response.description.summary,
        "description_confidence": response.description.confidence,
        **keyword_stats,
    }


# Main script


async def main(dry_run: bool = False, limit: int = None):
    """Main evaluation pipeline."""
    # Get git info for tracking
    git_info = get_git_info()

    # Locate annotation files
    annotations_dir = project_root / "tests" / "vision" / "golden" / "annotations"

    if not annotations_dir.exists():
        print(f"Error: Annotations directory not found: {annotations_dir}")
        return 1

    annotation_files = sorted(annotations_dir.glob("*.json"))

    if not annotation_files:
        print(f"Error: No annotation files found in {annotations_dir}")
        return 1

    # Apply limit if specified
    if limit:
        annotation_files = annotation_files[:limit]

    print(f"Vision Golden Evaluation (Stage A MVP)")
    print(f"=" * 60)
    print(f"Timestamp: {git_info['timestamp']}")
    print(f"Git Branch: {git_info['branch']}")
    print(f"Git Commit: {git_info['commit']}")
    if git_info["tag"]:
        print(f"Git Tag: {git_info['tag']}")
    print(f"Annotations directory: {annotations_dir}")
    print(f"Found {len(annotation_files)} annotation(s)")
    print()

    if dry_run:
        print("DRY RUN - would evaluate:")
        for ann_file in annotation_files:
            print(f"  - {ann_file.name}")
        return 0

    # Load sample image (using minimal 1x1 PNG for MVP)
    sample_image_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"

    # Evaluate each annotation
    results = []

    for ann_file in annotation_files:
        annotation = load_annotation(ann_file)
        sample_id = annotation.get("id", ann_file.stem)
        expected_keywords = annotation.get("expected_keywords", [])

        print(f"Evaluating: {sample_id} ({len(expected_keywords)} keywords)...", end=" ")

        result = await evaluate_sample(
            sample_id=sample_id, expected_keywords=expected_keywords, image_bytes=sample_image_bytes
        )

        results.append(result)

        if result["success"]:
            print(
                f"OK - Hit rate: {result['hit_rate']:.1%} ({result['hit_count']}/{result['total_keywords']})"
            )
        else:
            print(f"FAIL - Error: {result.get('error', 'Unknown')}")

    # Print summary table
    print()
    print(f"Results Summary")
    print(f"=" * 60)
    print(f"{'Sample ID':<20} {'Total':>6} {'Hits':>6} {'Rate':>8}")
    print(f"-" * 60)

    for result in results:
        if result["success"]:
            print(
                f"{result['sample_id']:<20} {result['total_keywords']:>6} {result['hit_count']:>6} {result['hit_rate']:>7.1%}"
            )
        else:
            print(f"{result['sample_id']:<20} {'ERROR':>6} {'':>6} {'':>8}")

    # Calculate overall stats
    successful = [r for r in results if r["success"]]
    if successful:
        avg_hit_rate = sum(r["hit_rate"] for r in successful) / len(successful)
        min_result = min(successful, key=lambda r: r["hit_rate"])
        max_result = max(successful, key=lambda r: r["hit_rate"])

        print(f"-" * 60)
        print(f"{'AVERAGE':<20} {'':>6} {'':>6} {avg_hit_rate:>7.1%}")

        # Additional statistics
        print()
        print(f"Statistics")
        print(f"=" * 60)
        print(f"{'NUM_SAMPLES':<20} {len(results)}")
        print(f"{'SUCCESSFUL':<20} {len(successful)}")
        print(f"{'FAILED':<20} {len(results) - len(successful)}")
        print(f"{'AVG_HIT_RATE':<20} {avg_hit_rate:.1%}")
        print(f"{'MIN_HIT_RATE':<20} {min_result['hit_rate']:.1%} ({min_result['sample_id']})")
        print(f"{'MAX_HIT_RATE':<20} {max_result['hit_rate']:.1%} ({max_result['sample_id']})")

        # Baseline comparison
        current_metrics = {
            "avg_hit_rate": avg_hit_rate,
            "min_hit_rate": min_result["hit_rate"],
            "max_hit_rate": max_result["hit_rate"],
        }
        print_baseline_diff(current_metrics, BASELINE)

    print()
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vision Golden Evaluation - Stage A MVP")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be evaluated")
    parser.add_argument("--limit", type=int, help="Maximum number of samples to evaluate")

    args = parser.parse_args()

    exit_code = asyncio.run(main(dry_run=args.dry_run, limit=args.limit))
    sys.exit(exit_code)
