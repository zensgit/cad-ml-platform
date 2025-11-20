#!/usr/bin/env python3
"""
Generate status badges for README.

Creates shields.io compatible badges for evaluation scores.

Usage:
    python3 scripts/generate_badge.py [--format json|url|markdown]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_latest_evaluation(history_dir: str = "reports/eval_history") -> Optional[Dict]:
    """Load the most recent evaluation results."""
    json_files = sorted(Path(history_dir).glob("*_combined.json"))

    if not json_files:
        return None

    latest_file = json_files[-1]

    try:
        with open(latest_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {latest_file}: {e}", file=sys.stderr)
        return None


def get_score_color(score: float) -> str:
    """Get badge color based on score."""
    if score >= 0.9:
        return "brightgreen"
    elif score >= 0.8:
        return "green"
    elif score >= 0.7:
        return "yellow"
    elif score >= 0.6:
        return "orange"
    else:
        return "red"


def get_status_color(status: str) -> str:
    """Get badge color based on status."""
    status_colors = {
        "passing": "brightgreen",
        "warning": "yellow",
        "failing": "red",
        "error": "lightgrey"
    }
    return status_colors.get(status, "blue")


def generate_badge_url(label: str, message: str, color: str) -> str:
    """Generate shields.io badge URL."""
    # URL encode special characters
    label = label.replace("-", "--").replace("_", "__").replace(" ", "_")
    message = message.replace("-", "--").replace("_", "__").replace(" ", "_")

    return f"https://img.shields.io/badge/{label}-{message}-{color}"


def generate_badges(evaluation: Dict) -> Dict[str, str]:
    """Generate all badge URLs from evaluation data."""
    badges = {}

    # Extract scores (handle both formats)
    if "scores" in evaluation:
        combined = evaluation["scores"]["combined"]
        vision = evaluation["scores"]["vision"]["score"]
        ocr = evaluation["scores"]["ocr"]["normalized"]
    elif "combined" in evaluation:
        combined = evaluation["combined"].get("combined_score", 0)
        vision = evaluation["combined"].get("vision_score", 0)
        ocr = evaluation["combined"].get("ocr_score", 0)
    else:
        return badges

    # Combined score badge
    badges["combined"] = generate_badge_url(
        "Combined",
        f"{combined:.3f}",
        get_score_color(combined)
    )

    # Vision score badge
    badges["vision"] = generate_badge_url(
        "Vision",
        f"{vision:.3f}",
        get_score_color(vision)
    )

    # OCR score badge
    badges["ocr"] = generate_badge_url(
        "OCR",
        f"{ocr:.3f}",
        get_score_color(ocr)
    )

    # Overall status badge
    if combined >= 0.8:
        status = "passing"
    elif combined >= 0.6:
        status = "warning"
    else:
        status = "failing"

    badges["status"] = generate_badge_url(
        "Evaluation",
        status,
        get_status_color(status)
    )

    # Integrity badge (always monitored)
    badges["integrity"] = generate_badge_url(
        "Integrity",
        "monitored",
        "blue"
    )

    return badges


def format_markdown(badges: Dict[str, str]) -> str:
    """Format badges as Markdown."""
    lines = []

    # Main badges in order
    badge_order = ["status", "combined", "vision", "ocr", "integrity"]

    for key in badge_order:
        if key in badges:
            alt_text = key.capitalize()
            lines.append(f"[![{alt_text}]({badges[key]})](docs/EVAL_SYSTEM_COMPLETE_GUIDE.md)")

    return " ".join(lines)


def update_readme(badges_markdown: str, readme_path: str = "README.md") -> bool:
    """Update README.md with new badges."""
    try:
        with open(readme_path, "r") as f:
            content = f.read()

        # Find badge section (between specific markers)
        start_marker = "<!-- BADGES_START -->"
        end_marker = "<!-- BADGES_END -->"

        if start_marker in content and end_marker in content:
            # Replace existing badges
            before = content.split(start_marker)[0]
            after = content.split(end_marker)[1]

            new_content = f"{before}{start_marker}\n{badges_markdown}\n{end_marker}{after}"

            with open(readme_path, "w") as f:
                f.write(new_content)

            return True
        else:
            print(f"Badge markers not found in {readme_path}", file=sys.stderr)
            return False

    except Exception as e:
        print(f"Error updating README: {e}", file=sys.stderr)
        return False


def save_badge_json(badges: Dict[str, str], output_path: str = "reports/badges.json") -> None:
    """Save badge URLs to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(badges, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate status badges")
    parser.add_argument("--format", choices=["json", "url", "markdown"],
                        default="markdown", help="Output format")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--update-readme", action="store_true",
                        help="Update README.md with badges")
    parser.add_argument("--key", help="Specific badge key to output")

    args = parser.parse_args()

    # Load latest evaluation
    evaluation = load_latest_evaluation()
    if not evaluation:
        print("No evaluation data found", file=sys.stderr)
        return 1

    # Generate badges
    badges = generate_badges(evaluation)

    if not badges:
        print("Could not generate badges from evaluation data", file=sys.stderr)
        return 1

    # Handle specific key request
    if args.key:
        if args.key in badges:
            print(badges[args.key])
        else:
            print(f"Badge key '{args.key}' not found", file=sys.stderr)
            return 1
        return 0

    # Format output
    if args.format == "json":
        output = json.dumps(badges, indent=2)
    elif args.format == "url":
        output = "\n".join(f"{k}: {v}" for k, v in badges.items())
    else:  # markdown
        output = format_markdown(badges)

    # Output or save
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Badges saved to: {args.output}")
    else:
        print(output)

    # Update README if requested
    if args.update_readme:
        markdown = format_markdown(badges)
        if update_readme(markdown):
            print("README.md updated with new badges")
        else:
            print("Failed to update README.md", file=sys.stderr)
            return 1

    # Also save JSON for reference
    save_badge_json(badges)

    return 0


if __name__ == "__main__":
    sys.exit(main())