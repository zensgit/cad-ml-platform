#!/usr/bin/env python3
"""Summarize Graph2D soft-override review templates into CSV reports."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

DEFAULT_REVIEW = (
    "reports/experiments/20260123/"
    "soft_override_calibrated_added_review_template_20260124.csv"
)
DEFAULT_SUMMARY = (
    "reports/experiments/20260123/"
    "soft_override_calibrated_added_review_summary_20260124.csv"
)
DEFAULT_CORRECT = (
    "reports/experiments/20260123/"
    "soft_override_calibrated_added_correct_label_counts_20260124.csv"
)

AGREE_VALUES = {"yes", "y", "true", "1", "是", "同意", "agree"}
DISAGREE_VALUES = {"no", "n", "false", "0", "否", "不同意", "disagree"}


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Review template not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarize(rows: list[dict[str, str]]) -> dict[str, str]:
    reviewed = 0
    agree_count = 0
    disagree_count = 0
    correct_label_counts: Counter[str] = Counter()

    for row in rows:
        value = (row.get("agree_with_graph2d") or "").strip().lower()
        if value:
            reviewed += 1
            if value in AGREE_VALUES:
                agree_count += 1
            elif value in DISAGREE_VALUES:
                disagree_count += 1
        correct_label = (row.get("correct_label") or "").strip()
        if correct_label:
            correct_label_counts[correct_label] += 1

    total = len(rows)
    unknown = max(0, total - agree_count - disagree_count)
    agree_rate = (agree_count / reviewed) if reviewed else 0.0
    return {
        "summary": {
            "total": str(total),
            "reviewed": str(reviewed),
            "agree_with_graph2d": str(agree_count),
            "disagree_with_graph2d": str(disagree_count),
            "unknown": str(unknown),
            "agree_rate": f"{agree_rate:.4f}",
            "suggested_min_conf": "",
            "notes": "",
        },
        "correct_labels": correct_label_counts,
    }


def _write_summary(path: Path, summary: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "total",
                "reviewed",
                "agree_with_graph2d",
                "disagree_with_graph2d",
                "unknown",
                "agree_rate",
                "suggested_min_conf",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(summary)


def _write_correct_labels(path: Path, counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["correct_label", "count"])
        writer.writeheader()
        for label, count in counts.most_common():
            writer.writerow({"correct_label": label, "count": str(count)})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize soft-override review template into CSV reports."
    )
    parser.add_argument("--review-template", default=DEFAULT_REVIEW)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY)
    parser.add_argument("--correct-labels-out", default=DEFAULT_CORRECT)
    args = parser.parse_args()

    review_path = Path(args.review_template)
    summary_path = Path(args.summary_out)
    correct_path = Path(args.correct_labels_out)

    rows = _load_rows(review_path)
    summary_data = _summarize(rows)
    _write_summary(summary_path, summary_data["summary"])
    _write_correct_labels(correct_path, summary_data["correct_labels"])

    print(f"Review template: {review_path}")
    print(f"Summary written: {summary_path}")
    print(f"Correct labels written: {correct_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
