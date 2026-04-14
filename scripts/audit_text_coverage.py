#!/usr/bin/env python3
"""Audit text extraction coverage and keyword hit rates across the dataset.

Usage:
    python scripts/audit_text_coverage.py \
        --manifest data/graph_cache/cache_manifest.csv \
        --sample 200 \
        --output docs/design/B5_1_TEXT_AUDIT.md
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ml.text_extractor import extract_text_from_path
from src.ml.text_classifier import TextContentClassifier


def audit(manifest_csv: str, sample_per_class: int) -> dict:
    with open(manifest_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["taxonomy_v2_class"]].append(r)

    clf = TextContentClassifier()
    class_stats: dict[str, dict] = {}

    total_files = 0
    total_with_text = 0
    total_with_kw_hit = 0
    total_correct_top1 = 0

    for cls, cls_rows in sorted(by_class.items()):
        sample = cls_rows if len(cls_rows) <= sample_per_class else \
                 random.Random(42).sample(cls_rows, sample_per_class)

        n_total = len(sample)
        n_has_text = 0
        n_kw_hit = 0
        n_correct_top1 = 0
        text_lengths = []

        for r in sample:
            text = extract_text_from_path(r["file_path"])
            text_len = len(text.strip())
            text_lengths.append(text_len)

            if text_len >= 4:
                n_has_text += 1
                probs = clf.predict_probs(text)
                if probs:
                    n_kw_hit += 1
                    top = max(probs, key=probs.get)
                    if top == cls:
                        n_correct_top1 += 1

        class_stats[cls] = {
            "total": n_total,
            "has_text": n_has_text,
            "kw_hit": n_kw_hit,
            "correct_top1": n_correct_top1,
            "text_coverage": n_has_text / max(n_total, 1),
            "kw_coverage": n_kw_hit / max(n_total, 1),
            "text_precision": n_correct_top1 / max(n_kw_hit, 1),
            "avg_text_len": sum(text_lengths) / max(len(text_lengths), 1),
        }

        total_files += n_total
        total_with_text += n_has_text
        total_with_kw_hit += n_kw_hit
        total_correct_top1 += n_correct_top1

        print(
            f"{cls:16s} text={n_has_text}/{n_total} "
            f"({n_has_text/max(n_total,1):.0%}) "
            f"kw_hit={n_kw_hit}/{n_total} "
            f"({n_kw_hit/max(n_total,1):.0%}) "
            f"prec={n_correct_top1/max(n_kw_hit,1):.0%}",
            flush=True,
        )

    overall = {
        "total_files": total_files,
        "pct_with_text": total_with_text / max(total_files, 1),
        "pct_with_kw_hit": total_with_kw_hit / max(total_files, 1),
        "text_precision": total_correct_top1 / max(total_with_kw_hit, 1),
        "class_stats": class_stats,
    }
    return overall


def write_report(stats: dict, path: str) -> None:
    cs = stats["class_stats"]
    lines = [
        "# B5.1 文字覆盖率审计报告",
        "",
        f"**总文件数**: {stats['total_files']}  ",
        f"**有文字比例**: {stats['pct_with_text']:.1%}  ",
        f"**关键词命中比例**: {stats['pct_with_kw_hit']:.1%}  ",
        f"**文字分类 Top-1 精度**: {stats['text_precision']:.1%}（仅统计有命中样本）  ",
        "",
        "## 各类别详情",
        "",
        "| 类别 | 总计 | 有文字 | 覆盖率 | 关键词命中 | 命中率 | Top-1 精度 |",
        "|------|------|--------|--------|-----------|--------|-----------|",
    ]
    for cls, s in sorted(cs.items(), key=lambda x: -x[1]["text_coverage"]):
        lines.append(
            f"| {cls} | {s['total']} | {s['has_text']} | {s['text_coverage']:.0%} "
            f"| {s['kw_hit']} | {s['kw_coverage']:.0%} | {s['text_precision']:.0%} |"
        )
    lines += [
        "",
        "## 总体结论",
        "",
        f"- **文字覆盖率 {stats['pct_with_text']:.1%}**：{'✓ 达标（≥60%）' if stats['pct_with_text'] >= 0.6 else '⚠️ 低于目标60%'}",
        f"- **关键词命中率 {stats['pct_with_kw_hit']:.1%}**：{'✓ 达标（≥40%）' if stats['pct_with_kw_hit'] >= 0.4 else '⚠️ 低于目标40%'}",
        f"- **文字分类精度 {stats['text_precision']:.1%}**：{'✓ 高置信信号' if stats['text_precision'] >= 0.5 else '⚠️ 精度偏低，建议扩充关键词词典'}",
        "",
        "*报告生成: 2026-04-14*",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DXF text coverage.")
    parser.add_argument("--manifest", default="data/graph_cache/cache_manifest.csv")
    parser.add_argument("--sample", type=int, default=20,
                        help="Max samples per class (default 20)")
    parser.add_argument("--output", default="docs/design/B5_1_TEXT_AUDIT.md")
    args = parser.parse_args()

    print(f"Auditing text coverage (up to {args.sample} samples/class)...\n")
    stats = audit(args.manifest, args.sample)
    print(f"\nOverall text coverage : {stats['pct_with_text']:.1%}")
    print(f"Keyword hit rate      : {stats['pct_with_kw_hit']:.1%}")
    print(f"Text classifier prec  : {stats['text_precision']:.1%}")
    write_report(stats, args.output)


if __name__ == "__main__":
    main()
