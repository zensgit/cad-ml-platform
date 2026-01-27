#!/usr/bin/env python3
"""Normalize DXF label manifest by mapping labels to canonical buckets."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

LABEL_MAP = {
    "罐体部分": "罐体",
    "上封头组件": "罐体",
    "下锥体组件": "罐体",
    "上筒体组件": "罐体",
    "下筒体组件": "罐体",
    "再沸器": "设备",
    "汽水分离器": "设备",
    "自动进料装置": "设备",
    "电加热箱": "设备",
    "真空组件": "设备",
    "出料正压隔离器": "设备",
    "拖车": "设备",
    "管束": "设备",
    "阀体": "设备",
    "蜗轮蜗杆传动出料机构": "传动件",
    "旋转组件": "传动件",
    "轴头组件": "传动件",
    "搅拌桨组件": "传动件",
    "搅拌轴组件": "传动件",
    "搅拌器组件": "传动件",
    "手轮组件": "传动件",
    "拖轮组件": "传动件",
    "液压开盖组件": "传动件",
    "侧推料组件": "传动件",
    "轴向定位轴承": "轴承件",
    "轴承座": "轴承件",
    "下轴承支架组件": "轴承件",
    "短轴承座(盖)": "轴承件",
    "支承座": "轴承件",
    "超声波法兰": "法兰",
    "出料凸缘": "法兰",
    "对接法兰": "法兰",
    "人孔法兰": "法兰",
    "连接法兰(大)": "法兰",
    "保护罩组件": "罩盖件",
    "搅拌减速机机罩": "罩盖件",
    "防爆视灯组件": "罩盖件",
    "下封板": "罩盖件",
    "过滤托架": "过滤组件",
    "过滤芯组件": "过滤组件",
    "捕集器组件": "过滤组件",
    "捕集口": "开孔件",
    "人孔": "开孔件",
    "罐体支腿": "支撑件",
    "底板": "支撑件",
    "调节螺栓": "紧固件",
    "扭转弹簧": "弹簧",
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row]


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize DXF label manifest CSV.")
    parser.add_argument("--input-csv", required=True, help="Input manifest CSV")
    parser.add_argument("--output-csv", required=True, help="Output normalized CSV")
    parser.add_argument(
        "--default-label",
        default="other",
        help="Fallback label for unmapped entries",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when encountering unmapped labels",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    rows = _load_rows(input_path)
    if not rows:
        raise SystemExit("No rows found in input manifest.")

    counts = Counter()
    normalized_rows: list[dict[str, str]] = []
    unmapped = set()

    for row in rows:
        label = (row.get("label_cn") or "").strip()
        if not label:
            continue
        mapped = LABEL_MAP.get(label)
        if mapped is None:
            unmapped.add(label)
            if args.strict:
                raise SystemExit(f"Unmapped label: {label}")
            mapped = args.default_label
        new_row = dict(row)
        new_row["label_raw"] = label
        new_row["label_cn"] = mapped
        normalized_rows.append(new_row)
        counts[mapped] += 1

    fieldnames = list(rows[0].keys())
    if "label_raw" not in fieldnames:
        fieldnames.append("label_raw")

    _write_rows(output_path, fieldnames, normalized_rows)

    print(f"rows_in={len(rows)}")
    print(f"rows_out={len(normalized_rows)}")
    print(f"labels_out={len(counts)}")
    print(f"output={output_path}")
    for label, count in counts.most_common():
        print(f"{label},{count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
