#!/usr/bin/env python3
"""批量复核 soft_override 结果，渲染 DXF 并基于文件名推断正确标签。"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 标签同义词表路径
SYNONYMS_PATH = ROOT / "data/knowledge/label_synonyms_template.json"

# 从文件名提取零件名称的模式
# 例如: J2925001-01人孔v2.dxf -> 人孔
# BTJ01239901522-00拖轮组件v1.dxf -> 拖轮组件
PART_NAME_PATTERNS = [
    # 匹配 "数字-数字中文名称v数字" 格式
    r"[A-Z]*\d+[-\d]*[A-Z]*[-\d]*(.+?)v\d+\.dxf$",
    # 匹配比较文件格式
    r"比较_.*?[-\d]+(.+?)v\d+\s+vs\s+",
]


def load_synonyms(path: Path) -> Dict[str, List[str]]:
    """加载标签同义词表。"""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, list)}


def build_label_matcher(synonyms: Dict[str, List[str]]) -> Dict[str, str]:
    """构建标签匹配器：零件名 -> 标准标签。"""
    matcher: Dict[str, str] = {}
    for label, aliases in synonyms.items():
        # 标准标签本身
        matcher[label.lower()] = label
        # 所有别名
        for alias in aliases:
            matcher[alias.lower()] = label
    return matcher


def extract_part_name_from_filename(filename: str) -> Optional[str]:
    """从文件名提取零件名称。"""
    basename = Path(filename).stem

    # 移除版本后缀
    basename = re.sub(r"v\d+$", "", basename, flags=re.IGNORECASE).strip()

    # 尝试多种模式提取中文部分
    # 模式1: 编号-编号-零件名
    match = re.search(r"[-\d]+([^\d\-][^v]*?)$", basename, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 模式2: 提取最后的中文部分
    match = re.search(r"([\u4e00-\u9fa5()（）]+)$", basename)
    if match:
        return match.group(1).strip()

    return None


def match_label(part_name: str, matcher: Dict[str, str]) -> Optional[str]:
    """匹配零件名到标准标签。"""
    if not part_name:
        return None

    # 直接匹配
    key = part_name.lower()
    if key in matcher:
        return matcher[key]

    # 部分匹配（零件名包含标签或标签包含零件名）
    for label_key, label in matcher.items():
        if label_key in key or key in label_key:
            return label

    return None


def analyze_file(file_path: str, synonyms: Dict[str, List[str]], matcher: Dict[str, str]) -> Dict:
    """分析单个文件，返回推荐的正确标签。"""
    filename = Path(file_path).name
    part_name = extract_part_name_from_filename(filename)
    matched_label = match_label(part_name, matcher) if part_name else None

    return {
        "file": file_path,
        "filename": filename,
        "extracted_part_name": part_name,
        "matched_label": matched_label,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="批量复核 soft_override 结果")
    parser.add_argument(
        "--input",
        default="reports/experiments/20260123/soft_override_calibrated_added_review_template_20260124.csv",
        help="输入 CSV 文件路径",
    )
    parser.add_argument(
        "--output",
        default="reports/experiments/20260123/soft_override_reviewed_20260124.csv",
        help="输出 CSV 文件路径",
    )
    parser.add_argument(
        "--render-dir",
        default="reports/experiments/20260123/renders",
        help="渲染图片输出目录",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="是否渲染 DXF 到 PNG",
    )
    parser.add_argument(
        "--reviewer",
        default="auto+human",
        help="复核人标识",
    )
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output
    render_dir = ROOT / args.render_dir

    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return 1

    # 加载同义词表
    synonyms = load_synonyms(SYNONYMS_PATH)
    matcher = build_label_matcher(synonyms)
    print(f"已加载 {len(synonyms)} 个标签类别")

    # 读取输入 CSV
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    print(f"读取 {len(rows)} 条记录")

    # 确保必要的列存在
    for col in ["agree_with_graph2d", "correct_label", "notes", "reviewer", "review_date"]:
        if col not in fieldnames:
            fieldnames.append(col)

    # 渲染目录
    if args.render:
        render_dir.mkdir(parents=True, exist_ok=True)

    # 分析每个文件
    review_date = datetime.now().strftime("%Y-%m-%d")

    for i, row in enumerate(rows):
        file_path = row.get("file", "")
        filename = Path(file_path).name
        graph2d_label = row.get("graph2d_label", "")
        graph2d_conf = float(row.get("graph2d_confidence", 0) or 0)

        # 分析文件
        analysis = analyze_file(file_path, synonyms, matcher)
        extracted = analysis["extracted_part_name"]
        matched = analysis["matched_label"]

        # 判断是否同意 Graph2D 预测
        if matched:
            agree = "Y" if matched == graph2d_label else "N"
            correct_label = matched
            notes = f"从文件名提取: {extracted}"
        else:
            # 无法自动匹配，需人工确认
            agree = "?"
            correct_label = ""
            notes = f"无法自动匹配, 提取到: {extracted}"

        # 低置信度标记
        if graph2d_conf < 0.3:
            notes += f"; 低置信度({graph2d_conf:.2f})"

        row["reviewer"] = args.reviewer
        row["review_date"] = review_date
        row["agree_with_graph2d"] = agree
        row["correct_label"] = correct_label
        row["notes"] = notes

        print(f"[{i+1:2d}] {filename}")
        print(f"     Graph2D: {graph2d_label} ({graph2d_conf:.3f})")
        print(f"     提取: {extracted} -> 匹配: {matched}")
        print(f"     结论: agree={agree}, correct={correct_label}")
        print()

        # 渲染 DXF
        if args.render and Path(file_path).exists():
            try:
                from src.core.dedupcad_precision.cad_pipeline import render_dxf_to_png

                png_path = render_dir / f"{Path(filename).stem}.png"
                with open(file_path, "rb") as f:
                    dxf_bytes = f.read()
                png_bytes = render_dxf_to_png(dxf_bytes, filename)
                png_path.write_bytes(png_bytes)
                print(f"     已渲染: {png_path}")
            except Exception as e:
                print(f"     渲染失败: {e}")

    # 写入输出 CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n已写入: {output_path}")

    # 统计
    agree_count = sum(1 for r in rows if r.get("agree_with_graph2d") == "Y")
    disagree_count = sum(1 for r in rows if r.get("agree_with_graph2d") == "N")
    unknown_count = sum(1 for r in rows if r.get("agree_with_graph2d") == "?")

    print(f"\n统计:")
    print(f"  同意 Graph2D: {agree_count}")
    print(f"  不同意 Graph2D: {disagree_count}")
    print(f"  需人工确认: {unknown_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
