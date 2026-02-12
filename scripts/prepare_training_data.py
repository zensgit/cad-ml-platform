#!/usr/bin/env python3
"""
准备训练数据集脚本

将DXF文件按大类分组，生成训练数据集
"""

import json
import os
import re
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# 源数据目录
SOURCE_DIR = Path("/Users/huazhou/Downloads/训练图纸/训练图纸_dxf")
# 输出目录
OUTPUT_DIR = Path("/Users/huazhou/Downloads/Github/cad-ml-platform/data/training")

# 大类映射规则
CATEGORY_RULES = {
    "组件": ["组件"],
    "法兰": ["法兰"],
    "轴承": ["轴承", "轴承座"],
    "弹簧": ["弹簧"],
    "阀体": ["阀体", "阀门"],
    "罐体": ["罐体", "筒体", "封头"],
    "其他": []  # 默认类别
}


def extract_part_name(filename: str) -> str:
    """从文件名提取部件名称"""
    # 去除版本后缀 v1, v2, v3
    name = re.sub(r'v\d+(-\w+)?\.dxf$', '', filename, flags=re.IGNORECASE)
    # 去除.dxf后缀
    name = re.sub(r'\.dxf$', '', name, flags=re.IGNORECASE)
    # 提取中文部分
    match = re.search(r'[\u4e00-\u9fff].*', name)
    if match:
        return match.group().strip()
    return name


def classify_part(part_name: str) -> str:
    """将部件分类到大类"""
    for category, keywords in CATEGORY_RULES.items():
        if category == "其他":
            continue
        for keyword in keywords:
            if keyword in part_name:
                return category
    return "其他"


def prepare_dataset() -> Dict[str, List[Tuple[str, str]]]:
    """准备数据集，返回 {类别: [(文件路径, 部件名)]}"""
    dataset = defaultdict(list)

    if not SOURCE_DIR.exists():
        print(f"错误: 源目录不存在 {SOURCE_DIR}")
        return dataset

    for dxf_file in SOURCE_DIR.glob("*.dxf"):
        filename = dxf_file.name

        # 跳过比较文件
        if filename.startswith("比较_"):
            continue

        part_name = extract_part_name(filename)
        category = classify_part(part_name)
        dataset[category].append((str(dxf_file), part_name))

    return dataset


def create_training_structure(dataset: Dict[str, List[Tuple[str, str]]]):
    """创建训练目录结构并复制文件"""
    # 清理并创建输出目录
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # 创建标签映射
    labels = sorted(dataset.keys())
    label_map = {label: idx for idx, label in enumerate(labels)}

    # 保存标签映射
    with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump({
            "label_to_id": label_map,
            "id_to_label": {v: k for k, v in label_map.items()}
        }, f, ensure_ascii=False, indent=2)

    # 创建数据清单
    manifest = []

    for category, files in dataset.items():
        category_dir = OUTPUT_DIR / category
        category_dir.mkdir(exist_ok=True)

        for src_path, part_name in files:
            filename = Path(src_path).name
            dst_path = category_dir / filename

            # 复制文件
            shutil.copy2(src_path, dst_path)

            # 添加到清单
            manifest.append({
                "file": str(dst_path.relative_to(OUTPUT_DIR)),
                "category": category,
                "label_id": label_map[category],
                "part_name": part_name
            })

    # 保存清单
    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return labels, manifest


def print_summary(dataset: Dict[str, List], labels: List[str], manifest: List):
    """打印数据集统计"""
    print("=" * 60)
    print("训练数据集准备完成")
    print("=" * 60)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"总类别数: {len(labels)}")
    print(f"总文件数: {len(manifest)}")

    print("\n各类别统计:")
    print("-" * 40)
    for category in labels:
        count = len(dataset[category])
        print(f"  {category}: {count}个文件")

    print("\n生成的文件:")
    print(f"  - labels.json: 标签映射")
    print(f"  - manifest.json: 数据清单")
    print(f"  - <类别>/: 各类别DXF文件")


def main():
    print("开始准备训练数据...")

    # 准备数据集
    dataset = prepare_dataset()

    if not dataset:
        print("错误: 未找到任何DXF文件")
        return

    # 创建目录结构
    labels, manifest = create_training_structure(dataset)

    # 打印统计
    print_summary(dataset, labels, manifest)

    print("\n✅ 数据准备完成！")


if __name__ == "__main__":
    main()
