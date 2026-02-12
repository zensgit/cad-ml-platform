#!/usr/bin/env python3
"""
准备训练数据集 V5

合并为5个主要类别，去掉样本过少的类别
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

# 数据源
SOURCE = Path("data/training_merged")
OUTPUT_DIR = Path("data/training_v5")

# 5类映射：将8类合并为5类
CATEGORY_MERGE = {
    # 保留的主要类别
    "轴类": ["轴类"],
    "壳体类": ["壳体类"],
    "传动件": ["传动件"],
    "连接件": ["连接件"],
    # 合并小类到"其他"
    "其他": ["其他", "叶轮类", "弹簧类", "装配图"],
}

# 反向映射
OLD_TO_NEW = {}
for new_cat, old_cats in CATEGORY_MERGE.items():
    for old_cat in old_cats:
        OLD_TO_NEW[old_cat] = new_cat


def prepare_dataset():
    """准备5类数据集"""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # 加载原数据
    manifest = json.loads((SOURCE / "manifest.json").read_text())

    all_files = defaultdict(list)
    for item in manifest:
        old_cat = item["category"]
        new_cat = OLD_TO_NEW.get(old_cat, "其他")
        src_path = SOURCE / item["file"]
        if src_path.exists():
            all_files[new_cat].append((str(src_path), old_cat))

    # 创建标签映射
    labels = sorted(all_files.keys())
    label_map = {label: idx for idx, label in enumerate(labels)}

    with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump({
            "label_to_id": label_map,
            "id_to_label": {v: k for k, v in label_map.items()},
            "version": "v5"
        }, f, ensure_ascii=False, indent=2)

    # 复制文件
    new_manifest = []
    for category, files in all_files.items():
        category_dir = OUTPUT_DIR / category
        category_dir.mkdir(exist_ok=True)

        for src_path, original_cat in files:
            filename = Path(src_path).name
            dst_path = category_dir / filename

            counter = 1
            while dst_path.exists():
                stem = Path(src_path).stem
                suffix = Path(src_path).suffix
                dst_path = category_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.copy2(src_path, dst_path)

            new_manifest.append({
                "file": str(dst_path.relative_to(OUTPUT_DIR)),
                "category": category,
                "label_id": label_map[category],
                "original_category": original_cat
            })

    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(new_manifest, f, ensure_ascii=False, indent=2)

    # 打印统计
    print("=" * 50)
    print("5类数据集准备完成")
    print("=" * 50)
    print(f"总类别: {len(labels)}")
    print(f"总文件: {len(new_manifest)}")
    print("\n各类别统计:")
    for cat in sorted(labels, key=lambda x: len(all_files[x]), reverse=True):
        print(f"  {cat}: {len(all_files[cat])}个")


if __name__ == "__main__":
    prepare_dataset()
