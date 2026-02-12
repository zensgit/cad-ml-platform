#!/usr/bin/env python3
"""
合并数据集并训练

合并 training (109个) + training_v3 (186个) = 295个样本
统一为 8 个类别
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

# 数据源
SOURCE1 = Path("data/training")      # 原始109个
SOURCE2 = Path("data/training_v3")   # 新增186个
OUTPUT_DIR = Path("data/training_merged")

# 类别映射：将旧类别映射到新的8类
OLD_TO_NEW = {
    # training 原有类别
    "组件": "其他",
    "其他": "其他",
    "法兰": "壳体类",
    "罐体": "壳体类",
    "轴承": "轴类",
    "弹簧": "弹簧类",
    "阀体": "壳体类",
    # training_v3 类别 (已经是8类)
    "传动件": "传动件",
    "叶轮类": "叶轮类",
    "壳体类": "壳体类",
    "装配图": "装配图",
    "轴类": "轴类",
    "连接件": "连接件",
}


def merge_datasets():
    """合并数据集"""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    all_files = defaultdict(list)

    # 加载 SOURCE1
    if SOURCE1.exists():
        manifest1 = json.loads((SOURCE1 / "manifest.json").read_text())
        for item in manifest1:
            old_cat = item["category"]
            new_cat = OLD_TO_NEW.get(old_cat, "其他")
            src_path = SOURCE1 / item["file"]
            if src_path.exists():
                all_files[new_cat].append((str(src_path), f"v1:{old_cat}"))
        print(f"从 training 加载 {len(manifest1)} 个文件")

    # 加载 SOURCE2
    if SOURCE2.exists():
        manifest2 = json.loads((SOURCE2 / "manifest.json").read_text())
        for item in manifest2:
            cat = item["category"]
            new_cat = OLD_TO_NEW.get(cat, cat)  # 大部分已经是新类别
            src_path = SOURCE2 / item["file"]
            if src_path.exists():
                all_files[new_cat].append((str(src_path), f"v3:{item.get('original_folder', cat)}"))
        print(f"从 training_v3 加载 {len(manifest2)} 个文件")

    # 创建标签映射
    labels = sorted(all_files.keys())
    label_map = {label: idx for idx, label in enumerate(labels)}

    with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump({
            "label_to_id": label_map,
            "id_to_label": {v: k for k, v in label_map.items()},
            "version": "merged"
        }, f, ensure_ascii=False, indent=2)

    # 复制文件
    manifest = []
    for category, files in all_files.items():
        category_dir = OUTPUT_DIR / category
        category_dir.mkdir(exist_ok=True)

        for src_path, source_info in files:
            filename = Path(src_path).name
            dst_path = category_dir / filename

            counter = 1
            while dst_path.exists():
                stem = Path(src_path).stem
                suffix = Path(src_path).suffix
                dst_path = category_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.copy2(src_path, dst_path)

            manifest.append({
                "file": str(dst_path.relative_to(OUTPUT_DIR)),
                "category": category,
                "label_id": label_map[category],
                "source": source_info
            })

    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 打印统计
    print("\n" + "=" * 50)
    print("合并数据集完成")
    print("=" * 50)
    print(f"总类别: {len(labels)}")
    print(f"总文件: {len(manifest)}")
    print("\n各类别统计:")
    for cat in sorted(labels, key=lambda x: len(all_files[x]), reverse=True):
        print(f"  {cat}: {len(all_files[cat])}个")


if __name__ == "__main__":
    merge_datasets()
