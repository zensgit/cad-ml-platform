#!/usr/bin/env python3
"""
修正manifest.json中的错误标注
"""

import json
from pathlib import Path

MANIFEST_PATH = Path("data/training_v7/manifest.json")

# 需要修正的标注（基于人工审核）
CORRECTIONS = {
    "其他/old_0033.dxf": "轴类",      # 明显是轴类
    "其他/old_0085.dxf": "传动件",    # 明显是传动件（手轮/皮带轮）
    "其他/new_0208.dxf": "壳体类",    # 明显是壳体类（端盖）
    # old_0008.dxf 保留原标注（存疑）
}

def main():
    print("=" * 60)
    print("修正manifest.json中的错误标注")
    print("=" * 60)

    # 读取manifest
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    print(f"总样本数: {len(manifest)}")

    # 修正标注
    corrected_count = 0
    for item in manifest:
        file_name = item["file"]
        if file_name in CORRECTIONS:
            old_cat = item["category"]
            new_cat = CORRECTIONS[file_name]
            print(f"\n修正: {file_name}")
            print(f"  {old_cat} → {new_cat}")
            item["category"] = new_cat
            item["corrected"] = True
            item["original_category"] = old_cat
            corrected_count += 1

    print(f"\n共修正 {corrected_count} 个标注")

    # 备份原文件
    backup_path = MANIFEST_PATH.with_suffix('.json.bak')
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(json.load(open(MANIFEST_PATH, 'r', encoding='utf-8')), f, ensure_ascii=False, indent=2)
    print(f"原文件已备份到: {backup_path}")

    # 保存修正后的manifest
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"已保存修正后的manifest: {MANIFEST_PATH}")

    # 统计各类别数量
    print("\n修正后各类别统计:")
    from collections import Counter
    cat_counts = Counter(item["category"] for item in manifest)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
