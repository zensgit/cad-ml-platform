#!/usr/bin/env python3
"""
检查5个错误样本的详细特征，判断是否是标注问题
并创建V6+V14超级集成模型
"""

import json
import logging
import sys
import io
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REAL_DATA_DIR = Path("data/training_v7")
MODEL_DIR = Path("models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

IMG_SIZE = 128


def analyze_dxf_detailed(dxf_path: str):
    """详细分析DXF文件内容"""
    try:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        entity_types = Counter()
        layer_names = []
        circle_count = 0
        arc_count = 0
        line_count = 0

        for entity in msp:
            etype = entity.dxftype()
            entity_types[etype] += 1
            if hasattr(entity.dxf, 'layer'):
                layer_names.append(entity.dxf.layer)

            if etype == "LINE":
                line_count += 1
            elif etype == "CIRCLE":
                circle_count += 1
            elif etype == "ARC":
                arc_count += 1

        total = sum(entity_types.values())
        unique_layers = set(layer_names)

        return {
            "total_entities": total,
            "entity_types": dict(entity_types),
            "layers": list(unique_layers),
            "layer_count": len(unique_layers),
            "line_ratio": line_count / max(total, 1),
            "circle_ratio": circle_count / max(total, 1),
            "arc_ratio": arc_count / max(total, 1),
            "curved_ratio": (circle_count + arc_count) / max(total, 1)
        }
    except Exception as e:
        return {"error": str(e)}


def get_class_typical_features():
    """获取各类别的典型特征"""
    manifest_path = REAL_DATA_DIR / "manifest.json"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    class_features = {}
    for cat in ["轴类", "传动件", "壳体类", "连接件", "其他"]:
        class_features[cat] = {
            "line_ratios": [],
            "circle_ratios": [],
            "arc_ratios": [],
            "curved_ratios": [],
            "total_entities": []
        }

    for item in manifest:
        file_path = REAL_DATA_DIR / item["file"]
        category = item["category"]

        info = analyze_dxf_detailed(str(file_path))
        if "error" in info:
            continue

        class_features[category]["line_ratios"].append(info["line_ratio"])
        class_features[category]["circle_ratios"].append(info["circle_ratio"])
        class_features[category]["arc_ratios"].append(info["arc_ratio"])
        class_features[category]["curved_ratios"].append(info["curved_ratio"])
        class_features[category]["total_entities"].append(info["total_entities"])

    # 计算均值和标准差
    for cat in class_features:
        for key in ["line_ratios", "circle_ratios", "arc_ratios", "curved_ratios", "total_entities"]:
            values = class_features[cat][key]
            if values:
                class_features[cat][f"{key}_mean"] = np.mean(values)
                class_features[cat][f"{key}_std"] = np.std(values)

    return class_features


def main():
    logger.info("=" * 60)
    logger.info("检查5个错误样本")
    logger.info("=" * 60)

    # 5个错误样本
    error_samples = [
        {"file": "其他/old_0033.dxf", "true": "其他", "pred": "轴类"},
        {"file": "其他/old_0085.dxf", "true": "其他", "pred": "传动件"},
        {"file": "其他/old_0086.dxf", "true": "其他", "pred": "壳体类"},
        {"file": "连接件/old_0008.dxf", "true": "连接件", "pred": "传动件"},
        {"file": "其他/new_0208.dxf", "true": "其他", "pred": "壳体类"},
    ]

    logger.info("\n获取各类别典型特征...")
    class_features = get_class_typical_features()

    logger.info("\n各类别特征统计:")
    for cat in ["轴类", "传动件", "壳体类", "连接件", "其他"]:
        f = class_features[cat]
        logger.info(f"\n{cat}:")
        logger.info(f"  LINE比例: {f.get('line_ratios_mean', 0):.3f} ± {f.get('line_ratios_std', 0):.3f}")
        logger.info(f"  曲线比例: {f.get('curved_ratios_mean', 0):.3f} ± {f.get('curved_ratios_std', 0):.3f}")
        logger.info(f"  实体数: {f.get('total_entities_mean', 0):.1f} ± {f.get('total_entities_std', 0):.1f}")

    logger.info("\n" + "=" * 60)
    logger.info("分析错误样本")
    logger.info("=" * 60)

    for sample in error_samples:
        file_path = REAL_DATA_DIR / sample["file"]
        info = analyze_dxf_detailed(str(file_path))

        logger.info(f"\n文件: {sample['file']}")
        logger.info(f"  标注: {sample['true']} → 预测: {sample['pred']}")

        if "error" not in info:
            logger.info(f"  实体总数: {info['total_entities']}")
            logger.info(f"  LINE比例: {info['line_ratio']:.3f}")
            logger.info(f"  曲线比例: {info['curved_ratio']:.3f}")
            logger.info(f"  图层: {info['layers'][:5]}...")  # 前5个图层

            # 判断更接近哪个类别
            true_cat = sample["true"]
            pred_cat = sample["pred"]

            true_line_mean = class_features[true_cat].get('line_ratios_mean', 0)
            pred_line_mean = class_features[pred_cat].get('line_ratios_mean', 0)
            true_curved_mean = class_features[true_cat].get('curved_ratios_mean', 0)
            pred_curved_mean = class_features[pred_cat].get('curved_ratios_mean', 0)

            # 计算距离
            true_dist = abs(info['line_ratio'] - true_line_mean) + abs(info['curved_ratio'] - true_curved_mean)
            pred_dist = abs(info['line_ratio'] - pred_line_mean) + abs(info['curved_ratio'] - pred_curved_mean)

            logger.info(f"  距离标注类({true_cat}): {true_dist:.3f}")
            logger.info(f"  距离预测类({pred_cat}): {pred_dist:.3f}")

            if pred_dist < true_dist:
                logger.info(f"  ⚠️ 特征更接近预测类，可能是标注错误!")
            else:
                logger.info(f"  ✓ 特征更接近标注类，是困难样本")

    # 检查重复文件
    logger.info("\n" + "=" * 60)
    logger.info("检查是否有重复文件")
    logger.info("=" * 60)

    file1 = REAL_DATA_DIR / "其他/old_0086.dxf"
    file2 = REAL_DATA_DIR / "其他/new_0208.dxf"

    info1 = analyze_dxf_detailed(str(file1))
    info2 = analyze_dxf_detailed(str(file2))

    if info1.get("total_entities") == info2.get("total_entities") and \
       info1.get("line_ratio") == info2.get("line_ratio"):
        logger.info("old_0086.dxf 和 new_0208.dxf 几何特征相同，可能是重复文件")
    else:
        logger.info("两个文件不是重复的")


if __name__ == "__main__":
    main()
