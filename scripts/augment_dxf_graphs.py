#!/usr/bin/env python3
"""Graph-level data augmentation for cached DXF graph tensors.

Augments rare classes by applying topology-preserving transformations on
the pre-processed .pt graph files produced by preprocess_dxf_to_graphs.py.

Augmentation operations (all class-agnostic):
  - edge_dropout   : randomly remove p% of edges
  - feature_noise  : add Gaussian noise to node features
  - node_mask      : randomly zero out q% of nodes
  - edge_attr_noise: add Gaussian noise to edge attributes

Strategy: augment classes below a sample threshold to reach target count.

Usage:
    python scripts/augment_dxf_graphs.py \
        --manifest data/graph_cache/cache_manifest.csv \
        --aug-cache-dir data/graph_cache_aug/ \
        --output-manifest data/graph_cache_aug/cache_manifest_aug.csv \
        --target-samples 80 \
        --bearing-target 200
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch


def augment_graph(
    data: dict,
    edge_dropout_p: float = 0.10,
    feature_noise_sigma: float = 0.02,
    node_mask_p: float = 0.05,
    edge_attr_noise_sigma: float = 0.02,
    seed: int | None = None,
) -> dict:
    """Apply random topology-preserving augmentation to a graph dict.

    Args:
        data: dict with keys x, edge_index, edge_attr (optional), label
        edge_dropout_p: fraction of edges to randomly remove
        feature_noise_sigma: std of Gaussian noise added to node features
        node_mask_p: fraction of node feature vectors to zero out
        edge_attr_noise_sigma: std of Gaussian noise added to edge attributes
        seed: optional RNG seed for reproducibility

    Returns:
        Augmented copy of data (tensors cloned, not in-place).
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    x = data["x"].clone().float()
    edge_index = data["edge_index"].clone()
    edge_attr = data.get("edge_attr")
    if edge_attr is not None:
        edge_attr = edge_attr.clone().float()

    # --- Node feature Gaussian noise ---
    if feature_noise_sigma > 0 and x.numel() > 0:
        x = x + torch.randn_like(x) * feature_noise_sigma

    # --- Random node masking (zero-out) ---
    if node_mask_p > 0 and x.size(0) > 1:
        mask = torch.rand(x.size(0)) < node_mask_p
        x[mask] = 0.0

    # --- Edge dropout ---
    if edge_dropout_p > 0 and edge_index.size(1) > 0:
        keep = torch.rand(edge_index.size(1)) > edge_dropout_p
        edge_index = edge_index[:, keep]
        if edge_attr is not None and edge_attr.size(0) > 0:
            edge_attr = edge_attr[keep]

    # --- Edge attr noise ---
    if edge_attr is not None and edge_attr_noise_sigma > 0 and edge_attr.numel() > 0:
        edge_attr = edge_attr + torch.randn_like(edge_attr) * edge_attr_noise_sigma

    result = {"x": x, "edge_index": edge_index, "label": data.get("label", "")}
    if edge_attr is not None:
        result["edge_attr"] = edge_attr
    return result


def load_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_path", "cache_path", "taxonomy_v2_class"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def augment_class(
    rows: list[dict],
    target_count: int,
    aug_dir: Path,
    aug_configs: list[dict],
) -> list[dict]:
    """Generate augmented samples for `rows` until reaching `target_count`.

    Returns list of new manifest rows (original rows NOT included).
    """
    new_rows = []
    current = len(rows)
    needed = max(0, target_count - current)
    if needed == 0:
        return new_rows

    aug_dir.mkdir(parents=True, exist_ok=True)
    cfg_cycle = aug_configs * ((needed // len(aug_configs)) + 2)

    for i in range(needed):
        src_row = rows[i % len(rows)]
        src_path = src_row["cache_path"]
        cls = src_row["taxonomy_v2_class"]
        cfg = cfg_cycle[i]

        try:
            orig = torch.load(src_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [WARN] Failed to load {src_path}: {e}")
            continue

        aug = augment_graph(orig, seed=i * 17 + hash(src_path) % 997, **cfg)

        # Unique filename: hash of source + augmentation index
        uid = hashlib.md5(f"{src_path}_{i}".encode()).hexdigest()[:12]
        aug_path = aug_dir / f"{cls}_{uid}.pt"
        torch.save(aug, str(aug_path))

        new_rows.append({
            "file_path": str(aug_path),
            "cache_path": str(aug_path),
            "taxonomy_v2_class": cls,
        })

        if (i + 1) % 20 == 0 or (i + 1) == needed:
            print(f"  {cls}: {i+1}/{needed} augmented", flush=True)

    return new_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment rare graph classes.")
    parser.add_argument("--manifest", default="data/graph_cache/cache_manifest.csv")
    parser.add_argument("--aug-cache-dir", default="data/graph_cache_aug/aug_graphs/")
    parser.add_argument("--output-manifest", default="data/graph_cache_aug/cache_manifest_aug.csv")
    parser.add_argument(
        "--target-samples", type=int, default=80,
        help="Augment all classes below this count up to this number."
    )
    parser.add_argument(
        "--bearing-target", type=int, default=200,
        help="Special higher target for 轴承座 (bearing housing)."
    )
    parser.add_argument(
        "--valve-target", type=int, default=120,
        help="Special higher target for 阀门 (valve)."
    )
    args = parser.parse_args()

    print("Loading manifest...", flush=True)
    orig_rows = load_manifest(args.manifest)
    print(f"  {len(orig_rows)} original samples")

    # Group by class
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in orig_rows:
        by_class[r["taxonomy_v2_class"]].append(r)

    counts = Counter({k: len(v) for k, v in by_class.items()})
    print("\nClass distribution:")
    for cls, cnt in sorted(counts.items(), key=lambda x: x[1]):
        print(f"  {cnt:4d}  {cls}")

    # Augmentation configs (varied parameter combinations for diversity)
    aug_configs = [
        {"edge_dropout_p": 0.10, "feature_noise_sigma": 0.02, "node_mask_p": 0.05, "edge_attr_noise_sigma": 0.02},
        {"edge_dropout_p": 0.15, "feature_noise_sigma": 0.03, "node_mask_p": 0.03, "edge_attr_noise_sigma": 0.03},
        {"edge_dropout_p": 0.08, "feature_noise_sigma": 0.01, "node_mask_p": 0.08, "edge_attr_noise_sigma": 0.01},
        {"edge_dropout_p": 0.12, "feature_noise_sigma": 0.04, "node_mask_p": 0.04, "edge_attr_noise_sigma": 0.04},
        {"edge_dropout_p": 0.05, "feature_noise_sigma": 0.02, "node_mask_p": 0.10, "edge_attr_noise_sigma": 0.02},
    ]

    aug_dir = Path(args.aug_cache_dir)
    all_new_rows: list[dict] = []

    # Determine per-class targets
    class_targets: dict[str, int] = {}
    for cls, rows_for_cls in by_class.items():
        if cls == "轴承座":
            target = args.bearing_target
        elif cls == "阀门":
            target = args.valve_target
        else:
            target = args.target_samples
        class_targets[cls] = target

    # Only augment classes that are below their target
    classes_to_aug = {
        cls: rows for cls, rows in by_class.items()
        if len(rows) < class_targets[cls]
    }

    print(f"\nAugmenting {len(classes_to_aug)} classes...")
    total_new = 0
    for cls, rows_for_cls in sorted(classes_to_aug.items(), key=lambda x: len(x[1])):
        target = class_targets[cls]
        needed = target - len(rows_for_cls)
        print(f"\n{cls}: {len(rows_for_cls)} → {target} (+{needed})")
        new_rows = augment_class(
            rows_for_cls, target,
            aug_dir / cls.replace("/", "_"),
            aug_configs,
        )
        all_new_rows.extend(new_rows)
        total_new += len(new_rows)

    # Combine original + augmented
    combined_rows = orig_rows + all_new_rows

    # Normalize to output manifest format
    out_rows = []
    for r in combined_rows:
        out_rows.append({
            "file_path": r.get("file_path", r.get("cache_path", "")),
            "cache_path": r.get("cache_path", ""),
            "taxonomy_v2_class": r["taxonomy_v2_class"],
        })

    write_manifest(out_rows, args.output_manifest)

    final_counts = Counter(r["taxonomy_v2_class"] for r in out_rows)
    print(f"\nFinal manifest: {len(out_rows)} samples ({total_new} new augmented)")
    print("\nFinal class distribution:")
    for cls, cnt in sorted(final_counts.items(), key=lambda x: x[1]):
        orig_cnt = counts.get(cls, 0)
        print(f"  {cnt:4d}  {cls}  (was {orig_cnt})")
    print(f"\nManifest written to: {args.output_manifest}")


if __name__ == "__main__":
    main()
