#!/usr/bin/env python3
"""DXF Data Augmentation Tool.

Applies geometric augmentations to DXF files for training data expansion:
- Rotation (0, 90, 180, 270 degrees)
- Scale (0.8x - 1.2x)
- Mirror (horizontal / vertical)
- Entity dropout (5-10% random removal)

Usage:
    python scripts/augment_dxf_data.py --input-dir data/training/ --dry-run
    python scripts/augment_dxf_data.py --input-dir data/training/ --copies 5
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

# Augmentation parameters
ROTATION_ANGLES = [0, 90, 180, 270]
SCALE_RANGE = (0.8, 1.2)
DROPOUT_RANGE = (0.05, 0.10)


def iter_dxf_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    """Find all DXF files."""
    patterns = ["*.dxf", "*.DXF"]
    paths: List[Path] = []
    for pattern in patterns:
        if recursive:
            paths.extend(sorted(input_dir.rglob(pattern)))
        else:
            paths.extend(sorted(input_dir.glob(pattern)))
    seen: set = set()
    unique: List[Path] = []
    for p in paths:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def generate_augmentation_params(
    copy_idx: int,
    seed: Optional[int] = None,
) -> dict:
    """Generate a deterministic set of augmentation parameters for one copy."""
    rng = random.Random(seed if seed is not None else copy_idx)
    angle = rng.choice(ROTATION_ANGLES)
    scale = round(rng.uniform(*SCALE_RANGE), 3)
    mirror_x = rng.choice([True, False])
    mirror_y = rng.choice([True, False])
    dropout_rate = round(rng.uniform(*DROPOUT_RANGE), 4)
    return {
        "angle": angle,
        "scale": scale,
        "mirror_x": mirror_x,
        "mirror_y": mirror_y,
        "dropout_rate": dropout_rate,
    }


def apply_augmentation(input_path: Path, output_path: Path, params: dict) -> None:
    """Apply geometric augmentation to a DXF file using ezdxf.

    ezdxf is imported here (deferred) so --dry-run works without it installed.
    """
    import ezdxf  # type: ignore[import-untyped]

    doc = ezdxf.readfile(str(input_path))
    msp = doc.modelspace()

    entities = list(msp)
    angle = params["angle"]
    scale = params["scale"]
    mirror_x = params["mirror_x"]
    mirror_y = params["mirror_y"]
    dropout_rate = params["dropout_rate"]

    # Entity dropout: randomly remove a fraction of entities
    if dropout_rate > 0 and entities:
        n_drop = max(1, int(len(entities) * dropout_rate))
        rng = random.Random(hash(str(output_path)))
        drop_indices = set(rng.sample(range(len(entities)), min(n_drop, len(entities))))
        for idx in sorted(drop_indices, reverse=True):
            msp.delete_entity(entities[idx])
        # Refresh entity list after deletion
        entities = list(msp)

    # Build transformation matrix
    # We apply: mirror -> scale -> rotate
    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))

    sx = -scale if mirror_x else scale
    sy = -scale if mirror_y else scale

    # Combined 2D affine: [sx*cos, -sy*sin, sx*sin, sy*cos]
    m00 = sx * cos_a
    m01 = -sy * sin_a
    m10 = sx * sin_a
    m11 = sy * cos_a

    # Apply transformation to supported entity types
    for entity in entities:
        dxftype = entity.dxftype()
        try:
            if hasattr(entity, "transform"):
                # ezdxf >= 0.15 supports generic transform via Matrix44
                from ezdxf.math import Matrix44  # type: ignore[import-untyped]
                matrix = Matrix44([
                    m00, m01, 0, 0,
                    m10, m11, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1,
                ])
                entity.transform(matrix)
        except Exception:
            # Skip entities that cannot be transformed
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(output_path))


def run_augmentation(
    input_dir: Path,
    output_dir: Path,
    copies: int = 3,
    seed: int = 42,
    dry_run: bool = False,
    recursive: bool = True,
) -> None:
    """Main augmentation loop."""
    dxf_files = iter_dxf_files(input_dir, recursive=recursive)
    print(f"Found {len(dxf_files)} DXF files in {input_dir}")
    print(f"Generating {copies} augmented copies per file")
    print(f"Output directory: {output_dir}")
    if dry_run:
        print("[DRY RUN] No files will be written.\n")

    total_generated = 0
    total_errors = 0

    for dxf_path in dxf_files:
        rel = dxf_path.relative_to(input_dir) if input_dir in dxf_path.parents else Path(dxf_path.name)
        stem = rel.stem
        suffix = rel.suffix
        parent = rel.parent

        for i in range(copies):
            copy_seed = seed + hash(str(dxf_path)) + i
            params = generate_augmentation_params(i, seed=copy_seed)
            out_name = f"{stem}_aug{i:02d}_r{params['angle']}_s{params['scale']}{suffix}"
            out_path = output_dir / parent / out_name

            if dry_run:
                print(f"  {dxf_path.name} -> {out_name}  {params}")
                total_generated += 1
                continue

            try:
                apply_augmentation(dxf_path, out_path, params)
                total_generated += 1
            except Exception as e:
                print(f"  ERROR augmenting {dxf_path.name} copy {i}: {e}", file=sys.stderr)
                total_errors += 1

    print(f"\nDone. generated={total_generated} errors={total_errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment DXF data with geometric transformations")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory of DXF files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "augmented", help="Output directory")
    parser.add_argument("--copies", type=int, default=3, help="Number of augmented copies per file (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"ERROR: input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    run_augmentation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copies=args.copies,
        seed=args.seed,
        dry_run=args.dry_run,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
