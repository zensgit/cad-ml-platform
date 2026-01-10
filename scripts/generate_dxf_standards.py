"""
DXF Standard Parts Generator.

Uses ezdxf to procedurally generate valid 2D CAD drawings of standard parts.
"""

import json
import os
import math
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import ezdxf
from ezdxf.enums import TextEntityAlignment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DxfGen")

OUTPUT_DIR = "data/standards_dxf"

def create_dxf(filename, setup_func):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    
    setup_func(msp)
    
    path = os.path.join(OUTPUT_DIR, filename)
    doc.saveas(path)
    logger.info(f"Generated: {path}")
    return path

def write_manifest(files: List[str], config: Dict[str, Any]) -> None:
    manifest = {
        "version": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/generate_dxf_standards.py",
        "output_dir": OUTPUT_DIR,
        "config": config,
        "file_count": len(files),
        "files": sorted(files),
    }
    path = os.path.join(OUTPUT_DIR, "MANIFEST.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)
    logger.info(f"Wrote manifest: {path}")

def draw_hex_bolt(msp, size_m, length):
    """Draws a Hex Bolt (Side View & Top View)."""
    # Dimensions (Approx)
    head_height = size_m * 0.7
    width_flats = size_m * 1.6
    width_corners = width_flats / math.cos(math.radians(30))
    
    # 1. Side View (Rectangle shank + Head)
    # Head (simplified as rectangle for side view)
    msp.add_lwpolyline([(0, 0), (width_corners, 0), (width_corners, head_height), (0, head_height), (0, 0)], close=True)
    
    # Shank
    shank_start_x = (width_corners - size_m) / 2
    msp.add_lwpolyline([
        (shank_start_x, head_height), 
        (shank_start_x + size_m, head_height), 
        (shank_start_x + size_m, head_height + length), 
        (shank_start_x, head_height + length),
        (shank_start_x, head_height)
    ], close=True)
    
    # Thread lines (schematic)
    msp.add_line((shank_start_x, head_height + length * 0.2), (shank_start_x + size_m, head_height + length * 0.2))
    
    # 2. Top View (Hexagon) - Offset by 2x Length
    center = (0, -length - size_m * 2)
    # Hexagon points
    pts = []
    r = width_corners / 2
    for i in range(6):
        angle = math.radians(30 + 60 * i)
        pts.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
    msp.add_lwpolyline(pts, close=True)
    
    # Inner circle (shank dia)
    msp.add_circle(center, size_m / 2)
    
    # Text Label
    msp.add_text(f"ISO 4014 M{size_m}x{length}", height=size_m/2).set_placement((0, -size_m), align=TextEntityAlignment.CENTER)

def draw_washer(msp, size_m):
    """Draws a Flat Washer (Top View)."""
    id = size_m + 0.5
    od = size_m * 2.0
    
    msp.add_circle((0, 0), id / 2)
    msp.add_circle((0, 0), od / 2)
    
    msp.add_text(f"ISO 7089 M{size_m}", height=size_m/3).set_placement((0, -od/1.5), align=TextEntityAlignment.CENTER)

def draw_flange(msp, size_dn):
    """Draws a Flange with bolt holes."""
    # Approx dims
    od = size_dn * 3.0
    id = size_dn * 1.1
    pcd = size_dn * 2.2 # Pitch Circle Diameter
    hole_dia = size_dn * 0.2
    num_holes = 4 if size_dn < 50 else 8
    
    msp.add_circle((0, 0), od / 2)
    msp.add_circle((0, 0), id / 2)
    
    # Bolt holes
    for i in range(num_holes):
        angle = math.radians(360 / num_holes * i)
        x = (pcd / 2) * math.cos(angle)
        y = (pcd / 2) * math.sin(angle)
        msp.add_circle((x, y), hole_dia / 2)
        
    msp.add_text(f"Flange DN{size_dn}", height=size_dn/4).set_placement((0, -od/1.8), align=TextEntityAlignment.CENTER)

def generate_all():
    bolt_sizes = [6, 8, 10, 12, 16]
    bolt_lengths = [20, 40, 60]
    washer_sizes = [6, 8, 10, 12, 20]
    flange_sizes = [25, 40, 50, 80, 100]

    config = {
        "bolts": {"sizes_m": bolt_sizes, "lengths": bolt_lengths},
        "washers": {"sizes_m": washer_sizes},
        "flanges": {"sizes_dn": flange_sizes},
    }

    files = []

    # Bolts
    for m in bolt_sizes:
        for l in bolt_lengths:
            path = create_dxf(
                f"Bolt_M{m}x{l}.dxf",
                lambda msp, m=m, l=l: draw_hex_bolt(msp, m, l),
            )
            files.append(path)

    # Washers
    for m in washer_sizes:
        path = create_dxf(f"Washer_M{m}.dxf", lambda msp, m=m: draw_washer(msp, m))
        files.append(path)

    # Flanges
    for dn in flange_sizes:
        path = create_dxf(
            f"Flange_DN{dn}.dxf",
            lambda msp, dn=dn: draw_flange(msp, dn),
        )
        files.append(path)

    write_manifest(files, config)

if __name__ == "__main__":
    generate_all()
