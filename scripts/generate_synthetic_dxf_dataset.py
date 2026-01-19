"""
Synthetic DXF Dataset Generator.

Generates random mechanical 2D drawings with ground truth labels.
Used to train Feature Recognition models when real labeled data is scarce.

Features:
- Random Plates (Rectangular/Circular)
- Features: Holes (Circle), Slots (Obround), Cutouts (Rect)
- Semantic Layers: 'VISIBLE', 'CENTER', 'DIMENSION'
- Annotations: Generates a label file (JSON) describing the features.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple

import ezdxf
from ezdxf.math import Vec2

logger = logging.getLogger(__name__)

OUTPUT_DIR = "data/synthetic_dxf"
LABEL_FILE = os.path.join(OUTPUT_DIR, "labels.json")


class DxfGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.labels = []

    def generate_batch(self, count: int = 10):
        """Generate N synthetic drawings."""
        print(f"Generating {count} synthetic DXF files...")
        for i in range(count):
            file_name = f"synth_{i:04d}.dxf"
            path = os.path.join(self.output_dir, file_name)
            
            # 1. Create Layout
            doc = ezdxf.new("R2010")
            msp = doc.modelspace()
            self._setup_layers(doc)

            # 2. Generate Geometry & Features
            features_meta = self._draw_random_part(msp)

            # 3. Save
            doc.saveas(path)
            
            # 4. Record Labels
            self.labels.append({
                "file": file_name,
                "features": features_meta
            })
            
        # Save Label Manifest
        with open(LABEL_FILE, "w") as f:
            json.dump(self.labels, f, indent=2)
        print(f"✅ Generated {count} files. Labels saved to {LABEL_FILE}")

    def _setup_layers(self, doc):
        """Standard mechanical layers."""
        doc.layers.add("VISIBLE", color=7)  # White/Black (Outline)
        doc.layers.add("CENTER", color=1, linetype="CENTER")  # Red
        doc.layers.add("DIMENSION", color=3)  # Green
        doc.layers.add("HIDDEN", color=4, linetype="HIDDEN")  # Cyan

    def _draw_random_part(self, msp) -> List[Dict[str, Any]]:
        """Draws a base shape and adds random features."""
        features = []
        
        # Base Plate
        width = random.randint(50, 200)
        height = random.randint(50, 200)
        
        # Draw Outline (Rectangle)
        points = [
            (0, 0), (width, 0), (width, height), (0, height), (0, 0)
        ]
        msp.add_lwpolyline(points, dxfattribs={"layer": "VISIBLE"})
        
        features.append({
            "type": "plate",
            "bbox": [0, 0, width, height],
            "area": width * height
        })

        # Add Holes (Random Count 1-5)
        num_holes = random.randint(1, 5)
        for _ in range(num_holes):
            # Ensure hole is inside
            r = random.randint(3, 10)
            cx = random.randint(r + 5, width - r - 5)
            cy = random.randint(r + 5, height - r - 5)
            
            msp.add_circle((cx, cy), radius=r, dxfattribs={"layer": "VISIBLE"})
            
            # Center Mark
            msp.add_line((cx - r - 2, cy), (cx + r + 2, cy), dxfattribs={"layer": "CENTER"})
            msp.add_line((cx, cy - r - 2), (cx, cy + r + 2), dxfattribs={"layer": "CENTER"})

            features.append({
                "type": "hole",
                "center": [cx, cy],
                "radius": r,
                "diameter": r * 2
            })
            
        # Add Slot (Randomly 30% chance)
        if random.random() < 0.3:
            w = random.randint(20, 40)
            h = random.randint(5, 10)
            x = random.randint(10, width - w - 10)
            y = random.randint(10, height - h - 10)
            
            # Simple Rect Slot
            slot_pts = [(x, y), (x+w, y), (x+w, y+h), (x, y+h), (x, y)]
            msp.add_lwpolyline(slot_pts, dxfattribs={"layer": "VISIBLE"})
            
            features.append({
                "type": "slot",
                "bbox": [x, y, w, h]
            })

        return features

if __name__ == "__main__":
    gen = DxfGenerator(OUTPUT_DIR)
    gen.generate_batch(20)
