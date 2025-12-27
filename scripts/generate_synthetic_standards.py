"""
Synthetic Standard Parts Generator.

Uses PythonOCC to procedurally generate standard parts (Bolts, Washers, Pins)
and save them as STEP files. Useful for populating a demo Public Library.
"""

import os
import math
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StdGen")

try:
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism
    from OCC.Core.gp import gp_Pnt, gp_Ax2, gp_Dir, gp_Vec, gp_Circ, gp_Ax1
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
    HAS_OCC = True
except ImportError:
    HAS_OCC = False
    logger.warning("PythonOCC not found. Cannot generate geometry.")

OUTPUT_DIR = "data/standards_raw"

def save_step(shape, filename):
    if not HAS_OCC: return
    
    path = os.path.join(OUTPUT_DIR, filename)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(path)
    if status == 1:
        logger.info(f"Generated: {path}")
    else:
        logger.error(f"Failed to write: {path}")

def make_hex_head(width_across_flats, height):
    # Create a hexagonal prism
    # Radius of circumscribed circle = width / sqrt(3)
    # Actually, width_across_flats = 2 * r * sin(60) = r * sqrt(3)
    # so r = width / sqrt(3)
    r = width_across_flats / math.sqrt(3)
    
    # Create 6 points
    points = []
    for i in range(6):
        angle = math.radians(60 * i)
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        points.append(gp_Pnt(x, y, 0))
    points.append(points[0]) # Close loop
    
    # Build wire -> Face -> Prism
    edges = []
    for i in range(6):
        edges.append(BRepBuilderAPI_MakeEdge(points[i], points[i+1]).Edge())
    
    wire = BRepBuilderAPI_MakeWire()
    for e in edges:
        wire.Add(e)
        
    face = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
    prism = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, height)).Shape()
    return prism

def gen_bolt(size_m, length):
    """Generate a simplified Hex Bolt (M{size} x {length})."""
    if not HAS_OCC: return

    # Head dimensions (Approx ISO 4014)
    head_height = size_m * 0.7
    width_flats = size_m * 1.6
    
    # 1. Head
    head = make_hex_head(width_flats, head_height)
    
    # 2. Shank (Cylinder)
    # Shank starts at z=0, goes to z=-length
    # Head starts at z=0, goes to z=head_height
    # Actually standard usually has head at top. Let's put head at Z+
    
    shank = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0,0,0), gp_Dir(0,0,-1)), size_m/2.0, length).Shape()
    
    # Fuse
    bolt = BRepAlgoAPI_Fuse(head, shank).Shape()
    save_step(bolt, f"Bolt_Hex_M{size_m}x{length}.step")

def gen_washer(size_m):
    """Generate a Flat Washer (M{size})."""
    if not HAS_OCC: return
    
    # ISO 7089 approx
    id = size_m + 0.5 # Clearance
    od = size_m * 2.0
    thickness = size_m * 0.15
    if thickness < 1.0: thickness = 1.0
    
    # Make outer cylinder
    outer = BRepPrimAPI_MakeCylinder(od/2.0, thickness).Shape()
    
    # Make inner cylinder (tool)
    inner = BRepPrimAPI_MakeCylinder(id/2.0, thickness * 1.1).Shape() # Slightly longer
    
    # Cut
    washer = BRepAlgoAPI_Cut(outer, inner).Shape()
    save_step(washer, f"Washer_Flat_M{size_m}.step")

def gen_pin(diameter, length):
    """Generate a Cylindrical Pin."""
    if not HAS_OCC: return
    
    pin = BRepPrimAPI_MakeCylinder(diameter/2.0, length).Shape()
    save_step(pin, f"Pin_Cyl_D{diameter}x{length}.step")

def generate_all():
    if not HAS_OCC:
        logger.error("Cannot generate standards: pythonocc-core not installed.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Bolts
    sizes = [6, 8, 10, 12, 16]
    lengths = [20, 30, 40, 50, 60, 80, 100]
    for s in sizes:
        for l in lengths:
            gen_bolt(s, l)
            
    # 2. Washers
    for s in sizes:
        gen_washer(s)
        
    # 3. Pins
    pins = [(5, 20), (5, 40), (10, 40), (10, 60), (20, 100)]
    for d, l in pins:
        gen_pin(d, l)
        
    logger.info("Generation complete.")

if __name__ == "__main__":
    generate_all()
