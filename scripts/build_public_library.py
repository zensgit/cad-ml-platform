"""
Public Library Builder Script.

Ingests a directory of standard part STEP files and builds a
read-only Faiss index for the 'Public' partition.
"""

import os
import argparse
import logging
import time
from src.core.geometry.engine import get_geometry_engine
from src.ml.vision_3d import get_3d_encoder
from src.core.vectors.stores.faiss_store import FaissStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PublicLibBuilder")

def build_library(source_dir: str, output_path: str):
    if not os.path.exists(source_dir):
        logger.error(f"Source {source_dir} not found.")
        return

    # Init Components
    geo = get_geometry_engine()
    encoder = get_3d_encoder()
    
    # Init Index
    # Assuming 128 dim from UVNetEncoder mock/real
    store = FaissStore(dimension=128) 
    
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.step', '.stp'))]
    logger.info(f"Found {len(files)} files to ingest.")
    
    count = 0
    for f in files:
        path = os.path.join(source_dir, f)
        try:
            with open(path, 'rb') as file_obj:
                content = file_obj.read()
                
            shape = geo.load_step(content, f)
            if shape:
                feats = geo.extract_brep_features(shape)
                # DFM features optional for index, but good for meta
                dfm = geo.extract_dfm_features(shape)
                
                vec = encoder.encode(feats)
                
                # Metadata
                meta = {
                    "filename": f,
                    "type": "standard_part",
                    "volume": feats.get("volume"),
                    "dfm": dfm
                }
                
                store.add(f, vec, meta)
                count += 1
                if count % 10 == 0:
                    logger.info(f"Processed {count}/{len(files)}...")
                    
        except Exception as e:
            logger.error(f"Failed to process {f}: {e}")
            
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    store.save(output_path)
    logger.info(f"Successfully built public index with {count} items at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Directory with standard STEP files")
    parser.add_argument("--output", default="data/public_index", help="Output path prefix (no extension)")
    args = parser.parse_args()
    
    build_library(args.source, args.output)
