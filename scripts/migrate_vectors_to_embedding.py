#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.similarity import _VECTOR_STORE, _VECTOR_META, _VECTOR_LOCK
from src.ml.metric_embedder import MetricEmbedder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_vectors(model_path: str, dry_run: bool = True, target_version: str = "ml_embed_v1"):
    """
    Migrate vectors in the in-memory store to metric embeddings.
    Note: This script operates on the in-memory store of the RUNNING process if run via API,
    but here as a standalone script it only operates on a loaded snapshot (if persistence existed).
    
    Since the current system uses in-memory storage without persistence (based on code analysis),
    this script is primarily a template for how to migrate if persistence is added, 
    OR it can be used to verify the conversion process on a sample set.
    
    If the system has a persistence layer (e.g. loading from JSON on startup), 
    this script should load that, convert, and save back.
    """
    
    logger.info(f"Initializing MetricEmbedder from {model_path}...")
    embedder = MetricEmbedder(model_path)
    
    if embedder.model is None:
        logger.error("Failed to load model. Aborting.")
        return

    logger.info(f"Starting migration (Dry Run: {dry_run})...")
    
    # In a real persistent system, we would load the store here.
    # For this demo/template, we'll assume _VECTOR_STORE is populated or we populate it with dummy data
    if not _VECTOR_STORE:
        logger.warning("_VECTOR_STORE is empty. Populating with dummy v5 data for demonstration.")
        for i in range(10):
            _VECTOR_STORE[f"demo_{i}"] = [0.1] * 24 # v5 dim
            _VECTOR_META[f"demo_{i}"] = {"feature_version": "v5"}

    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    keys = list(_VECTOR_STORE.keys())
    total = len(keys)
    
    for i, vid in enumerate(keys):
        try:
            vec = _VECTOR_STORE[vid]
            meta = _VECTOR_META.get(vid, {})
            current_ver = meta.get("feature_version", "unknown")
            current_embed_ver = meta.get("embedding_version", "none")
            
            # Skip if already migrated
            if current_embed_ver == target_version:
                skipped_count += 1
                continue
                
            # Skip if not v5/v6 (unless we want to support others)
            if current_ver not in ["v5", "v6"]:
                skipped_count += 1
                continue
                
            # Convert
            new_vec = embedder.embed(vec)
            
            # Check dimension change
            if len(new_vec) == len(vec):
                # Embedding failed or model not loaded correctly (fallback behavior)
                logger.warning(f"Vector {vid} embedding returned same dimension {len(vec)}. Skipping.")
                error_count += 1
                continue
                
            if not dry_run:
                with _VECTOR_LOCK:
                    _VECTOR_STORE[vid] = new_vec
                    if vid not in _VECTOR_META:
                        _VECTOR_META[vid] = {}
                    _VECTOR_META[vid]["embedding_version"] = target_version
                    _VECTOR_META[vid]["original_dimension"] = len(vec)
            
            converted_count += 1
            
            if i % 100 == 0:
                logger.info(f"Processed {i}/{total}...")
                
        except Exception as e:
            logger.error(f"Error processing {vid}: {e}")
            error_count += 1

    logger.info("-" * 30)
    logger.info(f"Migration Complete")
    logger.info(f"Total: {total}")
    logger.info(f"Converted: {converted_count}")
    logger.info(f"Skipped: {skipped_count}")
    logger.info(f"Errors: {error_count}")
    
    if dry_run:
        logger.info("This was a DRY RUN. No changes were applied.")

def main():
    parser = argparse.ArgumentParser(description="Migrate vectors to metric embeddings")
    parser.add_argument("--model", type=str, default="models/metric_learning/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--apply", action="store_true", help="Apply changes (disable dry run)")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        logger.error(f"Model file {args.model} not found.")
        sys.exit(1)
        
    migrate_vectors(args.model, dry_run=not args.apply)

if __name__ == "__main__":
    main()
