#!/usr/bin/env python3
import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META
except ImportError:
    print("Warning: Could not import _VECTOR_STORE from src.core.similarity. Using empty store.")
    _VECTOR_STORE = {}
    _VECTOR_META = {}

def generate_synthetic_data(count: int = 100, dim: int = 26):
    """Generate synthetic data if store is empty."""
    print(f"Generating {count} synthetic vectors with dimension {dim}...")
    store = {}
    meta = {}
    families = ["bolt", "nut", "washer", "bracket", "plate"]
    materials = ["steel", "aluminum", "plastic"]
    
    for i in range(count):
        vid = f"syn_{i:04d}"
        # Create clusters by family
        family_idx = i % len(families)
        base_vec = np.random.rand(dim) * 0.1 + (family_idx * 0.2) # Separate clusters
        noise = np.random.rand(dim) * 0.05
        vec = base_vec + noise
        # Normalize
        vec = vec / np.linalg.norm(vec)
        
        store[vid] = vec.tolist()
        meta[vid] = {
            "family_id": families[family_idx],
            "material": random.choice(materials),
            "feature_version": "v5"
        }
    return store, meta

def create_triplets(store: Dict[str, List[float]], meta: Dict[str, Dict[str, Any]], num_triplets: int = 1000):
    """
    Create triplets (anchor, positive, negative).
    Positive: Same family_id
    Negative: Different family_id
    """
    triplets = []
    ids = list(store.keys())
    
    if not ids:
        return pd.DataFrame()

    print(f"Generating {num_triplets} triplets from {len(ids)} vectors...")
    
    # Group by family
    by_family = {}
    for vid in ids:
        fam = meta.get(vid, {}).get("family_id", "unknown")
        if fam not in by_family:
            by_family[fam] = []
        by_family[fam].append(vid)
        
    families = list(by_family.keys())
    
    for _ in range(num_triplets):
        # Select anchor
        fam = random.choice(families)
        if len(by_family[fam]) < 2:
            continue
            
        anchor_id = random.choice(by_family[fam])
        
        # Select positive (same family, different id)
        pos_id = random.choice(by_family[fam])
        while pos_id == anchor_id:
            pos_id = random.choice(by_family[fam])
            
        # Select negative (different family)
        neg_fam = random.choice(families)
        while neg_fam == fam:
            neg_fam = random.choice(families)
            
        if not by_family[neg_fam]:
            continue
            
        neg_id = random.choice(by_family[neg_fam])
        
        triplets.append({
            "anchor_id": anchor_id,
            "pos_id": pos_id,
            "neg_id": neg_id,
            "anchor_features": store[anchor_id],
            "pos_features": store[pos_id],
            "neg_features": store[neg_id],
            "family_id": fam
        })
        
    return pd.DataFrame(triplets)

def main():
    parser = argparse.ArgumentParser(description="Export metric learning training data")
    parser.add_argument("--output", type=str, default="data/metric_learning/triplets.parquet", help="Output path")
    parser.add_argument("--num-triplets", type=int, default=1000, help="Number of triplets to generate")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data generation")
    parser.add_argument("--dim", type=int, default=26, help="Dimension for synthetic data")
    args = parser.parse_args()
    
    store = _VECTOR_STORE
    metadata = _VECTOR_META
    
    if not store or args.synthetic:
        store, metadata = generate_synthetic_data(count=200, dim=args.dim)
        
    df = create_triplets(store, metadata, num_triplets=args.num_triplets)
    
    if df.empty:
        print("No triplets generated. Check data source.")
        return
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save as parquet
    try:
        df.to_parquet(args.output, index=False)
        print(f"Saved {len(df)} triplets to {args.output}")
    except ImportError:
        # Fallback to csv if parquet not available (though we installed pandas)
        csv_path = args.output.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} triplets to {csv_path} (parquet support missing)")

if __name__ == "__main__":
    main()
