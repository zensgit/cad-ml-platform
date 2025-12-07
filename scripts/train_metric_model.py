#!/usr/bin/env python3
"""
Active Learning Training Script (Scaffold).

This script demonstrates how to consume user feedback logs to fine-tune
the Metric Learning model.

Workflow:
1. Load feedback logs from `data/feedback/*.jsonl`.
2. Filter for 'hard negatives' or high-confidence corrections.
3. Construct Triplet/Contrastive dataset.
4. (Mock) Fine-tune the model.
"""

import argparse
import json
import logging
import os
import glob
from typing import List, Dict, Any
import sys

# Add project root to path
sys.path.append(os.getcwd())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_metric_model")

def load_feedback_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load and aggregate feedback logs."""
    files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    dataset = []
    logger.info(f"Found {len(files)} feedback log files in {data_dir}")
    
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        dataset.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {fpath}")
        except Exception as e:
            logger.error(f"Error reading {fpath}: {e}")
            
    return dataset

def prepare_training_triplets(feedback_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert feedback into training triplets (Anchor, Positive, Negative).
    
    For a correction:
    - Anchor: The analyzed document (analysis_id)
    - Positive: A sample from the 'correct_label' class (retrieved from vector store)
    - Negative: A sample from the originally predicted class (if available)
    """
    triplets = []
    logger.info(f"Processing {len(feedback_data)} feedback entries...")
    
    # Connect to VectorStore to fetch features
    try:
        from src.core.similarity import get_vector_store, get_vector_meta
        store = get_vector_store()
        
        # Build a simple index of ID -> Label for positive sampling
        # In a real system, this would be a pre-computed index or DB query
        label_to_ids = {}
        # Scan memory store for simplicity (Redis scan would be needed for prod)
        # Accessing protected member _store for this script context is acceptable
        if hasattr(store, "_store"):
            for vid in store._store.keys():
                meta = get_vector_meta(vid)
                if meta and "type" in meta:
                    lbl = meta["type"]
                    if lbl not in label_to_ids:
                        label_to_ids[lbl] = []
                    label_to_ids[lbl].append(vid)
    except ImportError:
        logger.warning("Could not import VectorStore. Running in offline mode.")
        store = None
        label_to_ids = {}
    
    import random
    
    for entry in feedback_data:
        anchor_id = entry["analysis_id"]
        correct_label = entry["correct_label"]
        
        # 1. Validate Anchor
        if store and not store.exists(anchor_id):
            logger.debug(f"Anchor {anchor_id} not found in store. Skipping.")
            continue
            
        # 2. Find Positive (same label)
        pos_candidates = label_to_ids.get(correct_label, [])
        # Exclude self
        pos_candidates = [pid for pid in pos_candidates if pid != anchor_id]
        
        if not pos_candidates:
            logger.debug(f"No positive candidates for label {correct_label}. Skipping.")
            continue
            
        positive_id = random.choice(pos_candidates)
        
        # 3. Find Negative (different label)
        neg_label = "unknown"
        
        # Hard Negative Mining: If user flagged as hard negative, try to find a sample 
        # from the *originally predicted* class (if we knew it) or just a very similar class.
        # For now, we don't have the original prediction in the feedback struct (unless we add it).
        # But we can use the 'is_hard_negative' flag to perhaps pick a "closer" negative if we had a similarity index.
        # Since we don't have an index here, we'll stick to random negative but log it.
        
        if entry.get("is_hard_negative"):
            # TODO: In a real system, query for the "nearest neighbor that is NOT correct_label"
            pass

        other_labels = [l for l in label_to_ids.keys() if l != correct_label]
        if not other_labels:
            continue
            
        neg_label = random.choice(other_labels)
        neg_candidates = label_to_ids.get(neg_label, [])
        if not neg_candidates:
            continue
            
        negative_id = random.choice(neg_candidates)
            
        triplet = {
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_id": negative_id,
            "anchor_label": correct_label,
            "negative_label": neg_label,
            "source": "user_feedback"
        }
        triplets.append(triplet)
        
    logger.info(f"Generated {len(triplets)} valid training triplets from store.")
    return triplets

def train_model(triplets: List[Dict[str, Any]], output_path: str, dry_run: bool = False):
    """Mock training loop (with PyTorch scaffold)."""
    if not triplets:
        logger.warning("No training data available.")
        return

    logger.info(f"Starting training with {len(triplets)} samples...")
    
    # PyTorch Training Loop Scaffold
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from src.core.similarity import get_vector_store
        from src.ml.metric_learning.model import MetricMLP
        
        store = get_vector_store()
        
        # Initialize model (v7 = 160 dims)
        model = MetricMLP(input_dim=160, embedding_dim=64)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.TripletMarginLoss(margin=1.0)
        
        logger.info("PyTorch detected. Running training steps...")
        model.train()
        
        # Prepare Batch Data
        # In a real script, use DataLoader. Here we load all into memory for simplicity.
        anchors = []
        positives = []
        negatives = []
        
        for t in triplets:
            va = store.get(t["anchor_id"])
            vp = store.get(t["positive_id"])
            vn = store.get(t["negative_id"])
            
            if va and vp and vn:
                # Ensure dimension match (pad if needed, though v7 should be 160)
                # For MVP, assume they match model input_dim
                anchors.append(va)
                positives.append(vp)
                negatives.append(vn)
        
        if not anchors:
            logger.warning("No valid vectors found for triplets.")
            return

        # Convert to Tensor
        t_a = torch.tensor(anchors, dtype=torch.float32)
        t_p = torch.tensor(positives, dtype=torch.float32)
        t_n = torch.tensor(negatives, dtype=torch.float32)
        
        # Training Loop
        epochs = 5
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            emb_a = model(t_a)
            emb_p = model(t_p)
            emb_n = model(t_n)
            
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
            
        if not dry_run:
            # Versioning: Append timestamp
            import time
            timestamp = int(time.time())
            final_path = output_path.replace(".pth", f"_v{timestamp}.pth")
            
            torch.save(model.state_dict(), final_path)
            logger.info(f"Model saved to {final_path}")
            
            # Update 'latest' symlink or config
            # os.symlink(final_path, output_path) # logic omitted for cross-platform safety
            
    except ImportError:
        logger.warning("PyTorch not found. Falling back to simulation.")
        # Simulate training delay
        import time
        time.sleep(1.0)
        
        if not dry_run:
            # Save "model" (timestamp placeholder)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                f.write(f"Model trained on {len(triplets)} samples at {time.ctime()}")
            logger.info(f"Model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Metric Learning Model from Feedback")
    parser.add_argument("--data-dir", default="data/feedback", help="Directory containing feedback logs")
    parser.add_argument("--output-model", default="models/metric_learning/fine_tuned.pth", help="Path to save model")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without saving")
    parser.add_argument("--mock-data", action="store_true", help="Populate vector store with mock data for testing")
    
    args = parser.parse_args()
    
    if args.mock_data:
        logger.info("Populating vector store with mock data...")
        from src.core.similarity import _VECTOR_STORE, _VECTOR_META
        import random
        
        # Mock data matching test_feedback.jsonl
        # Anchors
        _VECTOR_STORE["test_anchor_1"] = [random.random() for _ in range(160)]
        _VECTOR_META["test_anchor_1"] = {"type": "bolt"}
        
        _VECTOR_STORE["test_anchor_2"] = [random.random() for _ in range(160)]
        _VECTOR_META["test_anchor_2"] = {"type": "nut"}
        
        # Positives/Negatives candidates
        for i in range(5):
            _VECTOR_STORE[f"pos_bolt_{i}"] = [random.random() for _ in range(160)]
            _VECTOR_META[f"pos_bolt_{i}"] = {"type": "bolt"}
            
            _VECTOR_STORE[f"pos_nut_{i}"] = [random.random() for _ in range(160)]
            _VECTOR_META[f"pos_nut_{i}"] = {"type": "nut"}
            
            _VECTOR_STORE[f"neg_washer_{i}"] = [random.random() for _ in range(160)]
            _VECTOR_META[f"neg_washer_{i}"] = {"type": "washer"}
            
    if not os.path.exists(args.data_dir):
        logger.warning(f"Data directory {args.data_dir} does not exist. Creating it for demo.")
        os.makedirs(args.data_dir, exist_ok=True)
        
    data = load_feedback_data(args.data_dir)
    triplets = prepare_training_triplets(data)
    train_model(triplets, args.output_model, args.dry_run)

if __name__ == "__main__":
    main()
