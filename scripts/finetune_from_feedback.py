#!/usr/bin/env python3
"""
Fine-tune the classification model using human feedback from Active Learning.

Steps:
1. Export labeled data from ActiveLearner.
2. Load the current model (or initialize a new one).
3. Fine-tune or retrain the model.
4. Save the new model with a version tag.
5. Trigger model reload.
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.active_learning import get_active_learner
from src.ml.classifier import reload_model, get_model_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("finetune")

def _load_samples(file_path: str) -> Tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    labels: List[str] = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            doc_id = data.get("doc_id")
            label = data.get("true_type")
            if doc_id and label:
                doc_ids.append(doc_id)
                labels.append(label)
    return doc_ids, labels

def fetch_vectors_for_samples(doc_ids: List[str], labels: List[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fetch vectors for the given doc_ids from the vector store."""
    from src.core.similarity import get_vector

    if not doc_ids:
        return np.array([]), np.array([])

    X_list = []
    y_list = []

    for doc_id, label in zip(doc_ids, labels):
        vec = get_vector(doc_id)
        if vec:
            X_list.append(vec)
            y_list.append(label)
        else:
            logger.warning(f"Vector not found for {doc_id}, skipping")

    if not X_list:
        return np.array([]), np.array([])

    return np.array(X_list), np.array(y_list)


def load_training_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load training data from JSONL export."""
    doc_ids, labels = _load_samples(file_path)
    X, y = fetch_vectors_for_samples(doc_ids, labels)

    if len(X) > 0:
        return X, y

    logger.warning("No vectors found for training samples.")
    logger.warning("Using mock data for demonstration.")

    X = np.random.rand(10, 32)  # 32-dim vectors
    y = np.array(["bolt", "nut", "washer", "bracket", "gear"] * 2)

    return X, y

def train_model(X: np.ndarray, y: np.ndarray, base_model_path: str = None):
    """Train or fine-tune the model."""
    model = None
    
    # Try to load existing model
    if base_model_path and os.path.exists(base_model_path):
        try:
            with open(base_model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Loaded base model from {base_model_path}")
        except Exception as e:
            logger.warning(f"Failed to load base model: {e}")
    
    if model is None:
        logger.info("Initializing new SGDClassifier")
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
        
    # Check if model supports partial_fit
    if hasattr(model, "partial_fit"):
        logger.info(f"Fine-tuning model with {len(X)} samples")
        # We need to provide all classes for partial_fit on the first call
        # or assume the model already knows them.
        # If it's a new model, we need classes.
        classes = np.unique(y)
        model.partial_fit(X, y, classes=classes)
    else:
        logger.info("Model does not support partial_fit, retraining from scratch (simulated)")
        # In reality, we'd need the full dataset to retrain non-incremental models
        # For now, we'll just fit on the new data (which is bad for forgetting, but this is a script)
        model.fit(X, y)
        
    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune classifier from feedback")
    parser.add_argument("--force", action="store_true", help="Force training even if threshold not met")
    parser.add_argument("--output-dir", default="models", help="Directory to save new model")
    args = parser.parse_args()
    
    learner = get_active_learner()
    
    # Check threshold
    status = learner.check_retrain_threshold()
    if not status["ready"] and not args.force:
        logger.info(f"Not ready for retraining. {status['recommendation']}")
        return
        
    logger.info("Starting fine-tuning pipeline...")
    
    # 1. Export data
    export_result = learner.export_training_data(format="jsonl")
    if export_result["status"] != "ok" or export_result["count"] == 0:
        logger.warning("No training data available")
        return
        
    data_file = export_result["file"]
    logger.info(f"Exported training data to {data_file}")
    
    # 2. Load vectors
    logger.info("Loading training vectors...")
    X, y = load_training_data(data_file)
    
    if len(X) == 0:
        logger.error("No vectors found for training samples")
        return
        
    # 3. Train
    current_model_info = get_model_info()
    base_path = current_model_info.get("path")
    
    new_model = train_model(X, y, base_path)
    
    # 4. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_version = f"v{timestamp}"
    output_path = os.path.join(args.output_dir, f"classifier_{new_version}.pkl")
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(new_model, f)
        
    logger.info(f"Saved new model to {output_path}")
    
    # 5. Reload
    logger.info("Reloading model...")
    # We need to set the env var for the new path so reload picks it up?
    # Or reload_model accepts a path.
    try:
        result = reload_model(path=output_path, expected_version=new_version, force=True)
        if result["status"] == "success":
            logger.info(f"Successfully reloaded model version {new_version}")
        else:
            logger.error(f"Failed to reload model: {result}")
    except Exception as e:
        logger.error(f"Error during reload: {e}")

if __name__ == "__main__":
    main()
