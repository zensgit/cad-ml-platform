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
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Add project root to path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.active_learning import get_active_learner  # noqa: E402
from src.core.classification.coarse_labels import normalize_coarse_label  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("finetune")


def _resolve_feedback_label(
    data: Dict[str, Any],
    label_field: str,
) -> Optional[str]:
    alias_map = {
        "true_fine_type": ["correct_label", "correct_fine_label"],
        "true_coarse_type": ["correct_coarse_label"],
        "true_type": ["correct_label"],
    }
    candidates = [label_field]
    candidates.extend(alias_map.get(label_field, []))
    candidates.extend(["true_fine_type", "true_type", "true_coarse_type"])
    for field_name in candidates:
        value = data.get(field_name)
        text = str(value or "").strip()
        if text:
            return text
    return None


def _load_samples(
    file_path: str,
    label_field: str = "true_fine_type",
) -> Tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    labels: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            doc_id = data.get("doc_id") or data.get("analysis_id")
            label = _resolve_feedback_label(data, label_field)
            if doc_id and label:
                doc_ids.append(doc_id)
                labels.append(label)
    return doc_ids, labels


def _build_training_summary(
    labels: List[str],
    label_field: str,
    vector_count: Optional[int] = None,
) -> Dict[str, Any]:
    label_distribution = dict(sorted(Counter(labels).items()))
    coarse_labels = [
        normalize_coarse_label(label) or str(label).strip()
        for label in labels
        if str(label).strip()
    ]
    coarse_distribution = dict(sorted(Counter(coarse_labels).items()))
    return {
        "label_field": label_field,
        "sample_count": len(labels),
        "vector_count": vector_count if vector_count is not None else len(labels),
        "unique_label_count": len(label_distribution),
        "unique_coarse_label_count": len(coarse_distribution),
        "label_distribution": label_distribution,
        "coarse_label_distribution": coarse_distribution,
    }


def _write_training_summary(path: str, summary: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

def fetch_vectors_for_samples(
    doc_ids: List[str],
    labels: List[str],
) -> tuple[np.ndarray, np.ndarray]:
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


def load_training_data(
    file_path: str,
    label_field: str = "true_fine_type",
    allow_mock: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load training data from JSONL export.

    Returns real vectors when available.  If no vectors are found and
    *allow_mock* is False (the production default), logs an ERROR and raises
    SystemExit(1).  Pass allow_mock=True only in dev/test environments.
    """
    doc_ids, labels = _load_samples(file_path, label_field=label_field)
    X, y = fetch_vectors_for_samples(doc_ids, labels)

    if len(X) > 0:
        return X, y

    missing = len(doc_ids)
    if not allow_mock:
        logger.error(
            "No vectors found for %d training sample(s). "
            "Ensure the vector store is populated before retraining. "
            "Pass --allow-mock to enable synthetic fallback (dev/test only).",
            missing,
        )
        raise SystemExit(1)

    logger.warning(
        "No vectors found for %d training sample(s). "
        "ALLOW-MOCK is enabled — using synthetic mock data (dev/test only).",
        missing,
    )
    X = np.random.rand(10, 32)  # 32-dim vectors
    y = np.array(["bolt", "nut", "washer", "bracket", "gear"] * 2)
    return X, y

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    base_model_path: Optional[str] = None,
):
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
        logger.info(
            "Model does not support partial_fit, retraining from scratch (simulated)"
        )
        # In reality, we'd need the full dataset to retrain non-incremental models
        # For now, we'll just fit on the new data.
        model.fit(X, y)

    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune classifier from feedback")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force training even if threshold not met",
    )
    parser.add_argument("--output-dir", default="models", help="Directory to save new model")
    parser.add_argument(
        "--label-field",
        default="true_fine_type",
        choices=["true_fine_type", "true_coarse_type", "true_type"],
        help="Feedback label field used for training targets",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional JSON path to persist training label summary",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help=(
            "DEV/TEST ONLY: fall back to synthetic mock vectors when real "
            "vectors are missing. Must NOT be used in production."
        ),
    )
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
    if args.allow_mock:
        logger.warning("--allow-mock is set: synthetic fallback is enabled (dev/test only).")
    X, y = load_training_data(data_file, label_field=args.label_field, allow_mock=args.allow_mock)
    summary = _build_training_summary(list(y), args.label_field, vector_count=len(X))
    if args.summary_out:
        _write_training_summary(args.summary_out, summary)
        logger.info(f"Saved training summary to {args.summary_out}")
    else:
        logger.info(f"Training label summary: {summary}")

    if len(X) == 0:
        logger.error("No vectors found for training samples")
        return

    # 3. Train
    from src.ml.classifier import get_model_info, reload_model

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
