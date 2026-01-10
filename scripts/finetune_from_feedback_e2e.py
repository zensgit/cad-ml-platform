#!/usr/bin/env python3
"""Minimal end-to-end demo for fine-tuning from feedback.

This script seeds a labeled sample, registers a vector, exports training data,
loads vectors, and optionally trains a model.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import uuid

# Add project root to path for imports when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.active_learning import get_active_learner, reset_active_learner
from src.core import similarity
from scripts import finetune_from_feedback as finetune


def _reset_vector_store() -> None:
    similarity._VECTOR_STORE.clear()
    similarity._VECTOR_META.clear()
    similarity._VECTOR_TS.clear()
    similarity._VECTOR_LAST_ACCESS.clear()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune from feedback E2E demo")
    parser.add_argument("--dim", type=int, default=32, help="Vector dimension")
    parser.add_argument("--label", default="bolt", help="True label for demo sample")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    args = parser.parse_args()

    os.environ.setdefault("ACTIVE_LEARNING_STORE", "memory")

    reset_active_learner()
    _reset_vector_store()

    doc_id = f"demo_{uuid.uuid4().hex[:8]}"
    learner = get_active_learner()
    sample = learner.flag_for_review(
        doc_id=doc_id,
        predicted_type="unknown",
        confidence=0.2,
        alternatives=[{"type": "bolt", "score": 0.2}],
        score_breakdown={"rule_version": "demo"},
        uncertainty_reason="demo",
    )
    learner.submit_feedback(sample.id, true_type=args.label, reviewer_id="demo")

    rng = random.Random(0)
    vector = [rng.random() for _ in range(args.dim)]
    similarity.register_vector(doc_id, vector, meta={"material": "demo"})

    export_result = learner.export_training_data(format="jsonl")
    data_file = export_result.get("file")
    if not data_file:
        raise SystemExit("Failed to export training data")

    X, y = finetune.load_training_data(data_file)
    print(f"exported={export_result.get('count')} vectors={len(X)} labels={len(y)}")

    if args.skip_train:
        print("training=skipped")
        return 0

    try:
        model = finetune.train_model(X, y)
        print(f"training=ok model={type(model).__name__}")
    except ImportError:
        print("training=skipped reason=sklearn_missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
