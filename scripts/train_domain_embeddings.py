#!/usr/bin/env python3
"""Train manufacturing domain embedding model.

Fine-tunes a sentence-transformer (or TF-IDF fallback) on Chinese
manufacturing terminology using contrastive learning with the
ManufacturingCorpusBuilder.

Usage:
    python scripts/train_domain_embeddings.py
    python scripts/train_domain_embeddings.py --epochs 5 --output models/embeddings/v1
    python scripts/train_domain_embeddings.py --evaluate-only --model-path models/embeddings/v1
    python scripts/train_domain_embeddings.py --demo --model-path models/embeddings/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the project root is on sys.path so bare ``src.*`` imports work
# regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.embeddings.corpus_builder import ManufacturingCorpusBuilder
from src.ml.embeddings.model import DomainEmbeddingModel
from src.ml.embeddings.trainer import DomainEmbeddingTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Corpus building
# -----------------------------------------------------------------------


def build_corpus() -> tuple[list[dict[str, Any]], list[tuple[str, str, float]]]:
    """Build training data and evaluation pairs.

    Returns
    -------
    training_data
        List of ``{anchor, positive, negative}`` dicts produced by
        :class:`ManufacturingCorpusBuilder`.
    eval_pairs
        List of ``(text_a, text_b, expected_similarity)`` triples.
        Synonym pairs get ``expected_similarity = 0.8``; hard negative
        pairs get ``expected_similarity = 0.1``.
    """
    builder = ManufacturingCorpusBuilder()
    builder.build_all()

    training_data = builder.build_training_data()
    logger.info(
        "Built %d training triplets from %d synonym pairs and %d hard negatives",
        len(training_data),
        len(builder.synonym_pairs),
        len(builder.hard_negatives),
    )

    # Evaluation pairs -------------------------------------------------
    eval_pairs: list[tuple[str, str, float]] = []

    # Positive pairs (synonyms) — sample up to 60 to keep eval fast
    synonym_sample = builder.synonym_pairs
    if len(synonym_sample) > 60:
        synonym_sample = random.sample(synonym_sample, 60)
    for a, b in synonym_sample:
        eval_pairs.append((a, b, 0.8))

    # Negative pairs (hard negatives)
    neg_sample = builder.hard_negatives
    if len(neg_sample) > 40:
        neg_sample = random.sample(neg_sample, 40)
    for a, b in neg_sample:
        eval_pairs.append((a, b, 0.1))

    random.shuffle(eval_pairs)
    logger.info("Built %d evaluation pairs", len(eval_pairs))

    return training_data, eval_pairs


def _collect_unique_terms(builder: ManufacturingCorpusBuilder) -> list[str]:
    """Gather every unique term from the corpus builder."""
    terms: set[str] = set()
    for a, b in builder.synonym_pairs:
        terms.add(a)
        terms.add(b)
    for a, b in builder.hard_negatives:
        terms.add(a)
        terms.add(b)
    return sorted(terms)


# -----------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Full training pipeline."""
    print("=" * 60)
    print("  Manufacturing Domain Embedding Training")
    print("=" * 60)

    # 1. Build corpus
    print("\n[1/6] Building training corpus ...")
    training_data, eval_pairs = build_corpus()

    # 2. Export corpus to JSONL for reproducibility
    corpus_path = Path(args.output) / "training_corpus.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as fh:
        for record in training_data:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[2/6] Exported {len(training_data)} records to {corpus_path}")

    # 3. Initialise trainer
    print("[3/6] Initialising trainer ...")
    trainer = DomainEmbeddingTrainer(output_dir=args.output)
    if trainer.is_fallback:
        print("       (using TF-IDF fallback — install sentence-transformers for real training)")

    # 4. Train
    print(f"[4/6] Training for {args.epochs} epoch(s), batch_size={args.batch_size} ...")
    t0 = time.time()
    metrics = trainer.train(
        training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0
    print(f"       Training finished in {elapsed:.1f}s")
    print(f"       Final loss: {metrics.get('final_loss', 'n/a')}")

    # 5. Evaluate
    print("[5/6] Evaluating on held-out pairs ...")
    eval_metrics = trainer.evaluate(eval_pairs)
    metrics["eval"] = eval_metrics

    # 6. Save model
    saved_path = trainer.save()
    print(f"[6/6] Model saved to {saved_path}")

    # Summary
    _print_summary(metrics)
    return metrics


# -----------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate an existing model on manufactured eval pairs."""
    model_path = args.model_path or args.output
    print("=" * 60)
    print("  Evaluate Domain Embedding Model")
    print("=" * 60)
    print(f"  Model path: {model_path}\n")

    if not Path(model_path).exists():
        print(f"ERROR: Model path does not exist: {model_path}")
        print("Train a model first with:  python scripts/train_domain_embeddings.py")
        sys.exit(1)

    # Load model via trainer (supports fallback detection)
    trainer = DomainEmbeddingTrainer()
    trainer.load(model_path)
    if trainer.is_fallback:
        print("  (loaded TF-IDF fallback model)")

    # Build eval pairs
    _, eval_pairs = build_corpus()

    # Evaluate
    eval_metrics = trainer.evaluate(eval_pairs)

    # Detailed breakdown -----------------------------------------------
    builder = ManufacturingCorpusBuilder()
    builder.build_all()

    domain_categories = {
        "materials": ["钢", "铝", "钛", "铜", "铸铁", "尼龙"],
        "processes": ["加工", "铣削", "车削", "铸造", "焊接"],
        "gdt": ["平面度", "圆度", "位置度", "同轴度"],
        "parts": ["法兰", "轴", "齿轮", "轴承", "壳体"],
        "surface_finish": ["粗糙度", "Ra", "光洁度"],
        "tolerances": ["公差", "配合"],
    }

    print("\n  Per-domain accuracy:")
    for domain, keywords in domain_categories.items():
        domain_pairs = [
            p for p in eval_pairs
            if any(kw in p[0] or kw in p[1] for kw in keywords)
        ]
        if domain_pairs:
            sub_metrics = trainer.evaluate(domain_pairs)
            print(f"    {domain:20s}  acc={sub_metrics['accuracy_at_threshold']:.3f}  "
                  f"(n={len(domain_pairs)})")

    print("\n  Overall metrics:")
    for k, v in eval_metrics.items():
        print(f"    {k:30s}  {v}")

    return eval_metrics


# -----------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------


def demo(args: argparse.Namespace) -> None:
    """Interactive demo — type manufacturing queries, see similar terms."""
    model_path = args.model_path or args.output
    print("=" * 60)
    print("  Domain Embedding Demo")
    print("=" * 60)

    if Path(model_path).exists():
        print(f"  Loading model from {model_path} ...")
        model = DomainEmbeddingModel(model_path=model_path)
    else:
        print("  No fine-tuned model found; using base/fallback model.")
        model = DomainEmbeddingModel()

    info = model.get_model_info()
    print(f"  Model: {info['name']}  dim={info['dimension']}  "
          f"fallback={info['fallback']}\n")

    # Build term corpus
    builder = ManufacturingCorpusBuilder()
    builder.build_all()
    corpus = _collect_unique_terms(builder)
    print(f"  Corpus contains {len(corpus)} unique terms.")
    print("  Type a query to find similar terms (Ctrl-C to quit).\n")

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not query:
            continue

        results = model.search(query, corpus, top_k=5)
        for r in results:
            print(f"  [{r['rank']}] {r['score']:+.4f}  {r['text']}")
        print()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _print_summary(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"  Samples trained on : {metrics.get('samples', 'n/a')}")
    print(f"  Epochs             : {metrics.get('epochs', 'n/a')}")
    print(f"  Final loss         : {metrics.get('final_loss', 'n/a')}")
    if metrics.get("fallback"):
        print("  Backend            : TF-IDF fallback (no GPU training)")
    ev = metrics.get("eval", {})
    if ev:
        print(f"  Spearman corr.     : {ev.get('spearman_correlation', 'n/a')}")
        print(f"  Mean error         : {ev.get('mean_error', 'n/a')}")
        print(f"  Accuracy @0.5      : {ev.get('accuracy_at_threshold', 'n/a')}")
    print("=" * 60)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train / evaluate / demo the manufacturing domain embedding model.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--output",
        default="models/embeddings/manufacturing_v1",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation (requires a trained model)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Interactive demo: query similar manufacturing terms",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a trained model (for --evaluate-only or --demo)",
    )
    args = parser.parse_args()

    if args.demo:
        demo(args)
    elif args.evaluate_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
