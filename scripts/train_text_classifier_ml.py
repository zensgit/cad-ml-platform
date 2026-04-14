#!/usr/bin/env python3
"""Train a TF-IDF + MLP text classifier on DXF text content (B6.0b).

Replaces/supplements the keyword-based TextContentClassifier with a
learned text classifier that can capture word-frequency patterns
invisible to hand-crafted dictionaries.

Usage:
    python scripts/train_text_classifier_ml.py \
        --manifest data/graph_cache/cache_manifest.csv \
        --output models/text_classifier_tfidf.pth \
        --epochs 80
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ml.text_extractor import extract_text_from_path

logger = logging.getLogger(__name__)


# ── Text vectorisation (TF-IDF via simple token counting) ────────────────────

class SimpleVectorizer:
    """Minimal TF-IDF-like vectorizer (no sklearn dependency).

    Builds a vocabulary from training texts, then converts new texts
    to fixed-dimension count vectors weighted by inverse document frequency.
    """

    def __init__(self, max_features: int = 500, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple Chinese + English tokenization (character n-grams for Chinese)."""
        import re
        tokens = []
        # Split into Chinese and non-Chinese segments
        for segment in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9/]+', text.lower()):
            if all('\u4e00' <= c <= '\u9fff' for c in segment):
                # Chinese: 2-gram sliding window
                for i in range(len(segment) - 1):
                    tokens.append(segment[i:i+2])
                if len(segment) >= 3:
                    for i in range(len(segment) - 2):
                        tokens.append(segment[i:i+3])
            else:
                tokens.append(segment)
        return tokens

    def fit(self, texts: List[str]) -> "SimpleVectorizer":
        """Build vocabulary from training texts."""
        import math
        from collections import Counter

        # Document frequency
        df = Counter()
        for text in texts:
            tokens = set(self._tokenize(text))
            for tok in tokens:
                df[tok] += 1

        # Filter by min_df and select top features
        candidates = [(tok, freq) for tok, freq in df.items() if freq >= self.min_df]
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:self.max_features]

        n_docs = len(texts)
        self.vocab = {tok: i for i, (tok, _) in enumerate(candidates)}
        self.idf = {tok: math.log(n_docs / (freq + 1)) + 1
                     for tok, freq in candidates}
        return self

    def transform(self, text: str) -> torch.Tensor:
        """Convert a single text to a TF-IDF feature vector."""
        from collections import Counter
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        vec = torch.zeros(len(self.vocab))
        for tok, idx in self.vocab.items():
            if tok in counts:
                tf = counts[tok] / max(len(tokens), 1)
                vec[idx] = tf * self.idf.get(tok, 1.0)
        return vec

    @property
    def dim(self) -> int:
        return len(self.vocab)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TextMLDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vectorizer: SimpleVectorizer):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        vec = self.vectorizer.transform(self.texts[idx])
        return vec, self.labels[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class TextMLP(nn.Module):
    """2-layer MLP for TF-IDF text classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 24,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF text classifier (B6.0b)")
    parser.add_argument("--manifest", default="data/graph_cache/cache_manifest.csv")
    parser.add_argument("--output", default="models/text_classifier_tfidf.pth")
    parser.add_argument("--max-features", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0=all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    # Step 1: Extract text from all DXF files
    rows = list(csv.DictReader(open(args.manifest)))
    if args.limit > 0:
        rows = rows[:args.limit]

    label_map: Dict[str, int] = {}
    texts, labels = [], []
    skipped = 0

    logger.info("Extracting text from %d files...", len(rows))
    for i, row in enumerate(rows):
        fp = row.get("file_path", "").strip()
        label = (row.get("taxonomy_v2_class") or row.get("label") or "").strip()
        if not fp or not label:
            continue

        text = extract_text_from_path(fp)
        if not text or len(text.strip()) < 4:
            skipped += 1
            continue

        if label not in label_map:
            label_map[label] = len(label_map)

        texts.append(text)
        labels.append(label_map[label])

        if (i + 1) % 500 == 0:
            logger.info("  [%d/%d] extracted, %d with text", i + 1, len(rows), len(texts))

    logger.info("Text extraction: %d with text, %d skipped (no text)", len(texts), skipped)
    num_classes = len(label_map)

    if len(texts) < 50:
        logger.error("Too few text samples (%d). Need at least 50.", len(texts))
        return

    # Step 2: Build vectorizer
    vectorizer = SimpleVectorizer(max_features=args.max_features, min_df=2)
    vectorizer.fit(texts)
    logger.info("Vocabulary: %d features", vectorizer.dim)

    # Step 3: Train/val split
    dataset = TextMLDataset(texts, labels, vectorizer)
    n = len(dataset)
    n_val = max(1, int(0.2 * n))
    train_ds, val_ds = random_split(dataset, [n - n_val, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Step 4: Train
    model = TextMLP(input_dim=vectorizer.dim, num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tc = tt = 0
        for vecs, lbls in train_loader:
            optimizer.zero_grad()
            logits = model(vecs)
            loss = F.cross_entropy(logits, lbls)
            loss.backward()
            optimizer.step()
            tc += (logits.argmax(1) == lbls).sum().item()
            tt += len(lbls)
        scheduler.step()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for vecs, lbls in val_loader:
                logits = model(vecs)
                vc += (logits.argmax(1) == lbls).sum().item()
                vt += len(lbls)

        train_acc = tc / max(tt, 1)
        val_acc = vc / max(vt, 1)
        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            improved = " ✓"
            torch.save({
                "model_state": model.state_dict(),
                "label_map": label_map,
                "input_dim": vectorizer.dim,
                "hidden_dim": 64,
                "num_classes": num_classes,
                "best_val_acc": best_val_acc,
                "vectorizer_vocab": vectorizer.vocab,
                "vectorizer_idf": vectorizer.idf,
                "max_features": args.max_features,
            }, args.output)
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch <= 3 or improved:
            print(f"  {epoch:>3d}  train={train_acc:.3f}  val={val_acc:.3f}{improved}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest val acc: {best_val_acc*100:.1f}%")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
