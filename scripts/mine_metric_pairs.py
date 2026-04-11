"""Mine metric learning pairs from labeled DXF datasets.

Produces positive pairs, hard-negative pairs, and (anchor, positive, negative)
triplets suitable for contrastive / triplet loss training.

Usage::

    python scripts/mine_metric_pairs.py --data-dir data/training_v8 --output data/metric_learning
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarity matrix between rows of *A* and *B*.

    Both inputs are 2-D arrays of shape ``(n, d)`` and ``(m, d)`` respectively.
    Returns an ``(n, m)`` similarity matrix with values in ``[-1, 1]``.
    """
    # scipy cdist returns *distance*; similarity = 1 - distance
    # Handle zero-norm rows gracefully (they produce nan in cdist).
    A_safe = A.copy()
    B_safe = B.copy()
    A_norms = np.linalg.norm(A_safe, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B_safe, axis=1, keepdims=True)
    # Replace zero rows with tiny epsilon to avoid div-by-zero
    A_safe[A_norms.squeeze() == 0] = 1e-10
    B_safe[B_norms.squeeze() == 0] = 1e-10
    dist = cdist(A_safe, B_safe, metric="cosine")
    return 1.0 - dist


# ---------------------------------------------------------------------------
# MetricPairMiner
# ---------------------------------------------------------------------------

class MetricPairMiner:
    """Mine positive/negative pairs from labeled DXF datasets for metric learning."""

    FEATURE_DIM = 48  # matches extract_features_v6

    def __init__(self, data_dir: str = "data/training_v8", seed: int = 42):
        self.data_dir = Path(data_dir)
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_labeled_files(self) -> Dict[str, List[str]]:
        """Scan data directory for labeled files.

        Returns ``{label: [file_path, ...]}``.

        Supports:
        * Directory-based labelling (each subdirectory is a class).
        * ``manifest.json`` if present (takes precedence).
        """
        result: Dict[str, List[str]] = {}

        if not self.data_dir.exists():
            logger.warning("Data directory %s does not exist.", self.data_dir)
            return result

        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as fh:
                    entries = json.load(fh)
                for entry in entries:
                    cat = entry.get("category", "unknown")
                    fpath = str(self.data_dir / entry["file"])
                    result.setdefault(cat, []).append(fpath)
                logger.info(
                    "Loaded %d files from manifest.json across %d classes.",
                    sum(len(v) for v in result.values()),
                    len(result),
                )
                return result
            except Exception as exc:
                logger.warning("Failed to parse manifest.json, falling back to directory scan: %s", exc)

        # Fallback: directory-based labelling
        for child in sorted(self.data_dir.iterdir()):
            if child.is_dir():
                label = child.name
                files = sorted(
                    str(f) for f in child.iterdir()
                    if f.is_file() and f.suffix.lower() in {".dxf", ".DXF"}
                )
                if files:
                    result[label] = files

        logger.info(
            "Scanned %d files across %d classes from directory structure.",
            sum(len(v) for v in result.values()),
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Extract 48-dim features from DXF files.

        Returns ``[{"path": str, "label": str, "features": np.ndarray}, ...]``.
        Files that fail to parse are silently skipped.
        """
        results: List[Dict[str, Any]] = []
        try:
            from src.utils.dxf_features import extract_features_v6
        except ImportError:
            logger.warning(
                "Could not import extract_features_v6 -- returning empty batch. "
                "Use inject_features() to supply pre-computed feature vectors."
            )
            return results

        for fpath in file_paths:
            try:
                vec = extract_features_v6(fpath)
                if vec is not None:
                    # Derive label from parent directory name
                    label = Path(fpath).parent.name
                    results.append({"path": fpath, "label": label, "features": vec})
            except Exception as exc:
                logger.debug("Skipping %s: %s", fpath, exc)
        return results

    @staticmethod
    def inject_features(
        label_to_paths: Dict[str, List[str]],
        feature_map: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Build a features list from a pre-computed *feature_map*.

        Useful when feature extraction has already been done or during testing
        with synthetic vectors.

        Args:
            label_to_paths: ``{label: [path, ...]}`` as returned by :meth:`scan_labeled_files`.
            feature_map: ``{path: np.ndarray}`` mapping file path to feature vector.

        Returns:
            A list of dicts compatible with the mining methods.
        """
        results: List[Dict[str, Any]] = []
        for label, paths in label_to_paths.items():
            for p in paths:
                vec = feature_map.get(p)
                if vec is not None:
                    results.append({"path": p, "label": label, "features": vec})
        return results

    # ------------------------------------------------------------------
    # Mining: positive pairs
    # ------------------------------------------------------------------

    def mine_positive_pairs(
        self,
        features: List[Dict[str, Any]],
        pairs_per_class: int = 200,
    ) -> List[Dict[str, Any]]:
        """Mine positive pairs (same class).

        For each class, randomly samples up to *pairs_per_class* pairs of files.
        Cosine similarity is stored alongside each pair.
        """
        by_class: Dict[str, List[Dict[str, Any]]] = {}
        for item in features:
            by_class.setdefault(item["label"], []).append(item)

        pairs: List[Dict[str, Any]] = []
        for label, items in by_class.items():
            if len(items) < 2:
                continue

            # All possible index pairs
            n = len(items)
            all_combos = list(combinations(range(n), 2))
            self.rng.shuffle(all_combos)
            selected = all_combos[: pairs_per_class]

            vecs = np.array([it["features"] for it in items], dtype=np.float64)

            for i, j in selected:
                sim = float(_cosine_similarity_matrix(
                    vecs[i : i + 1], vecs[j : j + 1]
                )[0, 0])
                pairs.append({
                    "anchor": items[i]["path"],
                    "positive": items[j]["path"],
                    "label": label,
                    "similarity": round(sim, 6),
                })
        return pairs

    # ------------------------------------------------------------------
    # Mining: hard negatives
    # ------------------------------------------------------------------

    def mine_hard_negatives(
        self,
        features: List[Dict[str, Any]],
        negatives_per_anchor: int = 3,
    ) -> List[Dict[str, Any]]:
        """Mine hard-negative pairs (different class but geometrically similar).

        For each sample, finds nearest neighbours from *other* classes using
        cosine similarity and keeps the top *negatives_per_anchor*.
        """
        if not features:
            return []

        labels = np.array([f["label"] for f in features])
        vecs = np.array([f["features"] for f in features], dtype=np.float64)
        sim_matrix = _cosine_similarity_matrix(vecs, vecs)

        pairs: List[Dict[str, Any]] = []
        for idx in range(len(features)):
            anchor = features[idx]
            # Mask: only consider items from different classes
            mask = labels != anchor["label"]
            if not mask.any():
                continue

            sims = sim_matrix[idx].copy()
            sims[~mask] = -2.0  # exclude same-class items

            # Top-k most similar from other classes
            candidates = np.argsort(sims)[::-1][:negatives_per_anchor]
            for c_idx in candidates:
                if sims[c_idx] <= -2.0:
                    break  # no more valid candidates
                neg = features[int(c_idx)]
                pairs.append({
                    "anchor": anchor["path"],
                    "negative": neg["path"],
                    "anchor_label": anchor["label"],
                    "negative_label": neg["label"],
                    "similarity": round(float(sims[c_idx]), 6),
                })
        return pairs

    # ------------------------------------------------------------------
    # Mining: triplets
    # ------------------------------------------------------------------

    def mine_triplets(
        self,
        features: List[Dict[str, Any]],
        triplets_per_class: int = 100,
    ) -> List[Dict[str, Any]]:
        """Mine ``(anchor, positive, negative)`` triplets.

        Strategy: for each class, sample *triplets_per_class* anchor-positive
        pairs (same class), then pick the hardest negative (most similar item
        from a different class).
        """
        if not features:
            return []

        by_class: Dict[str, List[int]] = {}
        for i, item in enumerate(features):
            by_class.setdefault(item["label"], []).append(i)

        labels = np.array([f["label"] for f in features])
        vecs = np.array([f["features"] for f in features], dtype=np.float64)
        sim_matrix = _cosine_similarity_matrix(vecs, vecs)

        triplets: List[Dict[str, Any]] = []
        for label, indices in by_class.items():
            if len(indices) < 2:
                continue

            all_combos = list(combinations(indices, 2))
            self.rng.shuffle(all_combos)
            selected = all_combos[: triplets_per_class]

            for a_idx, p_idx in selected:
                # Hardest negative: most similar from other class
                mask = labels != label
                if not mask.any():
                    continue

                sims_a = sim_matrix[a_idx].copy()
                sims_a[~mask] = -2.0
                n_idx = int(np.argmax(sims_a))
                if sims_a[n_idx] <= -2.0:
                    continue

                pos_sim = float(sim_matrix[a_idx, p_idx])
                neg_sim = float(sims_a[n_idx])
                margin = pos_sim - neg_sim

                triplets.append({
                    "anchor": features[a_idx]["path"],
                    "positive": features[p_idx]["path"],
                    "negative": features[n_idx]["path"],
                    "anchor_label": label,
                    "positive_similarity": round(pos_sim, 6),
                    "negative_similarity": round(neg_sim, 6),
                    "margin": round(margin, 6),
                })
        return triplets

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pairs(self, output_path: str, pairs: List[Dict[str, Any]]) -> int:
        """Save mined pairs to a JSONL file.  Returns the number of records written."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            for rec in pairs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote %d records to %s", len(pairs), out)
        return len(pairs)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def generate_training_set(
        self,
        output_dir: str,
        positive_per_class: int = 200,
        negatives_per_anchor: int = 3,
        triplets_per_class: int = 100,
    ) -> Dict[str, Any]:
        """Full pipeline: scan -> extract -> mine -> save.

        Saves:
            * ``output_dir/positive_pairs.jsonl``
            * ``output_dir/hard_negatives.jsonl``
            * ``output_dir/triplets.jsonl``
            * ``output_dir/manifest.json`` (statistics)

        If the data directory does not exist or contains no parseable files, the
        method logs a warning and returns a stats dict with zero counts.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        label_to_paths = self.scan_labeled_files()
        if not label_to_paths:
            logger.warning(
                "No labeled files found in %s. "
                "Pass a valid --data-dir or place DXF files in category sub-directories.",
                self.data_dir,
            )
            stats = self._empty_stats(output_dir)
            self._write_manifest(out / "manifest.json", stats)
            return stats

        # Flatten file paths for batch extraction
        all_paths: List[str] = []
        for paths in label_to_paths.values():
            all_paths.extend(paths)

        features = self.extract_features_batch(all_paths)

        if not features:
            logger.warning(
                "Feature extraction returned 0 vectors (ezdxf may not be installed). "
                "Consider using inject_features() with pre-computed vectors."
            )
            stats = self._empty_stats(output_dir)
            self._write_manifest(out / "manifest.json", stats)
            return stats

        logger.info("Extracted features for %d / %d files.", len(features), len(all_paths))

        # Mine
        pos_pairs = self.mine_positive_pairs(features, pairs_per_class=positive_per_class)
        neg_pairs = self.mine_hard_negatives(features, negatives_per_anchor=negatives_per_anchor)
        triplets = self.mine_triplets(features, triplets_per_class=triplets_per_class)

        # Save
        n_pos = self.save_pairs(str(out / "positive_pairs.jsonl"), pos_pairs)
        n_neg = self.save_pairs(str(out / "hard_negatives.jsonl"), neg_pairs)
        n_tri = self.save_pairs(str(out / "triplets.jsonl"), triplets)

        # Class distribution
        class_counts = {}
        for f in features:
            class_counts[f["label"]] = class_counts.get(f["label"], 0) + 1

        stats: Dict[str, Any] = {
            "data_dir": str(self.data_dir),
            "output_dir": str(out),
            "total_files_scanned": len(all_paths),
            "total_files_with_features": len(features),
            "positive_pairs": n_pos,
            "hard_negatives": n_neg,
            "triplets": n_tri,
            "total_pairs": n_pos + n_neg + n_tri,
            "class_distribution": class_counts,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_manifest(out / "manifest.json", stats)
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_stats(output_dir: str) -> Dict[str, Any]:
        return {
            "data_dir": "",
            "output_dir": output_dir,
            "total_files_scanned": 0,
            "total_files_with_features": 0,
            "positive_pairs": 0,
            "hard_negatives": 0,
            "triplets": 0,
            "total_pairs": 0,
            "class_distribution": {},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _write_manifest(path: Path, stats: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
        logger.info("Manifest written to %s", path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine metric learning pairs from labeled DXF data",
    )
    parser.add_argument("--data-dir", default="data/training_v8",
                        help="Root directory containing labeled DXF files (default: data/training_v8)")
    parser.add_argument("--output", default="data/metric_learning",
                        help="Output directory for mined pairs (default: data/metric_learning)")
    parser.add_argument("--positive-per-class", type=int, default=200,
                        help="Number of positive pairs to mine per class (default: 200)")
    parser.add_argument("--negatives-per-anchor", type=int, default=3,
                        help="Number of hard negatives per anchor sample (default: 3)")
    parser.add_argument("--triplets-per-class", type=int, default=100,
                        help="Number of triplets to mine per class (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    miner = MetricPairMiner(data_dir=args.data_dir, seed=args.seed)
    stats = miner.generate_training_set(
        output_dir=args.output,
        positive_per_class=args.positive_per_class,
        negatives_per_anchor=args.negatives_per_anchor,
        triplets_per_class=args.triplets_per_class,
    )

    print(f"\nMined {stats['total_pairs']} pairs total:")
    print(f"  Positive pairs:  {stats['positive_pairs']}")
    print(f"  Hard negatives:  {stats['hard_negatives']}")
    print(f"  Triplets:        {stats['triplets']}")
    print(f"  Classes:         {list(stats['class_distribution'].keys())}")
    print(f"  Output:          {stats['output_dir']}")


if __name__ == "__main__":
    main()
