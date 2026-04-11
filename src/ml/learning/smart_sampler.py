"""
Smart Sampler -- intelligent active learning sample selection.

Implements multiple query strategies that identify the most informative
samples for human labeling, replacing naive FIFO sampling with strategies
grounded in active-learning research:

* **Uncertainty sampling** -- lowest max-confidence.
* **Margin sampling** -- smallest gap between top-2 class probabilities.
* **Entropy sampling** -- highest Shannon entropy across all classes.
* **Disagreement sampling** -- maximum classifier branch disagreement.
* **Diversity sampling** -- coverage across feature space via mini-batch k-means.
* **Combined sampling** -- weighted blend of all strategies above.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Branch keys used in per-sample predictions dict.
_BRANCH_KEYS = [
    "filename_pred",
    "graph2d_pred",
    "titleblock_pred",
    "process_pred",
    "history_sequence_pred",
]


def _safe_probs(pred: dict[str, Any]) -> np.ndarray:
    """Extract class probability vector from a prediction dict.

    Looks for ``"class_probs"`` (list/dict of floats) or falls back to
    a single-element array from ``"confidence"``.
    """
    raw = pred.get("class_probs")
    if raw is not None:
        if isinstance(raw, dict):
            vals = list(raw.values())
        elif isinstance(raw, (list, tuple)):
            vals = list(raw)
        else:
            vals = [float(raw)]
        arr = np.array(vals, dtype=np.float64)
        total = arr.sum()
        if total > 0:
            arr = arr / total
        return arr

    conf = float(pred.get("confidence", 0.5))
    conf = np.clip(conf, 0.0, 1.0)
    return np.array([conf, 1.0 - conf], dtype=np.float64)


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution (nats)."""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log(p)))


def _margin(probs: np.ndarray) -> float:
    """Difference between top-1 and top-2 class probabilities."""
    if len(probs) < 2:
        return 1.0  # only one class -- maximally certain
    sorted_p = np.sort(probs)[::-1]
    return float(sorted_p[0] - sorted_p[1])


class SmartSampler:
    """Selects the most informative samples for human labeling."""

    # ------------------------------------------------------------------
    # Uncertainty sampling
    # ------------------------------------------------------------------

    def uncertainty_sampling(
        self, predictions: list[dict[str, Any]], k: int = 10
    ) -> list[dict[str, Any]]:
        """Pick *k* samples with the lowest max confidence.

        Lower confidence -> higher uncertainty -> more informative.
        """
        if not predictions:
            return []
        k = min(k, len(predictions))

        scored = []
        for pred in predictions:
            probs = _safe_probs(pred)
            max_conf = float(probs.max()) if len(probs) > 0 else 0.5
            scored.append((max_conf, pred))

        # Sort ascending -- lowest confidence first.
        scored.sort(key=lambda t: t[0])
        return [pred for _, pred in scored[:k]]

    # ------------------------------------------------------------------
    # Margin sampling
    # ------------------------------------------------------------------

    def margin_sampling(
        self, predictions: list[dict[str, Any]], k: int = 10
    ) -> list[dict[str, Any]]:
        """Pick *k* samples with the smallest margin between top-2 predictions.

        Small margin -> model is torn between two classes -> informative.
        """
        if not predictions:
            return []
        k = min(k, len(predictions))

        scored = []
        for pred in predictions:
            probs = _safe_probs(pred)
            m = _margin(probs)
            scored.append((m, pred))

        scored.sort(key=lambda t: t[0])
        return [pred for _, pred in scored[:k]]

    # ------------------------------------------------------------------
    # Entropy sampling
    # ------------------------------------------------------------------

    def entropy_sampling(
        self, predictions: list[dict[str, Any]], k: int = 10
    ) -> list[dict[str, Any]]:
        """Pick *k* samples with the highest prediction entropy.

        High entropy -> class distribution is spread out -> informative.
        """
        if not predictions:
            return []
        k = min(k, len(predictions))

        scored = []
        for pred in predictions:
            probs = _safe_probs(pred)
            h = _entropy(probs)
            scored.append((h, pred))

        # Sort descending -- highest entropy first.
        scored.sort(key=lambda t: t[0], reverse=True)
        return [pred for _, pred in scored[:k]]

    # ------------------------------------------------------------------
    # Disagreement sampling (query-by-committee)
    # ------------------------------------------------------------------

    def disagreement_sampling(
        self, predictions: list[dict[str, Any]], k: int = 10
    ) -> list[dict[str, Any]]:
        """Pick *k* samples where classifier branches disagree most.

        Each prediction dict may contain keys like ``filename_pred``,
        ``graph2d_pred``, etc.  We measure disagreement as the number of
        distinct labels across branches.  Ties are broken by lower overall
        confidence.
        """
        if not predictions:
            return []
        k = min(k, len(predictions))

        scored = []
        for pred in predictions:
            labels = []
            for key in _BRANCH_KEYS:
                val = pred.get(key)
                if val is not None:
                    labels.append(str(val))

            if len(labels) <= 1:
                # Cannot measure disagreement with 0-1 branches.
                disagreement = 0.0
            else:
                n_unique = len(set(labels))
                # Normalise to [0, 1]: 1 unique / N branches = 0, N unique / N = 1.
                disagreement = (n_unique - 1) / (len(labels) - 1)

            conf = float(pred.get("confidence", 0.5))
            # We want: high disagreement first, low confidence as tiebreaker.
            scored.append((disagreement, -conf, pred))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [pred for _, _, pred in scored[:k]]

    # ------------------------------------------------------------------
    # Diversity sampling
    # ------------------------------------------------------------------

    def diversity_sampling(
        self, predictions: list[dict[str, Any]], k: int = 10
    ) -> list[dict[str, Any]]:
        """Pick *k* samples covering diverse regions of feature space.

        Uses mini-batch k-means on the class probability vectors.  One
        sample closest to each cluster centroid is selected.  If
        scikit-learn is not available, falls back to stratified random
        sampling across predicted labels.
        """
        if not predictions:
            return []
        k = min(k, len(predictions))

        # Build feature matrix from class_probs or confidence.
        features: list[np.ndarray] = []
        max_dim = 0
        for pred in predictions:
            probs = _safe_probs(pred)
            features.append(probs)
            max_dim = max(max_dim, len(probs))

        if max_dim == 0:
            return predictions[:k]

        # Pad all vectors to the same dimensionality.
        matrix = np.zeros((len(predictions), max_dim), dtype=np.float64)
        for i, feat in enumerate(features):
            matrix[i, : len(feat)] = feat

        try:
            return self._kmeans_diversity(predictions, matrix, k)
        except Exception:
            logger.debug(
                "k-means diversity sampling failed; using stratified fallback.",
                exc_info=True,
            )
            return self._stratified_fallback(predictions, k)

    def _kmeans_diversity(
        self,
        predictions: list[dict[str, Any]],
        matrix: np.ndarray,
        k: int,
    ) -> list[dict[str, Any]]:
        """Select diverse samples using mini-batch k-means."""
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = min(k, len(predictions))
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, batch_size=min(100, len(predictions))
        )
        kmeans.fit(matrix)
        labels = kmeans.labels_

        selected_indices: list[int] = []
        for cluster_id in range(n_clusters):
            members = np.where(labels == cluster_id)[0]
            if len(members) == 0:
                continue
            # Pick the member closest to the centroid.
            centroid = kmeans.cluster_centers_[cluster_id]
            dists = np.linalg.norm(matrix[members] - centroid, axis=1)
            best = members[int(np.argmin(dists))]
            selected_indices.append(int(best))

        # If we still need more (some clusters may have been empty), fill in.
        if len(selected_indices) < k:
            remaining = [
                i for i in range(len(predictions)) if i not in set(selected_indices)
            ]
            np.random.seed(42)
            extra = list(
                np.random.choice(
                    remaining,
                    size=min(k - len(selected_indices), len(remaining)),
                    replace=False,
                )
            )
            selected_indices.extend(int(x) for x in extra)

        return [predictions[i] for i in selected_indices[:k]]

    def _stratified_fallback(
        self,
        predictions: list[dict[str, Any]],
        k: int,
    ) -> list[dict[str, Any]]:
        """Stratified random sampling across predicted labels."""
        by_label: dict[str, list[dict[str, Any]]] = {}
        for pred in predictions:
            label = str(pred.get("label", "unknown"))
            by_label.setdefault(label, []).append(pred)

        selected: list[dict[str, Any]] = []
        rng = np.random.RandomState(42)
        labels = sorted(by_label.keys())
        per_label = max(1, k // len(labels)) if labels else k

        for label in labels:
            pool = by_label[label]
            n = min(per_label, len(pool))
            indices = rng.choice(len(pool), size=n, replace=False)
            for idx in indices:
                selected.append(pool[int(idx)])
            if len(selected) >= k:
                break

        return selected[:k]

    # ------------------------------------------------------------------
    # Combined sampling
    # ------------------------------------------------------------------

    def combined_sampling(
        self,
        predictions: list[dict[str, Any]],
        k: int = 10,
        strategy_weights: Optional[dict[str, float]] = None,
    ) -> list[dict[str, Any]]:
        """Combine multiple strategies with configurable weights.

        Each strategy assigns a rank-based score to every sample.  The
        final score is the weighted sum of ranks.  The *k* samples with
        the highest aggregate score are returned.

        Parameters
        ----------
        strategy_weights:
            Mapping of strategy name to weight.  Defaults to equal weight.
        """
        if not predictions:
            return []
        k = min(k, len(predictions))

        weights = strategy_weights or {
            "uncertainty": 0.3,
            "margin": 0.25,
            "entropy": 0.25,
            "disagreement": 0.2,
        }

        # Collect per-strategy orderings.  Each strategy returns samples
        # sorted by informativeness (best first).  We convert that to a
        # rank score: best sample gets len(predictions), worst gets 1.
        n = len(predictions)

        # Create a stable id -> index mapping.
        id_map: dict[int, int] = {id(p): i for i, p in enumerate(predictions)}

        aggregate_scores = np.zeros(n, dtype=np.float64)

        strategies: dict[str, Any] = {
            "uncertainty": self.uncertainty_sampling,
            "margin": self.margin_sampling,
            "entropy": self.entropy_sampling,
            "disagreement": self.disagreement_sampling,
        }

        for name, func in strategies.items():
            w = weights.get(name, 0.0)
            if w <= 0:
                continue
            ranked = func(predictions, k=n)  # full ranking
            for rank, pred in enumerate(ranked):
                idx = id_map.get(id(pred))
                if idx is not None:
                    # Highest rank score = most informative.
                    aggregate_scores[idx] += w * (n - rank)

        # Select top-k by aggregate score.
        top_indices = np.argsort(aggregate_scores)[::-1][:k]
        return [predictions[int(i)] for i in top_indices]
