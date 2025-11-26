"""Drift calculation utilities (lightweight PSI approximation).

All functions degrade gracefully without raising; used by analysis pipeline and drift endpoint.
"""

from __future__ import annotations

from typing import Dict, Iterable
import math

EPS = 1e-9

def _distribution(items: Iterable[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total = 0
    for it in items:
        counts[it] = counts.get(it, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def psi_score(current: Dict[str, float], baseline: Dict[str, float]) -> float:
    """Compute simplified Population Stability Index between two categorical distributions.

    Returns value in [0, ~]. We clamp to [0,1] for monitoring convenience.
    Formula (simplified): sum( (p - q) * ln( (p + EPS) / (q + EPS) ) ).
    """
    score = 0.0
    keys = set(baseline.keys()) | set(current.keys())
    for k in keys:
        p = current.get(k, EPS)
        q = baseline.get(k, EPS)
        score += (p - q) * math.log((p + EPS) / (q + EPS))
    if score < 0:
        score = abs(score)  # direction not critical for monitoring
    # Clamp for histogram buckets
    return min(score, 1.0)


def compute_drift(current_items: Iterable[str], baseline_items: Iterable[str]) -> float:
    cur_dist = _distribution(current_items)
    base_dist = _distribution(baseline_items)
    if not cur_dist or not base_dist:
        return 0.0
    return psi_score(cur_dist, base_dist)


__all__ = ["compute_drift", "psi_score"]
