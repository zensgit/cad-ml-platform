"""Multi-evidence confidence calibration (lightweight placeholder).

Combines raw provider confidence, parsing completeness and optional coverage
into a calibrated value using weighted average; configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvidenceWeights:
    w_raw: float = 0.5
    w_completeness: float = 0.25
    w_item_mean: float = 0.15
    w_fallback_recent: float = 0.05
    w_parse_error: float = 0.05


class MultiEvidenceCalibrator:
    """Calibrate confidence using multiple evidence signals.

    Evidence:
      - raw_confidence: provider-level (model or heuristic)
      - completeness: parsed key-field coverage ratio
      - item_mean: mean per-item confidence (dimensions + symbols)
      - fallback_recent: proportion of recent requests triggering fallback (penalize high values)
      - parse_error_rate: ratio of parse errors (penalize high values)
    """

    def __init__(self, weights: EvidenceWeights | None = None):
        self.weights = weights or EvidenceWeights()

    def calibrate(
        self,
        raw_confidence: Optional[float],
        completeness: Optional[float],
        item_mean: Optional[float] = None,
        fallback_recent: Optional[float] = None,
        parse_error_rate: Optional[float] = None,
    ) -> Optional[float]:
        evidences = []
        w = self.weights
        # Raw + completeness direct contributions
        if raw_confidence is not None:
            evidences.append((raw_confidence, w.w_raw))
        if completeness is not None:
            evidences.append((completeness, w.w_completeness))
        if item_mean is not None:
            evidences.append((item_mean, w.w_item_mean))
        # Penalize fallback_recent & parse_error_rate by mapping high values to low score
        if fallback_recent is not None:
            # map [0,1] -> [1,0] with smooth decay
            penalized = 1.0 - min(1.0, max(0.0, fallback_recent))
            evidences.append((penalized, w.w_fallback_recent))
        if parse_error_rate is not None:
            penalized = 1.0 - min(1.0, max(0.0, parse_error_rate))
            evidences.append((penalized, w.w_parse_error))
        if not evidences:
            return None
        total_w = sum(wt for _, wt in evidences)
        score = sum(val * wt for val, wt in evidences) / total_w
        # Clamp and slight smoothing toward midrange to avoid overconfidence early
        return max(0.0, min(1.0, score))

    def adaptive_reweight(self, observed_brier: float) -> None:
        """Optional adaptive adjustment based on observed Brier score.

        If Brier is poor (>0.3), increase completeness and item_mean weights
        to emphasize structural reliability; if very good (<0.15), lean more on raw.
        """
        if observed_brier > 0.3:
            self.weights.w_completeness = min(0.35, self.weights.w_completeness + 0.05)
            self.weights.w_item_mean = min(0.25, self.weights.w_item_mean + 0.05)
            # reduce raw slightly
            self.weights.w_raw = max(0.4, self.weights.w_raw - 0.05)
        elif observed_brier < 0.15:
            self.weights.w_raw = min(0.6, self.weights.w_raw + 0.05)
            # reduce completeness & item_mean slowly
            self.weights.w_completeness = max(0.2, self.weights.w_completeness - 0.02)
            self.weights.w_item_mean = max(0.1, self.weights.w_item_mean - 0.02)
