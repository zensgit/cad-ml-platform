"""Rolling statistics utilities for dynamic thresholding.
Implements an exponential moving average (EMA) with optional warmup.
Used to adapt the confidence fallback threshold based on observed
confidence/calibrated_confidence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RollingStats:
    alpha: float = 0.2  # smoothing factor (0,1]
    _ema: Optional[float] = None
    _count: int = 0

    def update(self, value: float) -> float:
        if self._ema is None:
            self._ema = value
        else:
            self._ema = self.alpha * value + (1 - self.alpha) * self._ema
        self._count += 1
        return self._ema

    @property
    def ema(self) -> Optional[float]:
        return self._ema

    @property
    def count(self) -> int:
        return self._count
