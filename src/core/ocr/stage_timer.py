"""Utility for fine-grained OCR stage timing (Batch A)."""
from __future__ import annotations

import time
from typing import Dict, Optional


class StageTimer:
    def __init__(self):
        self._t: Dict[str, list[Optional[float]]] = {}

    def start(self, name: str) -> None:
        self._t[name] = [time.time(), None]

    def end(self, name: str) -> None:
        if name in self._t and self._t[name][1] is None:
            self._t[name][1] = time.time()

    def durations_ms(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for k, (s, e) in self._t.items():
            if s is not None and e is not None:
                out[k] = int((e - s) * 1000)
        return out
