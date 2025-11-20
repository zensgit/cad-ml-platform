"""Centralized OCR configuration (versions & thresholds)."""

from __future__ import annotations

import os

PROMPT_VERSION = os.getenv("OCR_PROMPT_VERSION", "v1")
DATASET_VERSION = os.getenv("OCR_DATASET_VERSION", "v1.0")
DEFAULT_CONFIDENCE_FALLBACK = float(os.getenv("CONFIDENCE_FALLBACK", "0.85"))
