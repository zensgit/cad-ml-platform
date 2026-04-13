"""Prompt templates for local LLM inference.

Optimized for smaller context windows (7B models via vLLM).
Bilingual: Chinese primary, English fallback.
"""

from .cad_system_prompt import get_cad_system_prompt
from .classification_prompt import get_classification_prompt
from .ocr_extraction_prompt import get_ocr_extraction_prompt

__all__ = [
    "get_cad_system_prompt",
    "get_classification_prompt",
    "get_ocr_extraction_prompt",
]
