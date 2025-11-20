"""Prompt templates for OCR providers.
Centralizes prompt text and embeds versioning to make caching and
behavior reproducible. Keep prompts concise and deterministic.
"""

from __future__ import annotations

from ..config import PROMPT_VERSION


def deepseek_ocr_json_prompt() -> str:
    """Return the DeepSeek OCR extraction prompt.

    Embeds `PROMPT_VERSION` to couple output schema with cache keying.
    Output contract:
      - Primary mode: strict JSON (no comments) matching the schema
      - If incapable, wrap JSON in a single fenced code block ```json ... ```
      - Avoid prose outside JSON
    """
    return (
        "You are an OCR post-processor for CAD drawings. "
        f"Schema version: {PROMPT_VERSION}. "
        "Extract dimensions, tolerances, threads, and symbols. "
        "Respond ONLY with JSON using keys: dimensions[], symbols[], title_block{}. "
        "Each dimension: {type: diameter|radius|thread, value: number, "
        "tolerance: number|null, tol_pos: number|null, tol_neg: number|null, "
        'pitch: number|null, unit: "mm"}. '
        "Each symbol: {type: surface_roughness|perpendicularity|parallelism, value: string}. "
        'Example minimal output: {"dimensions":[],"symbols":[],"title_block":{}}. '
        "If the model cannot produce strict JSON, then return a single ```json fenced block containing ONLY the JSON."
    )
