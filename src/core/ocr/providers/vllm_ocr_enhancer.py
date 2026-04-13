"""vLLM OCR Enhancement Provider.

Post-processes raw PaddleOCR text through local vLLM for structured extraction
of title block fields (part_name, material, dimensions, tolerances, etc.).

Gated by ``vllm_ocr_enhancement_enabled`` feature flag.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_feature_flag(flag_name: str) -> bool:
    """Check a feature flag from config/feature_flags.json."""
    try:
        flags_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "config", "feature_flags.json"
        )
        flags_path = os.path.normpath(flags_path)
        with open(flags_path) as f:
            data = json.load(f)
        for flag in data.get("flags", []):
            if flag.get("name") == flag_name:
                return flag.get("enabled", False)
    except Exception:
        pass
    return False


class VLLMOcrEnhancer:
    """
    Enhances raw OCR results using local vLLM for structured extraction.

    Takes raw OCR text from PaddleOCR and sends it to vLLM with a
    domain-specific prompt to extract structured title block fields.
    """

    def __init__(
        self,
        vllm_endpoint: Optional[str] = None,
        vllm_model: Optional[str] = None,
        timeout: int = 30,
    ):
        self._endpoint = vllm_endpoint or os.getenv(
            "VLLM_ENDPOINT", "http://localhost:8100"
        )
        self._model = vllm_model or os.getenv(
            "VLLM_MODEL", "deepseek-coder-6.7b-awq"
        )
        self._timeout = timeout

    def is_enabled(self) -> bool:
        """Check if vLLM OCR enhancement is enabled."""
        return _load_feature_flag("vllm_ocr_enhancement_enabled")

    def _check_health(self) -> bool:
        """Check if vLLM server is reachable."""
        try:
            import requests

            resp = requests.get(f"{self._endpoint}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def _call_vllm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request to vLLM."""
        import requests

        response = requests.post(
            f"{self._endpoint}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 512,
                "stream": False,
            },
            timeout=self._timeout,
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"vLLM error: {response.status_code}")

    def _parse_extraction(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured dict."""
        text = response_text.strip()
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        return {}

    async def enhance(
        self,
        raw_ocr_text: str,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enhance raw OCR text with structured extraction via vLLM.

        Args:
            raw_ocr_text: Raw OCR text from PaddleOCR
            trace_id: Optional trace ID for logging

        Returns:
            Dict with extracted fields. Empty dict if enhancement fails/skipped.
            Keys may include: part_name, material, drawing_number, quantity,
            scale, revision, date, weight, surface_finish, designer.
        """
        if not raw_ocr_text or not raw_ocr_text.strip():
            return {}

        # Gate: feature flag
        if not self.is_enabled():
            logger.debug("vllm_ocr_enhancement_enabled flag is off, skipping")
            return {}

        # Gate: health check
        if not self._check_health():
            logger.warning("vLLM server unreachable, skipping OCR enhancement")
            return {}

        # Import prompt template
        from src.core.assistant.prompts.ocr_extraction_prompt import (
            get_ocr_extraction_prompt,
        )

        system_prompt = (
            "你是OCR标题栏信息提取专家。从OCR文本中提取结构化字段，输出纯JSON。"
        )
        user_prompt = get_ocr_extraction_prompt(raw_ocr_text)

        start = time.monotonic()
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None, self._call_vllm, system_prompt, user_prompt
            )
            latency_ms = (time.monotonic() - start) * 1000
            logger.info(
                "vllm_ocr_enhancer.enhance",
                extra={
                    "latency_ms": f"{latency_ms:.1f}",
                    "trace_id": trace_id,
                },
            )
            return self._parse_extraction(response_text)
        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            logger.warning(
                "vLLM OCR enhancement failed (%.1fms): %s",
                latency_ms,
                str(e),
            )
            return {}
