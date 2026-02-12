"""DeepSeek HF provider with lazy model load and fallback parsing.

Degrades gracefully to stub if transformers not available.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

"""DeepSeek HF OCR provider stub."""
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # transformers not installed
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from src.core.errors import ErrorCode
from src.utils.metrics import ocr_cold_start_seconds, ocr_errors_total, ocr_stage_duration_seconds

from ..base import (
    DimensionInfo,
    DimensionType,
    OcrClient,
    OcrResult,
    SymbolInfo,
    SymbolType,
    TitleBlock,
)
from ..parsing.bbox_mapper import assign_bboxes, polygon_to_bbox
from ..parsing.dimension_parser import parse_dimensions_and_symbols
from ..parsing.fallback_parser import FallbackParser
from ..parsing.title_block_parser import parse_title_block
from ..preprocessing.image_enhancer import enhance_image_for_ocr
from ..stage_timer import StageTimer
from ..utils.prompt_templates import deepseek_ocr_json_prompt

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None  # type: ignore


logger = logging.getLogger(__name__)


def _is_commit_hash(value: str) -> bool:
    return len(value) >= 7 and all(c in "0123456789abcdefABCDEF" for c in value)


class DeepSeekHfProvider(OcrClient):
    name = "deepseek_hf"

    def __init__(
        self,
        timeout_ms: int = 30000,
        model_name: str = "deepseek-ocr-mini",
        align_with_paddle: bool = True,
    ):
        self._model = None
        self._tokenizer = None
        self._lock: Optional[asyncio.Lock] = None
        self.timeout_ms = timeout_ms
        self.model_name = os.getenv("DEEPSEEK_HF_MODEL", model_name)
        self._revision = os.getenv("DEEPSEEK_HF_REVISION", "").strip()
        self._allow_unpinned = os.getenv("DEEPSEEK_HF_ALLOW_UNPINNED", "0") == "1"
        self._parser = FallbackParser()
        self._align_with_paddle = align_with_paddle
        self._paddle = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy init lock for Python 3.9 compatibility."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def warmup(self) -> None:
        if not self._model:
            await self._lazy_load()
        if self._align_with_paddle and PaddleOCR and self._paddle is None:
            try:
                self._paddle = PaddleOCR(lang="ch", use_angle_cls=True, use_gpu=False)
            except Exception:
                self._paddle = None

    async def _lazy_load(self):
        async with self._get_lock():
            if self._model is None:
                # If transformers are unavailable, record as a load error and fall back to stub.
                if not (AutoModelForCausalLM and AutoTokenizer):
                    ocr_errors_total.labels(
                        provider=self.name, code=ErrorCode.MODEL_LOAD_ERROR.value, stage="load"
                    ).inc()
                    logger.warning("Transformers backend not available; using stub model")
                    self._model = "stub"
                    return

                try:
                    start = time.time()
                    if not self._revision:
                        if not self._allow_unpinned:
                            ocr_errors_total.labels(
                                provider=self.name,
                                code=ErrorCode.MODEL_LOAD_ERROR.value,
                                stage="load",
                            ).inc()
                            logger.error("DEEPSEEK_HF_REVISION not set; refusing unpinned load")
                            self._model = "stub"
                            return
                        self._revision = "main"
                    if not _is_commit_hash(self._revision) and not self._allow_unpinned:
                        ocr_errors_total.labels(
                            provider=self.name,
                            code=ErrorCode.MODEL_LOAD_ERROR.value,
                            stage="load",
                        ).inc()
                        logger.error("DEEPSEEK_HF_REVISION is not a commit hash; refusing load")
                        self._model = "stub"
                        return
                    if not _is_commit_hash(self._revision):
                        logger.warning(
                            "DEEPSEEK_HF_REVISION is not a commit hash; using unpinned revision"
                        )
                    self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                        self.model_name,
                        revision=self._revision,
                    )
                    self._model = AutoModelForCausalLM.from_pretrained(  # nosec B615
                        self.model_name,
                        revision=self._revision,
                    )
                    ocr_cold_start_seconds.labels(provider=self.name).set(time.time() - start)
                except MemoryError as e:
                    ocr_errors_total.labels(
                        provider=self.name, code=ErrorCode.RESOURCE_EXHAUSTED.value, stage="load"
                    ).inc()
                    logger.error(f"Resource exhausted during model load: {e}")
                    self._model = "stub"
                except Exception as e:
                    ocr_errors_total.labels(
                        provider=self.name, code=ErrorCode.MODEL_LOAD_ERROR.value, stage="load"
                    ).inc()
                    logger.error(f"Failed to load model: {e}")
                    self._model = "stub"

    async def extract(self, image_bytes: bytes, trace_id: Optional[str] = None) -> OcrResult:
        start = time.time()
        timer = StageTimer()
        timer.start("preprocess")
        processed_bytes, _ = enhance_image_for_ocr(image_bytes, max_res=2048)
        if not self._model:
            await self._lazy_load()
        timer.end("preprocess")
        timer.start("infer")

        async def _infer():
            # model generation or stub fallback
            if self._model == "stub":
                return '```json\n{"dimensions": [{"type": "diameter", "value": 20, "tolerance": 0.02}], "symbols": []}\n```'
            prompt = deepseek_ocr_json_prompt()
            try:
                inputs = self._tokenizer(prompt, return_tensors="pt")
                outputs = self._model.generate(**inputs, max_new_tokens=128)
                return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            except MemoryError as e:
                ocr_errors_total.labels(
                    provider=self.name, code=ErrorCode.RESOURCE_EXHAUSTED.value, stage="infer"
                ).inc()
                logger.error(f"Memory exhausted during inference: {e}")
                return ""
            except Exception as e:
                ocr_errors_total.labels(
                    provider=self.name, code=ErrorCode.INTERNAL_ERROR.value, stage="infer"
                ).inc()
                logger.error(f"Inference failed: {e}")
                return ""

        try:
            raw_output = await asyncio.wait_for(_infer(), timeout=self.timeout_ms / 1000.0)
        except (asyncio.TimeoutError, TimeoutError):
            ocr_errors_total.labels(
                provider=self.name, code=ErrorCode.PROVIDER_TIMEOUT.value, stage="infer"
            ).inc()
            logger.warning(f"Provider timeout after {self.timeout_ms}ms")
            raw_output = ""
        except Exception as e:
            # Defensive: record unexpected outer errors as internal and continue gracefully.
            ocr_errors_total.labels(
                provider=self.name, code=ErrorCode.INTERNAL_ERROR.value, stage="infer"
            ).inc()
            logger.error(f"Unexpected error during inference: {e}")
            raw_output = ""

        if self._model == "stub":
            raw_output = '```json\n{"dimensions": [{"type": "diameter", "value": 20, "tolerance": 0.02}], "symbols": []}\n```'
        timer.end("infer")
        timer.start("parse")
        parsed = self._parser.parse(raw_output)
        text = raw_output

        dimensions = []
        symbols = []
        title_block_data = {}
        title_block_confidence = {}
        if parsed.success and parsed.data:
            title_block_data = parsed.data.get("title_block", {}) or {}
            candidate_confidence = parsed.data.get("title_block_confidence", {}) or {}
            if isinstance(candidate_confidence, dict):
                for key, value in candidate_confidence.items():
                    if isinstance(value, (int, float)):
                        title_block_confidence[key] = float(value)
            for d in parsed.data.get("dimensions", []):
                try:
                    dimensions.append(
                        DimensionInfo(
                            type=DimensionType(d.get("type")),
                            value=float(d.get("value")),
                            tolerance=d.get("tolerance"),
                        )
                    )
                except Exception:
                    ocr_errors_total.labels(
                        provider=self.name, code=ErrorCode.PARSE_FAILED.value, stage="parse"
                    ).inc()
                    continue
            for s in parsed.data.get("symbols", []):
                try:
                    symbols.append(
                        SymbolInfo(type=SymbolType(s.get("type")), value=str(s.get("value")))
                    )
                except Exception:
                    ocr_errors_total.labels(
                        provider=self.name, code=ErrorCode.PARSE_FAILED.value, stage="parse"
                    ).inc()
                    continue
        extraction_mode = "json_only"
        if text:
            regex_dims, regex_syms = parse_dimensions_and_symbols(text)
            existing_raws = {d.raw for d in dimensions if d.raw}
            for rd in regex_dims:
                if rd.raw not in existing_raws:
                    dimensions.append(rd)
            existing_symbol_raws = {s.raw for s in symbols if s.raw}
            for rs in regex_syms:
                if rs.raw not in existing_symbol_raws:
                    symbols.append(rs)
            if parsed.success and parsed.data and (regex_dims or regex_syms):
                extraction_mode = "json+regex_merge"
            elif not parsed.success:
                extraction_mode = "regex_only"
        timer.end("parse")
        if text:
            for field, value in parse_title_block(text).items():
                title_block_data.setdefault(field, value)
        title_block = TitleBlock(**title_block_data)
        # Optional alignment using PaddleOCR for bboxes
        timer.start("align")
        if self._align_with_paddle and self._paddle is None and PaddleOCR:
            try:
                self._paddle = PaddleOCR(lang="ch", use_angle_cls=True, use_gpu=False)
            except Exception:
                self._paddle = None
        if self._align_with_paddle and self._paddle is not None:
            try:
                ocr_result = self._paddle.ocr(processed_bytes, cls=True)
                ocr_lines = []
                for line in ocr_result:
                    for box, (txt, score) in line:
                        ocr_lines.append(
                            {
                                "text": txt,
                                "bbox": polygon_to_bbox(box),
                                "score": float(score) if score is not None else None,
                            }
                        )
                if ocr_lines:
                    assign_bboxes(dimensions, symbols, ocr_lines)
            except Exception:
                pass
        timer.end("align")
        timer.start("postprocess")
        stage_latencies = timer.durations_ms()
        for stage, ms in stage_latencies.items():
            ocr_stage_duration_seconds.labels(provider=self.name, stage=stage).observe(ms / 1000.0)
        timer.end("postprocess")
        return OcrResult(
            text=text,
            dimensions=dimensions,
            symbols=symbols,
            title_block=title_block,
            title_block_confidence=title_block_confidence,
            confidence=0.9,
            fallback_level=parsed.fallback_level.value,
            processing_time_ms=int((time.time() - start) * 1000),
            extraction_mode=extraction_mode,
            trace_id=trace_id,
            stages_latency_ms=stage_latencies,
        )

        # Structured provider log (best-effort)
        try:
            import logging

            logging.getLogger(__name__).info(
                "ocr.provider.extract",
                extra={
                    "provider": self.name,
                    "latency_ms": int((time.time() - start) * 1000),
                    "extraction_mode": extraction_mode,
                    "dimensions_count": len(dimensions),
                    "symbols_count": len(symbols),
                    "trace_id": trace_id,
                    "stage": "provider",
                },
            )
        except Exception:
            pass

    async def health_check(self) -> bool:
        return self._model is not None
