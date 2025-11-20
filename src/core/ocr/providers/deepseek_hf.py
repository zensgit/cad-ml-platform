"""DeepSeek HF provider with lazy model load and fallback parsing.

Degrades gracefully to stub if transformers not available.
"""

from __future__ import annotations

import asyncio
import time
"""DeepSeek HF OCR provider stub."""
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # transformers not installed
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

from ..base import (
    OcrClient,
    OcrResult,
    DimensionInfo,
    SymbolInfo,
    TitleBlock,
    DimensionType,
    SymbolType,
)
from ..parsing.fallback_parser import FallbackParser
from ..utils.prompt_templates import deepseek_ocr_json_prompt
from ..parsing.dimension_parser import parse_dimensions_and_symbols
from ..parsing.bbox_mapper import assign_bboxes, polygon_to_bbox
from src.core.ocr.exceptions import OCR_ERRORS
from src.utils.metrics import ocr_errors_total, ocr_cold_start_seconds, ocr_stage_duration_seconds
from ..stage_timer import StageTimer
from ..preprocessing.image_enhancer import enhance_image_for_ocr

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None  # type: ignore


class DeepSeekHfProvider(OcrClient):
    name = "deepseek_hf"

    def __init__(self, timeout_ms: int = 30000, model_name: str = "deepseek-ocr-mini", align_with_paddle: bool = True):
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()
        self.timeout_ms = timeout_ms
        self.model_name = model_name
        self._parser = FallbackParser()
        self._align_with_paddle = align_with_paddle
        self._paddle = None

    async def warmup(self) -> None:
        if not self._model:
            await self._lazy_load()
        if self._align_with_paddle and PaddleOCR and self._paddle is None:
            try:
                self._paddle = PaddleOCR(lang='ch', use_angle_cls=True, use_gpu=False)
            except Exception:
                self._paddle = None

    async def _lazy_load(self):
        async with self._lock:
            if self._model is None and AutoModelForCausalLM and AutoTokenizer:
                try:
                    start = time.time()
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    ocr_cold_start_seconds.labels(provider=self.name).set(time.time() - start)
                except Exception:
                    ocr_errors_total.labels(provider=self.name, code=OCR_ERRORS["PROVIDER_DOWN"], stage="load").inc()
                    self._model = "stub"
            elif self._model is None:
                self._model = "stub"

    async def extract(self, image_bytes: bytes, trace_id: str | None = None) -> OcrResult:
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
                return "```json\n{\"dimensions\": [{\"type\": \"diameter\", \"value\": 20, \"tolerance\": 0.02}], \"symbols\": []}\n```"
            prompt = deepseek_ocr_json_prompt()
            try:
                inputs = self._tokenizer(prompt, return_tensors="pt")
                outputs = self._model.generate(**inputs, max_new_tokens=128)
                return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception:
                ocr_errors_total.labels(provider=self.name, code=OCR_ERRORS["PROVIDER_DOWN"], stage="infer").inc()
                return ""

        try:
            raw_output = await asyncio.wait_for(_infer(), timeout=self.timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            ocr_errors_total.labels(provider=self.name, code=OCR_ERRORS["TIMEOUT"], stage="infer").inc()
            raw_output = ""

        if self._model == "stub":
            raw_output = "```json\n{\"dimensions\": [{\"type\": \"diameter\", \"value\": 20, \"tolerance\": 0.02}], \"symbols\": []}\n```"
        timer.end("infer")
        timer.start("parse")
        parsed = self._parser.parse(raw_output)
        text = raw_output

        dimensions = []
        symbols = []
        title_block = TitleBlock()
        if parsed.success and parsed.data:
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
                    ocr_errors_total.labels(provider=self.name, code=OCR_ERRORS["PARSE_FAIL"], stage="parse").inc()
                    continue
            for s in parsed.data.get("symbols", []):
                try:
                    symbols.append(SymbolInfo(type=SymbolType(s.get("type")), value=str(s.get("value"))))
                except Exception:
                    ocr_errors_total.labels(provider=self.name, code=OCR_ERRORS["PARSE_FAIL"], stage="parse").inc()
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
        # Optional alignment using PaddleOCR for bboxes
        timer.start("align")
        if self._align_with_paddle and self._paddle is None and PaddleOCR:
            try:
                self._paddle = PaddleOCR(lang='ch', use_angle_cls=True, use_gpu=False)
            except Exception:
                self._paddle = None
        if self._align_with_paddle and self._paddle is not None:
            try:
                ocr_result = self._paddle.ocr(processed_bytes, cls=True)
                ocr_lines = []
                for line in ocr_result:
                    for box, (txt, score) in line:
                        ocr_lines.append({"text": txt, "bbox": polygon_to_bbox(box), "score": float(score) if score is not None else None})
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
