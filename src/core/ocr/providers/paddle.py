"""PaddleOCR provider scaffold.

Minimal CPU-only implementation returning dummy structured result until PaddleOCR integrated.
Real integration will import PaddleOCR and run detection/recognition.
"""

from __future__ import annotations

import logging
import time

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None  # type: ignore

from src.core.errors import ErrorCode
from src.utils.metrics import ocr_errors_total, ocr_stage_duration_seconds

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
from ..parsing.title_block_parser import parse_title_block, parse_title_block_with_confidence
from ..preprocessing.image_enhancer import enhance_image_for_ocr
from ..stage_timer import StageTimer


class PaddleOcrProvider(OcrClient):
    name = "paddle"

    def __init__(self, enable_preprocess: bool = True, max_res: int = 2048, **paddle_kwargs):
        self._initialized = False
        self._enable_preprocess = enable_preprocess
        self._max_res = max_res
        self._paddle_kwargs = paddle_kwargs
        # Underlying OCR client; may be set by warmup() or injected by tests
        self._ocr = None

    async def warmup(self) -> None:
        if PaddleOCR and not self._initialized:
            # Allow external override via kwargs for fidelity tuning
            kwargs = {"lang": "ch", "use_angle_cls": True, "use_gpu": False}
            kwargs.update(self._paddle_kwargs or {})
            try:
                self._ocr = PaddleOCR(**kwargs)
            except MemoryError as e:
                ocr_errors_total.labels(
                    provider=self.name, code=ErrorCode.RESOURCE_EXHAUSTED.value, stage="init"
                ).inc()
                logging.error(f"Resource exhausted during PaddleOCR init: {e}")
                raise
            except Exception as e:
                # fallback to default minimal init
                logging.warning(f"Primary PaddleOCR init failed, falling back to defaults: {e}")
                try:
                    self._ocr = PaddleOCR(lang="ch", use_angle_cls=True, use_gpu=False)
                except Exception as fallback_error:
                    ocr_errors_total.labels(
                        provider=self.name, code=ErrorCode.MODEL_LOAD_ERROR.value, stage="init"
                    ).inc()
                    logging.error(f"Failed to initialize PaddleOCR: {fallback_error}")
                    raise
        self._initialized = True

    async def extract(self, image_bytes: bytes, trace_id: str | None = None) -> OcrResult:
        start = time.time()
        timer = StageTimer()
        timer.start("preprocess")
        # Basic CPU preprocessing to improve text clarity
        if self._enable_preprocess:
            processed_bytes, _ = enhance_image_for_ocr(image_bytes, max_res=self._max_res)
        else:
            processed_bytes = image_bytes
        timer.end("preprocess")
        timer.start("infer")
        text = ""
        dimensions = []
        symbols = []
        title_block_data = {}
        ocr_lines = []
        # Prefer an already-initialized client (including test mocks). If absent, try warmup when PaddleOCR is available.
        if self._ocr is None and PaddleOCR and not self._initialized:
            await self.warmup()

        if self._ocr is not None:
            try:
                ocr_result = self._ocr.ocr(processed_bytes, cls=True)
                texts = []
                # Support outputs that are either [ [ [points], (text,score) ], ... ] or flattened forms
                for line in ocr_result:
                    for item in line:
                        try:
                            box, (txt, score) = item
                        except ValueError:
                            # some versions may return dict-like
                            if isinstance(item, dict):
                                txt = item.get("text", "")
                                box = item.get("bbox", [])
                                score = item.get("score", 1.0)
                            else:
                                continue
                        except Exception as parse_error:
                            ocr_errors_total.labels(
                                provider=self.name, code=ErrorCode.PARSE_FAILED.value, stage="parse"
                            ).inc()
                            logging.warning(f"Failed to parse OCR item: {parse_error}")
                            continue
                        if txt:
                            texts.append(str(txt))
                            ocr_lines.append(
                                {
                                    "text": str(txt),
                                    "bbox": polygon_to_bbox(box),
                                    "score": float(score) if score is not None else None,
                                }
                            )
                text = " ".join(texts)
            except MemoryError as e:
                ocr_errors_total.labels(
                    provider=self.name, code=ErrorCode.RESOURCE_EXHAUSTED.value, stage="infer"
                ).inc()
                logging.error(f"Memory exhausted during OCR: {e}")
                text = ""
            except TimeoutError as e:
                ocr_errors_total.labels(
                    provider=self.name, code=ErrorCode.PROVIDER_TIMEOUT.value, stage="infer"
                ).inc()
                logging.error(f"OCR provider timeout: {e}")
                text = ""
            except Exception as e:
                ocr_errors_total.labels(
                    provider=self.name, code=ErrorCode.INTERNAL_ERROR.value, stage="infer"
                ).inc()
                logging.error(f"Paddle OCR failed: {e}")
                # Paddle path failed — keep empty to fall back to regex parsing below
                text = ""
        else:
            text = "Φ20±0.02 R5 M10×1.5 Ra3.2 Drawing No: PAD-001"
            dimensions = [
                DimensionInfo(type=DimensionType.diameter, value=20.0, tolerance=0.02),
                DimensionInfo(type=DimensionType.radius, value=5.0),
                DimensionInfo(type=DimensionType.thread, value=10.0, pitch=1.5),
            ]
            symbols = [SymbolInfo(type=SymbolType.surface_roughness, value="3.2")]
        timer.end("infer")
        extraction_mode = "provider_native"
        timer.start("parse")
        if text and not dimensions:
            parsed_dims, parsed_syms = parse_dimensions_and_symbols(text)
            dimensions = parsed_dims
            symbols = parsed_syms
            extraction_mode = "regex_only"
        # map bboxes if available
        if ocr_lines and (dimensions or symbols):
            assign_bboxes(dimensions, symbols, ocr_lines)
        if ocr_lines:
            title_block_data, _ = parse_title_block_with_confidence(ocr_lines)
        if text:
            for field, value in parse_title_block(text).items():
                title_block_data.setdefault(field, value)
        timer.end("parse")
        timer.start("postprocess")
        stage_latencies = timer.durations_ms()
        for stage, ms in stage_latencies.items():
            ocr_stage_duration_seconds.labels(provider=self.name, stage=stage).observe(ms / 1000.0)
        timer.end("postprocess")
        return OcrResult(
            text=text,
            dimensions=dimensions,
            symbols=symbols,
            title_block=TitleBlock(**title_block_data),
            confidence=0.82,
            processing_time_ms=int((time.time() - start) * 1000),
            extraction_mode=extraction_mode,
            trace_id=trace_id,
            stages_latency_ms=stage_latencies,
        )

        # Structured provider log (best-effort)
        try:
            logger = logging.getLogger(__name__)
            logger.info(
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
        return self._initialized
