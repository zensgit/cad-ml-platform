"""OCR Manager: routing + fallback + cache key composition (scaffold).

Simplified initial implementation; real provider logic added incrementally.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Dict, Optional

from src.core.resilience.adaptive_decorator import adaptive_rate_limit
from src.utils.cache import get_cache, set_cache
from src.utils.circuit_breaker import CircuitBreaker, CircuitConfig
from src.utils.metrics import (
    ocr_completeness_ratio,
    ocr_confidence_distribution,
    ocr_confidence_ema,
    ocr_confidence_fallback_threshold,
    ocr_errors_total,
    ocr_fallback_triggered,
    ocr_image_size_bytes,
    ocr_item_confidence_distribution,
    ocr_processing_duration_seconds,
    ocr_requests_total,
    update_ocr_error_ema,
)
from src.utils.metrics_helpers import safe_inc, safe_observe, safe_set
from src.utils.rate_limiter import RateLimiter

from .base import DimensionType, OcrClient, OcrResult, SymbolType
from .calibration import MultiEvidenceCalibrator
from .config import DATASET_VERSION, PROMPT_VERSION
from .exceptions import OcrError

# Versions centralized in config.


class OcrManager:
    def __init__(
        self, providers: Optional[Dict[str, OcrClient]] = None, confidence_fallback: float = 0.85
    ):
        self.providers = providers or {}
        self.confidence_fallback = confidence_fallback
        # dynamic threshold state
        try:
            from .rolling_stats import RollingStats

            self._conf_stats = RollingStats(alpha=0.2)
        except Exception:
            self._conf_stats = None
        self._calibrator = MultiEvidenceCalibrator()

    def register_provider(self, name: str, client: OcrClient):
        self.providers[name] = client
        # Mark provider model loaded
        try:
            from src.utils.metrics import ocr_model_loaded

            safe_set(ocr_model_loaded, 1, provider=name)
        except Exception:
            pass

    def _crop_cfg_hash(self, crop_cfg: Optional[Dict]) -> str:
        cfg = crop_cfg or {}
        cfg_str = json.dumps(cfg, sort_keys=True)
        return hashlib.sha256(cfg_str.encode()).hexdigest()[:8]

    def build_cache_key(
        self,
        image_bytes: bytes,
        provider: str,
        crop_cfg: Optional[Dict] = None,
        include_dataset: bool = False,
    ) -> str:
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        parts = [
            "ocr",
            image_hash,
            provider,
            PROMPT_VERSION,
            self._crop_cfg_hash(crop_cfg),
        ]
        if include_dataset:
            parts.append(DATASET_VERSION)
        return ":".join(parts)

    @adaptive_rate_limit(service="ocr", endpoint="extract")
    async def extract(
        self,
        image_bytes: bytes,
        strategy: str = "auto",
        crop_cfg: Optional[Dict] = None,
        trace_id: str | None = None,
    ) -> OcrResult:
        start = time.time()
        provider_name = self._select_provider(strategy)
        provider = self.providers.get(provider_name)
        if not provider:
            ocr_errors_total.labels(
                provider=provider_name, code="provider_down", stage="infer"
            ).inc()
            from src.core.errors import ErrorCode

            raise OcrError(
                ErrorCode.PROVIDER_DOWN,
                f"Provider '{provider_name}' not available",
                provider=provider_name,
                stage="infer",
            )
        # Rate limiting (per provider)
        if not hasattr(self, "_rate_limiters"):
            self._rate_limiters = {}
        rl = self._rate_limiters.setdefault(
            provider_name, RateLimiter(key=provider_name, qps=10.0, burst=10)
        )
        allowed = await rl.allow()
        if not allowed:
            from src.utils.metrics import ocr_rate_limited_total

            ocr_rate_limited_total.inc()
            ocr_errors_total.labels(
                provider=provider_name, code="rate_limit", stage="preprocess"
            ).inc()
            from src.core.errors import ErrorCode

            raise OcrError(
                ErrorCode.RATE_LIMIT, "Rate limited", provider=provider_name, stage="preprocess"
            )

        # Circuit breaker per provider
        if not hasattr(self, "_circuits"):
            self._circuits = {}
        cb = self._circuits.setdefault(
            provider_name, CircuitBreaker(provider_name, CircuitConfig())
        )
        if not await cb.should_allow():
            ocr_errors_total.labels(
                provider=provider_name, code="circuit_open", stage="infer"
            ).inc()
            from src.core.errors import ErrorCode

            raise OcrError(
                ErrorCode.CIRCUIT_OPEN, "Circuit open", provider=provider_name, stage="infer"
            )
        # Cache check (skip for deepseek for now to measure quality, enable later)
        cache_key = self.build_cache_key(image_bytes, provider_name, crop_cfg)
        cached = await get_cache(cache_key)
        if cached:
            safe_inc(ocr_requests_total, provider=provider_name, status="cache_hit")
            return OcrResult(**cached)

        safe_inc(ocr_requests_total, provider=provider_name, status="start")

        # Observe input size for OCR
        try:
            ocr_image_size_bytes.observe(len(image_bytes))
        except Exception:
            pass

        # Basic extraction (future: shared lock registry)
        # prevent cache stampede per image hash
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        lock_key = f"ocr_lock:{image_hash}:{provider_name}"
        # simple in-process lock registry
        if not hasattr(self, "_locks"):
            self._locks = {}
        lock = self._locks.setdefault(lock_key, asyncio.Lock())
        try:
            async with lock:
                result = await provider.extract(image_bytes, trace_id=trace_id)
            await cb.on_success()
            # success path contributes negatively to error EMA
            update_ocr_error_ema(False)
        except Exception:
            update_ocr_error_ema(True)
            await cb.on_error()
            raise
        result.provider = provider_name
        result.image_hash = image_hash
        result.trace_id = trace_id
        # completeness & calibrated confidence
        result.completeness = self._compute_completeness(result)
        item_mean = None
        if result.dimensions or result.symbols:
            vals = [d.confidence for d in result.dimensions if d.confidence is not None] + [
                s.confidence for s in result.symbols if s.confidence is not None
            ]
            if vals:
                item_mean = sum(vals) / len(vals)
        # simplistic recent fallback ratio & parse error rate placeholders (future: track counters)
        fallback_recent = 0.0  # could be derived from metrics over window
        parse_error_rate = 0.0
        result.calibrated_confidence = self._calibrator.calibrate(
            result.confidence,
            result.completeness,
            item_mean=item_mean,
            fallback_recent=fallback_recent,
            parse_error_rate=parse_error_rate,
        )
        if result.confidence is not None:
            safe_observe(ocr_confidence_distribution, result.confidence, provider=provider_name)
        if result.completeness is not None:
            safe_observe(ocr_completeness_ratio, result.completeness, provider=provider_name)
        # Update EMA and dynamic threshold (bounded)
        calib = (
            result.calibrated_confidence
            if result.calibrated_confidence is not None
            else result.confidence
        )
        if calib is not None and self._conf_stats:
            ema = self._conf_stats.update(calib)
            safe_set(ocr_confidence_ema, ema)
            # adjust threshold slightly below EMA with floor/ceiling
            new_thr = max(0.6, min(0.95, ema - 0.05))
            self.confidence_fallback = new_thr
            safe_set(ocr_confidence_fallback_threshold, self.confidence_fallback)

        # Missing-field fallback gate
        if strategy == "auto" and self._missing_key_fields(result):
            deepseek = self.providers.get("deepseek_hf")
            if provider_name != "deepseek_hf" and deepseek:
                safe_inc(ocr_fallback_triggered, reason="missing_fields")
                ds_result = await deepseek.extract(image_bytes, trace_id=trace_id)
                ds_result.provider = "deepseek_hf"
                ds_result.image_hash = image_hash
                ds_result.fallback_level = "missing_fields"
                ds_result.processing_time_ms = int((time.time() - start) * 1000)
                ds_result.completeness = self._compute_completeness(ds_result)
                item_mean_ds = None
                if ds_result.dimensions or ds_result.symbols:
                    vals_ds = [
                        d.confidence for d in ds_result.dimensions if d.confidence is not None
                    ] + [s.confidence for s in ds_result.symbols if s.confidence is not None]
                    if vals_ds:
                        item_mean_ds = sum(vals_ds) / len(vals_ds)
                ds_result.calibrated_confidence = self._calibrator.calibrate(
                    ds_result.confidence, ds_result.completeness, item_mean=item_mean_ds
                )
                if ds_result.confidence is not None:
                    ocr_confidence_distribution.labels(provider=ds_result.provider).observe(
                        ds_result.confidence
                    )
                if ds_result.completeness is not None:
                    ocr_completeness_ratio.labels(provider=ds_result.provider).observe(
                        ds_result.completeness
                    )
                ocr_processing_duration_seconds.labels(provider=ds_result.provider).observe(
                    ds_result.processing_time_ms / 1000.0
                )
                await set_cache(cache_key, ds_result.model_dump())
                ocr_requests_total.labels(provider=ds_result.provider, status="success").inc()
                return ds_result

        # Confidence fallback gate
        # use calibrated confidence if available
        conf_cmp = (
            result.calibrated_confidence
            if result.calibrated_confidence is not None
            else result.confidence
        )
        if conf_cmp is not None and conf_cmp < self.confidence_fallback and strategy == "auto":
            deepseek = self.providers.get("deepseek_hf")
            if provider_name != "deepseek_hf" and deepseek:
                ocr_fallback_triggered.labels(reason="low_confidence").inc()
                ds_result = await deepseek.extract(image_bytes, trace_id=trace_id)
                ds_result.provider = "deepseek_hf"
                ds_result.image_hash = image_hash
                ds_result.fallback_level = "confidence_fallback"
                ds_result.processing_time_ms = int((time.time() - start) * 1000)
                ds_result.completeness = self._compute_completeness(ds_result)
                item_mean_ds = None
                if ds_result.dimensions or ds_result.symbols:
                    vals_ds = [
                        d.confidence for d in ds_result.dimensions if d.confidence is not None
                    ] + [s.confidence for s in ds_result.symbols if s.confidence is not None]
                    if vals_ds:
                        item_mean_ds = sum(vals_ds) / len(vals_ds)
                ds_result.calibrated_confidence = self._calibrator.calibrate(
                    ds_result.confidence, ds_result.completeness, item_mean=item_mean_ds
                )
                if ds_result.confidence is not None:
                    ocr_confidence_distribution.labels(provider=ds_result.provider).observe(
                        ds_result.confidence
                    )
                if ds_result.completeness is not None:
                    ocr_completeness_ratio.labels(provider=ds_result.provider).observe(
                        ds_result.completeness
                    )
                ocr_processing_duration_seconds.labels(provider=ds_result.provider).observe(
                    ds_result.processing_time_ms / 1000.0
                )
                # pydantic v2
                await set_cache(cache_key, ds_result.model_dump())
                ocr_requests_total.labels(provider=ds_result.provider, status="success").inc()
                return ds_result

        result.processing_time_ms = int((time.time() - start) * 1000)
        try:
            import logging

            logging.getLogger(__name__).info(
                "ocr.manager.extract",
                extra={
                    "provider": result.provider,
                    "latency_ms": result.processing_time_ms,
                    "fallback_level": result.fallback_level,
                    "image_hash": result.image_hash,
                    "stage": "manager",
                    "trace_id": result.trace_id,
                    "extraction_mode": result.extraction_mode,
                    "completeness": result.completeness,
                    "calibrated_confidence": result.calibrated_confidence or result.confidence,
                    "dimensions_count": len(result.dimensions),
                    "symbols_count": len(result.symbols),
                    "stages_latency_ms": result.stages_latency_ms,
                },
            )
        except Exception:
            pass
        ocr_processing_duration_seconds.labels(provider=result.provider).observe(
            result.processing_time_ms / 1000.0
        )
        # Per-item confidence metrics
        for d in result.dimensions:
            if d.confidence is not None:
                ocr_item_confidence_distribution.labels(
                    provider=result.provider, item_type="dimension"
                ).observe(d.confidence)
        for s in result.symbols:
            if s.confidence is not None:
                ocr_item_confidence_distribution.labels(
                    provider=result.provider, item_type="symbol"
                ).observe(s.confidence)
        await set_cache(cache_key, result.model_dump())
        ocr_requests_total.labels(provider=result.provider, status="success").inc()
        return result

    def _select_provider(self, strategy: str) -> str:
        """Select provider based on strategy.

        Rules:
        - If strategy explicitly names a provider and it's not registered, return the same
          name to trigger a PROVIDER_DOWN path upstream (no silent fallback).
        - If strategy is 'auto' (or empty), prefer 'paddle' then 'deepseek_hf'; otherwise
          fallback to any available provider, or 'unknown' if none.
        """
        if strategy and strategy != "auto":
            # Respect explicit provider; do not silently fallback
            return strategy
        # auto strategy preference order
        if "paddle" in self.providers:
            return "paddle"
        if "deepseek_hf" in self.providers:
            return "deepseek_hf"
        # fallback to any available or unknown
        return next(iter(self.providers.keys())) if self.providers else "unknown"

    def _compute_completeness(self, result: OcrResult) -> float:
        text = result.text or ""
        token_expect = 0
        if any(t in text for t in ["Φ", "⌀", "∅"]):
            token_expect += 1
        if "R" in text:
            token_expect += 1
        if "M" in text:
            token_expect += 1
        if "Ra" in text or "RA" in text:
            token_expect += 1
        found = 0
        dim_types = {d.type for d in result.dimensions}
        if DimensionType.diameter in dim_types:
            found += 1
        if DimensionType.radius in dim_types:
            found += 1
        if DimensionType.thread in dim_types:
            found += 1
        sym_types = {s.type for s in result.symbols}
        if SymbolType.surface_roughness in sym_types:
            found += 1
        if token_expect == 0:
            return 1.0
        return min(found / token_expect, 1.0)

    def _missing_key_fields(self, result: OcrResult) -> bool:
        text = result.text or ""
        if not text:
            return False
        hints = [
            ("Φ", DimensionType.diameter),
            ("⌀", DimensionType.diameter),
            ("∅", DimensionType.diameter),
            ("R", DimensionType.radius),
            ("M", DimensionType.thread),
            ("Ra", SymbolType.surface_roughness),
        ]
        dims = {d.type for d in result.dimensions}
        syms = {s.type for s in result.symbols}
        for token, dtype in hints:
            if token in text:
                if isinstance(dtype, DimensionType) and dtype not in dims:
                    return True
                if isinstance(dtype, SymbolType) and dtype not in syms:
                    return True
        return False
