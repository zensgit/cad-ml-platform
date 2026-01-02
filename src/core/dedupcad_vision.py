"""Client for dedupcad-vision (2D dedup/search service).

This module provides a small HTTP client wrapper used by API routes to call the
separate `dedupcad-vision` service (default http://localhost:58001).
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from src.core.vision.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    FailureType,
    get_circuit_breaker,
)
from src.utils.analysis_metrics import (
    dedupcad_vision_circuit_state,
    dedupcad_vision_errors_total,
    dedupcad_vision_request_duration_seconds,
    dedupcad_vision_requests_total,
    dedupcad_vision_retry_total,
)


@dataclass(frozen=True)
class DedupCadVisionConfig:
    base_url: str = "http://localhost:58001"
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "DedupCadVisionConfig":
        timeout_seconds = float(
            os.getenv(
                "DEDUPCAD_VISION_TIMEOUT_SECONDS",
                str(cls.timeout_seconds),
            )
        )
        return cls(
            base_url=os.getenv("DEDUPCAD_VISION_URL", cls.base_url).rstrip("/"),
            timeout_seconds=timeout_seconds,
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class DedupCadVisionResilienceConfig:
    retry_max_attempts: int = 2
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 5.0
    circuit_enabled: bool = True
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout_seconds: float = 30.0
    circuit_half_open_max_calls: int = 2
    circuit_success_threshold: int = 2
    circuit_slow_call_threshold_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> "DedupCadVisionResilienceConfig":
        retry_max_attempts = max(1, _env_int("DEDUPCAD_VISION_RETRY_MAX_ATTEMPTS", 2))
        retry_base_delay = max(
            0.0, _env_float("DEDUPCAD_VISION_RETRY_BASE_DELAY_SECONDS", 0.5)
        )
        retry_max_delay = max(
            retry_base_delay,
            _env_float("DEDUPCAD_VISION_RETRY_MAX_DELAY_SECONDS", 5.0),
        )
        circuit_failure_threshold = max(
            1, _env_int("DEDUPCAD_VISION_CIRCUIT_FAILURE_THRESHOLD", 5)
        )
        circuit_recovery_timeout = max(
            1.0, _env_float("DEDUPCAD_VISION_CIRCUIT_RECOVERY_TIMEOUT_SECONDS", 30.0)
        )
        circuit_half_open_max_calls = max(
            1, _env_int("DEDUPCAD_VISION_CIRCUIT_HALF_OPEN_MAX_CALLS", 2)
        )
        circuit_success_threshold = max(
            1, _env_int("DEDUPCAD_VISION_CIRCUIT_SUCCESS_THRESHOLD", 2)
        )
        circuit_slow_call_threshold = max(
            0.1, _env_float("DEDUPCAD_VISION_CIRCUIT_SLOW_CALL_THRESHOLD_SECONDS", 5.0)
        )
        return cls(
            retry_max_attempts=retry_max_attempts,
            retry_base_delay_seconds=retry_base_delay,
            retry_max_delay_seconds=retry_max_delay,
            circuit_enabled=_env_bool("DEDUPCAD_VISION_CIRCUIT_ENABLED", True),
            circuit_failure_threshold=circuit_failure_threshold,
            circuit_recovery_timeout_seconds=circuit_recovery_timeout,
            circuit_half_open_max_calls=circuit_half_open_max_calls,
            circuit_success_threshold=circuit_success_threshold,
            circuit_slow_call_threshold_seconds=circuit_slow_call_threshold,
        )


class DedupCadVisionCircuitOpen(RuntimeError):
    """Raised when the dedupcad-vision circuit is open."""


class DedupCadVisionClient:
    def __init__(
        self,
        config: Optional[DedupCadVisionConfig] = None,
        resilience: Optional[DedupCadVisionResilienceConfig] = None,
    ) -> None:
        self.config = config or DedupCadVisionConfig.from_env()
        self.resilience = resilience or DedupCadVisionResilienceConfig.from_env()

    def _get_circuit_breaker(self, endpoint: str) -> Optional[CircuitBreaker]:
        if not self.resilience.circuit_enabled:
            return None
        config = CircuitBreakerConfig(
            failure_threshold=self.resilience.circuit_failure_threshold,
            success_threshold=self.resilience.circuit_success_threshold,
            timeout_seconds=self.resilience.circuit_recovery_timeout_seconds,
            half_open_max_calls=self.resilience.circuit_half_open_max_calls,
            slow_call_threshold_seconds=self.resilience.circuit_slow_call_threshold_seconds,
        )
        return get_circuit_breaker(f"dedupcad_vision:{endpoint}", config)

    @staticmethod
    def _circuit_state_value(state: CircuitState) -> int:
        return {
            CircuitState.CLOSED: 0,
            CircuitState.OPEN: 1,
            CircuitState.HALF_OPEN: 2,
        }.get(state, 0)

    def _record_circuit_state(self, endpoint: str, breaker: Optional[CircuitBreaker]) -> None:
        if breaker is None:
            return
        dedupcad_vision_circuit_state.labels(endpoint=endpoint).set(
            self._circuit_state_value(breaker.state)
        )

    def _retry_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        base = self.resilience.retry_base_delay_seconds
        max_delay = self.resilience.retry_max_delay_seconds
        delay = min(max_delay, base * (2 ** (attempt - 1)))
        delay *= random.uniform(0.5, 1.0)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay

    @staticmethod
    def _is_retryable_status(code: int) -> bool:
        return code >= 500 or code in {408, 429}

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        endpoint: str,
        allow_retry: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        max_attempts = self.resilience.retry_max_attempts if allow_retry else 1
        max_attempts = max(1, int(max_attempts))
        breaker = self._get_circuit_breaker(endpoint)

        for attempt in range(1, max_attempts + 1):
            if breaker and not breaker.can_execute():
                dedupcad_vision_requests_total.labels(
                    endpoint=endpoint, status="circuit_open"
                ).inc()
                self._record_circuit_state(endpoint, breaker)
                raise DedupCadVisionCircuitOpen(
                    f"dedupcad-vision circuit open for {endpoint}"
                )

            start_time = time.monotonic()
            try:
                async with httpx.AsyncClient(
                    base_url=self.config.base_url,
                    timeout=httpx.Timeout(self.config.timeout_seconds),
                ) as client:
                    resp = await client.request(method, path, **kwargs)

                duration = time.monotonic() - start_time
                if resp.status_code >= 400:
                    error_label = f"http_{resp.status_code}"
                    dedupcad_vision_errors_total.labels(
                        endpoint=endpoint, error=error_label
                    ).inc()
                    dedupcad_vision_request_duration_seconds.labels(
                        endpoint=endpoint, status="error"
                    ).observe(duration)
                    if breaker and resp.status_code >= 500:
                        breaker.record_failure(
                            duration * 1000,
                            FailureType.EXCEPTION,
                            error_label,
                        )
                    elif breaker and resp.status_code == 408:
                        breaker.record_failure(
                            duration * 1000,
                            FailureType.TIMEOUT,
                            error_label,
                        )
                    elif breaker and resp.status_code == 429:
                        breaker.record_failure(
                            duration * 1000,
                            FailureType.REJECTION,
                            error_label,
                        )
                    self._record_circuit_state(endpoint, breaker)

                    retry_after = None
                    if resp.status_code == 429:
                        header = resp.headers.get("Retry-After")
                        if header:
                            try:
                                retry_after = float(header)
                            except ValueError:
                                retry_after = None

                    if self._is_retryable_status(resp.status_code) and attempt < max_attempts:
                        dedupcad_vision_retry_total.labels(
                            endpoint=endpoint, reason=error_label
                        ).inc()
                        await asyncio.sleep(self._retry_delay(attempt, retry_after=retry_after))
                        continue

                    dedupcad_vision_requests_total.labels(endpoint=endpoint, status="error").inc()
                    resp.raise_for_status()

                try:
                    payload = resp.json()
                except ValueError as exc:
                    if breaker:
                        breaker.record_failure(
                            duration * 1000, FailureType.EXCEPTION, "invalid_json"
                        )
                        self._record_circuit_state(endpoint, breaker)
                    dedupcad_vision_errors_total.labels(
                        endpoint=endpoint, error="invalid_json"
                    ).inc()
                    dedupcad_vision_request_duration_seconds.labels(
                        endpoint=endpoint, status="error"
                    ).observe(duration)
                    if allow_retry and attempt < max_attempts:
                        dedupcad_vision_retry_total.labels(
                            endpoint=endpoint, reason="invalid_json"
                        ).inc()
                        await asyncio.sleep(self._retry_delay(attempt))
                        continue
                    dedupcad_vision_requests_total.labels(endpoint=endpoint, status="error").inc()
                    raise

                if breaker:
                    breaker.record_success(duration * 1000)
                    self._record_circuit_state(endpoint, breaker)

                dedupcad_vision_requests_total.labels(endpoint=endpoint, status="success").inc()
                dedupcad_vision_request_duration_seconds.labels(
                    endpoint=endpoint, status="success"
                ).observe(duration)
                return payload

            except httpx.TimeoutException as exc:
                duration = time.monotonic() - start_time
                if breaker:
                    breaker.record_failure(duration * 1000, FailureType.TIMEOUT, str(exc))
                    self._record_circuit_state(endpoint, breaker)
                dedupcad_vision_errors_total.labels(endpoint=endpoint, error="timeout").inc()
                dedupcad_vision_request_duration_seconds.labels(
                    endpoint=endpoint, status="error"
                ).observe(duration)
                if allow_retry and attempt < max_attempts:
                    dedupcad_vision_retry_total.labels(endpoint=endpoint, reason="timeout").inc()
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                dedupcad_vision_requests_total.labels(endpoint=endpoint, status="error").inc()
                raise
            except httpx.RequestError as exc:
                duration = time.monotonic() - start_time
                if breaker:
                    breaker.record_failure(duration * 1000, FailureType.EXCEPTION, str(exc))
                    self._record_circuit_state(endpoint, breaker)
                dedupcad_vision_errors_total.labels(
                    endpoint=endpoint, error=type(exc).__name__
                ).inc()
                dedupcad_vision_request_duration_seconds.labels(
                    endpoint=endpoint, status="error"
                ).observe(duration)
                if allow_retry and attempt < max_attempts:
                    dedupcad_vision_retry_total.labels(
                        endpoint=endpoint, reason=type(exc).__name__
                    ).inc()
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                dedupcad_vision_requests_total.labels(endpoint=endpoint, status="error").inc()
                raise

    async def health(self) -> Dict[str, Any]:
        return await self._request_json(
            "GET",
            "/health",
            endpoint="health",
            allow_retry=True,
        )

    async def rebuild_indexes(self) -> Dict[str, Any]:
        """Trigger a full (re)build of vision-side L1/L2 indexes."""
        return await self._request_json(
            "POST",
            "/api/v2/index/rebuild",
            endpoint="rebuild_indexes",
            allow_retry=False,
        )

    async def search_2d(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        content_type: str,
        mode: str = "balanced",
        max_results: int = 50,
        compute_diff: bool = True,
        enable_ml: bool = False,
        enable_geometric: bool = False,
    ) -> Dict[str, Any]:
        files = {"file": (file_name, file_bytes, content_type)}
        data = {
            "mode": mode,
            "max_results": str(max_results),
            "compute_diff": "true" if compute_diff else "false",
            "enable_ml": "true" if enable_ml else "false",
            "enable_geometric": "true" if enable_geometric else "false",
        }

        return await self._request_json(
            "POST",
            "/api/v2/search",
            endpoint="search_2d",
            allow_retry=True,
            files=files,
            data=data,
        )

    async def index_add_2d(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        content_type: str,
        user_name: str,
        upload_to_s3: bool = True,
    ) -> Dict[str, Any]:
        files = {"file": (file_name, file_bytes, content_type)}
        params = {
            "user_name": user_name,
            "upload_to_s3": "true" if upload_to_s3 else "false",
        }

        return await self._request_json(
            "POST",
            "/api/index/add",
            endpoint="index_add_2d",
            allow_retry=False,
            files=files,
            params=params,
        )
