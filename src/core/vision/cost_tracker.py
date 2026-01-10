"""Cost tracking for vision API usage.

Provides:
- Per-provider cost tracking
- Token usage estimation
- Budget alerts and limits
- Usage reporting
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .base import VisionDescription, VisionProvider, VisionProviderError

logger = logging.getLogger(__name__)


class CostUnit(Enum):
    """Units for cost calculation."""

    USD = "usd"
    TOKENS = "tokens"
    REQUESTS = "requests"


@dataclass
class PricingTier:
    """Pricing information for a provider."""

    provider: str
    input_cost_per_1k_tokens: float  # USD
    output_cost_per_1k_tokens: float  # USD
    image_cost_per_image: float  # USD for image input
    currency: str = "USD"
    effective_date: datetime = field(default_factory=datetime.now)

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        images: int = 1,
    ) -> float:
        """
        Calculate total cost for a request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            images: Number of images processed

        Returns:
            Total cost in USD
        """
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k_tokens
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k_tokens
        image_cost = images * self.image_cost_per_image
        return input_cost + output_cost + image_cost


# Default pricing (approximate as of 2024)
DEFAULT_PRICING: Dict[str, PricingTier] = {
    "openai": PricingTier(
        provider="openai",
        input_cost_per_1k_tokens=0.005,  # GPT-4o input
        output_cost_per_1k_tokens=0.015,  # GPT-4o output
        image_cost_per_image=0.00765,  # ~510 tokens at low detail
    ),
    "anthropic": PricingTier(
        provider="anthropic",
        input_cost_per_1k_tokens=0.003,  # Claude Sonnet input
        output_cost_per_1k_tokens=0.015,  # Claude Sonnet output
        image_cost_per_image=0.0048,  # Approximate
    ),
    "deepseek": PricingTier(
        provider="deepseek",
        input_cost_per_1k_tokens=0.00014,  # DeepSeek input
        output_cost_per_1k_tokens=0.00028,  # DeepSeek output
        image_cost_per_image=0.001,  # Approximate
    ),
    "deepseek_stub": PricingTier(
        provider="deepseek_stub",
        input_cost_per_1k_tokens=0.0,
        output_cost_per_1k_tokens=0.0,
        image_cost_per_image=0.0,
    ),
}


@dataclass
class UsageRecord:
    """Record of a single API usage."""

    provider: str
    timestamp: datetime
    input_tokens: int
    output_tokens: int
    images: int
    cost_usd: float
    success: bool
    latency_ms: float
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageSummary:
    """Summary of usage over a period."""

    provider: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_images: int
    total_cost_usd: float
    average_latency_ms: float

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def cost_per_request(self) -> float:
        """Calculate average cost per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_cost_usd / self.total_requests


@dataclass
class BudgetConfig:
    """Budget configuration for cost control."""

    daily_limit_usd: float = 10.0
    monthly_limit_usd: float = 100.0
    per_request_limit_usd: float = 1.0
    alert_threshold_percent: float = 80.0  # Alert at 80% of limit
    hard_limit: bool = False  # If True, reject requests over limit


BudgetAlertCallback = Callable[[str, float, float], None]


class CostTracker:
    """
    Track and manage API costs across providers.

    Features:
    - Real-time cost tracking
    - Budget limits and alerts
    - Usage history and reporting
    - Per-provider statistics
    """

    def __init__(
        self,
        pricing: Optional[Dict[str, PricingTier]] = None,
        budget_config: Optional[BudgetConfig] = None,
        alert_callback: Optional[BudgetAlertCallback] = None,
    ):
        """
        Initialize cost tracker.

        Args:
            pricing: Custom pricing tiers per provider
            budget_config: Budget limits configuration
            alert_callback: Callback for budget alerts
        """
        self._pricing = pricing or dict(DEFAULT_PRICING)
        self._budget = budget_config or BudgetConfig()
        self._alert_callback = alert_callback

        self._usage_history: List[UsageRecord] = []
        self._daily_costs: Dict[str, float] = {}  # date -> cost
        self._monthly_costs: Dict[str, float] = {}  # month -> cost
        self._lock = asyncio.Lock()

    async def record_usage(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        images: int = 1,
        success: bool = True,
        latency_ms: float = 0.0,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """
        Record API usage.

        Args:
            provider: Provider name
            input_tokens: Input token count
            output_tokens: Output token count
            images: Number of images
            success: Whether request succeeded
            latency_ms: Request latency
            request_id: Optional request ID
            metadata: Additional metadata

        Returns:
            UsageRecord for this request
        """
        async with self._lock:
            # Get pricing
            pricing = self._pricing.get(provider)
            if pricing:
                cost = pricing.calculate_cost(input_tokens, output_tokens, images)
            else:
                cost = 0.0
                logger.warning(f"No pricing found for provider: {provider}")

            # Create record
            now = datetime.now()
            record = UsageRecord(
                provider=provider,
                timestamp=now,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                images=images,
                cost_usd=cost,
                success=success,
                latency_ms=latency_ms,
                request_id=request_id,
                metadata=metadata or {},
            )

            # Update history
            self._usage_history.append(record)

            # Update daily/monthly totals
            date_key = now.strftime("%Y-%m-%d")
            month_key = now.strftime("%Y-%m")
            self._daily_costs[date_key] = self._daily_costs.get(date_key, 0.0) + cost
            self._monthly_costs[month_key] = self._monthly_costs.get(month_key, 0.0) + cost

            # Check budget alerts
            await self._check_budget_alerts(date_key, month_key)

            return record

    async def _check_budget_alerts(self, date_key: str, month_key: str) -> None:
        """Check and trigger budget alerts."""
        daily_cost = self._daily_costs.get(date_key, 0.0)
        monthly_cost = self._monthly_costs.get(month_key, 0.0)

        # Daily alert
        daily_percent = (daily_cost / self._budget.daily_limit_usd) * 100
        if daily_percent >= self._budget.alert_threshold_percent:
            if self._alert_callback:
                self._alert_callback(
                    "daily",
                    daily_cost,
                    self._budget.daily_limit_usd,
                )
            logger.warning(
                f"Daily budget alert: ${daily_cost:.2f} / "
                f"${self._budget.daily_limit_usd:.2f} "
                f"({daily_percent:.0f}%)"
            )

        # Monthly alert
        monthly_percent = (monthly_cost / self._budget.monthly_limit_usd) * 100
        if monthly_percent >= self._budget.alert_threshold_percent:
            if self._alert_callback:
                self._alert_callback(
                    "monthly",
                    monthly_cost,
                    self._budget.monthly_limit_usd,
                )
            logger.warning(
                f"Monthly budget alert: ${monthly_cost:.2f} / "
                f"${self._budget.monthly_limit_usd:.2f} "
                f"({monthly_percent:.0f}%)"
            )

    async def check_budget(self, estimated_cost: float = 0.0) -> bool:
        """
        Check if request is within budget.

        Args:
            estimated_cost: Estimated cost of upcoming request

        Returns:
            True if within budget, False otherwise
        """
        if not self._budget.hard_limit:
            return True

        async with self._lock:
            now = datetime.now()
            date_key = now.strftime("%Y-%m-%d")
            month_key = now.strftime("%Y-%m")

            daily_cost = self._daily_costs.get(date_key, 0.0) + estimated_cost
            monthly_cost = self._monthly_costs.get(month_key, 0.0) + estimated_cost

            # Check per-request limit
            if estimated_cost > self._budget.per_request_limit_usd:
                return False

            # Check daily limit
            if daily_cost > self._budget.daily_limit_usd:
                return False

            # Check monthly limit
            if monthly_cost > self._budget.monthly_limit_usd:
                return False

            return True

    def get_usage_summary(
        self,
        provider: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> UsageSummary:
        """
        Get usage summary for a period.

        Args:
            provider: Filter by provider (None for all)
            start_date: Start of period
            end_date: End of period

        Returns:
            UsageSummary for the period
        """
        end = end_date or datetime.now()
        start = start_date or (end - timedelta(days=30))

        # Filter records
        records = [
            r
            for r in self._usage_history
            if start <= r.timestamp <= end and (provider is None or r.provider == provider)
        ]

        if not records:
            return UsageSummary(
                provider=provider or "all",
                period_start=start,
                period_end=end,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_images=0,
                total_cost_usd=0.0,
                average_latency_ms=0.0,
            )

        # Calculate summary
        total_requests = len(records)
        successful = sum(1 for r in records if r.success)
        failed = total_requests - successful
        input_tokens = sum(r.input_tokens for r in records)
        output_tokens = sum(r.output_tokens for r in records)
        images = sum(r.images for r in records)
        cost = sum(r.cost_usd for r in records)
        avg_latency = sum(r.latency_ms for r in records) / total_requests

        return UsageSummary(
            provider=provider or "all",
            period_start=start,
            period_end=end,
            total_requests=total_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_images=images,
            total_cost_usd=cost,
            average_latency_ms=avg_latency,
        )

    def get_daily_cost(self, date: Optional[datetime] = None) -> float:
        """Get cost for a specific day."""
        date = date or datetime.now()
        date_key = date.strftime("%Y-%m-%d")
        return self._daily_costs.get(date_key, 0.0)

    def get_monthly_cost(self, date: Optional[datetime] = None) -> float:
        """Get cost for a specific month."""
        date = date or datetime.now()
        month_key = date.strftime("%Y-%m")
        return self._monthly_costs.get(month_key, 0.0)

    def get_total_cost(self) -> float:
        """Get total cost across all time."""
        return sum(r.cost_usd for r in self._usage_history)

    def get_provider_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by provider."""
        breakdown: Dict[str, float] = {}
        for record in self._usage_history:
            breakdown[record.provider] = breakdown.get(record.provider, 0.0) + record.cost_usd
        return breakdown

    def clear_history(self, before: Optional[datetime] = None) -> int:
        """
        Clear usage history.

        Args:
            before: Clear records before this date (None for all)

        Returns:
            Number of records cleared
        """
        if before is None:
            count = len(self._usage_history)
            self._usage_history.clear()
            self._daily_costs.clear()
            self._monthly_costs.clear()
            return count

        original_count = len(self._usage_history)
        self._usage_history = [r for r in self._usage_history if r.timestamp >= before]
        return original_count - len(self._usage_history)

    def update_pricing(self, provider: str, pricing: PricingTier) -> None:
        """Update pricing for a provider."""
        self._pricing[provider] = pricing
        logger.info(f"Updated pricing for {provider}")


class CostTrackedVisionProvider:
    """
    Wrapper that adds cost tracking to any VisionProvider.

    Tracks token usage and costs for each request.
    """

    def __init__(
        self,
        provider: VisionProvider,
        cost_tracker: CostTracker,
        estimate_tokens: bool = True,
    ):
        """
        Initialize cost-tracked provider.

        Args:
            provider: The underlying vision provider
            cost_tracker: CostTracker instance
            estimate_tokens: Whether to estimate token counts
        """
        self._provider = provider
        self._tracker = cost_tracker
        self._estimate_tokens = estimate_tokens

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """
        Analyze image with cost tracking.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description

        Returns:
            VisionDescription with analysis results
        """
        start_time = time.time()

        # Estimate input tokens (rough approximation)
        # Images are typically ~765 tokens at low detail, ~1105+ at high detail
        estimated_input_tokens = 1000  # Base prompt
        estimated_image_tokens = max(765, len(image_data) // 1000)

        # Check budget before request
        pricing = self._tracker._pricing.get(self._provider.provider_name)
        if pricing:
            estimated_cost = pricing.calculate_cost(
                estimated_input_tokens + estimated_image_tokens,
                500,  # Estimated output
                1,
            )
            if not await self._tracker.check_budget(estimated_cost):
                raise VisionProviderError(
                    self._provider.provider_name,
                    "Budget limit exceeded",
                )

        try:
            result = await self._provider.analyze_image(image_data, include_description)
            latency_ms = (time.time() - start_time) * 1000

            # Estimate output tokens from result
            output_text = result.summary + " ".join(result.details)
            estimated_output_tokens = len(output_text) // 4  # ~4 chars per token

            # Record usage
            await self._tracker.record_usage(
                provider=self._provider.provider_name,
                input_tokens=estimated_input_tokens + estimated_image_tokens,
                output_tokens=estimated_output_tokens,
                images=1,
                success=True,
                latency_ms=latency_ms,
                metadata={"confidence": result.confidence},
            )

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Record failed request
            await self._tracker.record_usage(
                provider=self._provider.provider_name,
                input_tokens=estimated_input_tokens + estimated_image_tokens,
                output_tokens=0,
                images=1,
                success=False,
                latency_ms=latency_ms,
                metadata={"error": str(e)},
            )
            raise

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name

    @property
    def cost_tracker(self) -> CostTracker:
        """Get the cost tracker."""
        return self._tracker


# Global cost tracker instance
_global_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """
    Get the global cost tracker instance.

    Returns:
        CostTracker singleton
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def create_cost_tracked_provider(
    provider: VisionProvider,
    budget_config: Optional[BudgetConfig] = None,
    alert_callback: Optional[BudgetAlertCallback] = None,
) -> CostTrackedVisionProvider:
    """
    Factory to create a cost-tracked provider wrapper.

    Args:
        provider: The underlying vision provider
        budget_config: Optional budget configuration
        alert_callback: Optional budget alert callback

    Returns:
        CostTrackedVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> tracked = create_cost_tracked_provider(
        ...     provider,
        ...     budget_config=BudgetConfig(daily_limit_usd=5.0),
        ... )
        >>> result = await tracked.analyze_image(image_bytes)
        >>> print(f"Daily cost: ${tracked.cost_tracker.get_daily_cost():.2f}")
    """
    tracker = CostTracker(
        budget_config=budget_config,
        alert_callback=alert_callback,
    )
    return CostTrackedVisionProvider(provider=provider, cost_tracker=tracker)
