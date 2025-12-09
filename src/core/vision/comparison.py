"""Provider comparison mode for vision analysis.

Provides:
- Run multiple providers on same image
- Compare results and confidence scores
- Select best result or aggregate insights
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .base import VisionDescription, VisionProvider, VisionProviderError

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Strategy for selecting best result."""

    HIGHEST_CONFIDENCE = "highest_confidence"
    FIRST_SUCCESS = "first_success"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"


@dataclass
class ProviderResult:
    """Result from a single provider."""

    provider_name: str
    success: bool
    result: Optional[VisionDescription] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class ComparisonResult:
    """Result of multi-provider comparison."""

    provider_results: Dict[str, ProviderResult]
    selected_result: Optional[VisionDescription]
    selected_provider: Optional[str]
    aggregated_summary: Optional[str]
    total_time_ms: float
    strategy_used: SelectionStrategy

    @property
    def success_count(self) -> int:
        """Count successful providers."""
        return sum(1 for r in self.provider_results.values() if r.success)

    @property
    def providers_compared(self) -> int:
        """Total providers compared."""
        return len(self.provider_results)

    @property
    def confidence_scores(self) -> Dict[str, float]:
        """Get confidence scores from successful providers."""
        return {
            name: r.result.confidence
            for name, r in self.provider_results.items()
            if r.success and r.result
        }


class ProviderComparator:
    """
    Compare multiple vision providers on the same image.

    Features:
    - Concurrent provider execution
    - Multiple selection strategies
    - Result aggregation
    - Performance comparison
    """

    def __init__(
        self,
        providers: List[VisionProvider],
        selection_strategy: SelectionStrategy = SelectionStrategy.HIGHEST_CONFIDENCE,
    ):
        """
        Initialize provider comparator.

        Args:
            providers: List of providers to compare
            selection_strategy: How to select best result
        """
        self._providers = providers
        self._selection_strategy = selection_strategy

    async def compare(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> ComparisonResult:
        """
        Run all providers on same image and compare results.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include descriptions

        Returns:
            ComparisonResult with all provider results and selection
        """
        start_time = time.time()

        # Run all providers concurrently
        tasks = [
            self._run_provider(provider, image_data, include_description)
            for provider in self._providers
        ]

        results = await asyncio.gather(*tasks)

        # Build results dictionary
        provider_results = {
            r.provider_name: r for r in results
        }

        # Select best result
        selected_result, selected_provider = self._select_result(provider_results)

        # Generate aggregated summary
        aggregated = self._aggregate_summaries(provider_results)

        total_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Comparison complete: {sum(1 for r in results if r.success)}/"
            f"{len(results)} providers succeeded, "
            f"selected: {selected_provider}"
        )

        return ComparisonResult(
            provider_results=provider_results,
            selected_result=selected_result,
            selected_provider=selected_provider,
            aggregated_summary=aggregated,
            total_time_ms=total_time_ms,
            strategy_used=self._selection_strategy,
        )

    async def _run_provider(
        self,
        provider: VisionProvider,
        image_data: bytes,
        include_description: bool,
    ) -> ProviderResult:
        """Run single provider and capture result."""
        start_time = time.time()
        provider_name = provider.provider_name

        try:
            result = await provider.analyze_image(image_data, include_description)
            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"[{provider_name}] Success: confidence={result.confidence:.2f}, "
                f"latency={latency_ms:.0f}ms"
            )

            return ProviderResult(
                provider_name=provider_name,
                success=True,
                result=result,
                latency_ms=latency_ms,
            )

        except VisionProviderError as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"[{provider_name}] Failed: {e}")

            return ProviderResult(
                provider_name=provider_name,
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"[{provider_name}] Unexpected error: {e}")

            return ProviderResult(
                provider_name=provider_name,
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )

    def _select_result(
        self,
        results: Dict[str, ProviderResult],
    ) -> tuple[Optional[VisionDescription], Optional[str]]:
        """Select best result based on strategy."""
        successful = {
            name: r for name, r in results.items()
            if r.success and r.result
        }

        if not successful:
            return None, None

        if self._selection_strategy == SelectionStrategy.HIGHEST_CONFIDENCE:
            best_name = max(
                successful.keys(),
                key=lambda n: successful[n].result.confidence,  # type: ignore
            )
            return successful[best_name].result, best_name

        elif self._selection_strategy == SelectionStrategy.FIRST_SUCCESS:
            # Return first successful result
            first_name = next(iter(successful))
            return successful[first_name].result, first_name

        elif self._selection_strategy == SelectionStrategy.MAJORITY_VOTE:
            # For majority vote, we'd need semantic similarity
            # Fall back to highest confidence for now
            best_name = max(
                successful.keys(),
                key=lambda n: successful[n].result.confidence,  # type: ignore
            )
            return successful[best_name].result, best_name

        elif self._selection_strategy == SelectionStrategy.WEIGHTED_AVERAGE:
            # Compute weighted average confidence
            total_weight = sum(r.result.confidence for r in successful.values())  # type: ignore
            if total_weight == 0:
                best_name = next(iter(successful))
                return successful[best_name].result, best_name

            # Use result with highest confidence as base, adjust confidence
            best_name = max(
                successful.keys(),
                key=lambda n: successful[n].result.confidence,  # type: ignore
            )
            best_result = successful[best_name].result
            avg_confidence = total_weight / len(successful)

            return VisionDescription(
                summary=best_result.summary,  # type: ignore
                details=best_result.details,  # type: ignore
                confidence=avg_confidence,
            ), best_name

        return None, None

    def _aggregate_summaries(
        self,
        results: Dict[str, ProviderResult],
    ) -> Optional[str]:
        """Aggregate summaries from all successful providers."""
        summaries = [
            f"[{name}] {r.result.summary}"
            for name, r in results.items()
            if r.success and r.result
        ]

        if not summaries:
            return None

        if len(summaries) == 1:
            return summaries[0]

        return " | ".join(summaries)


async def compare_providers(
    image_data: bytes,
    providers: List[VisionProvider],
    strategy: SelectionStrategy = SelectionStrategy.HIGHEST_CONFIDENCE,
) -> ComparisonResult:
    """
    Convenience function for provider comparison.

    Args:
        image_data: Raw image bytes
        providers: List of providers to compare
        strategy: Selection strategy for best result

    Returns:
        ComparisonResult with all results and selection

    Example:
        >>> providers = [
        ...     create_vision_provider("openai"),
        ...     create_vision_provider("anthropic"),
        ...     create_vision_provider("deepseek"),
        ... ]
        >>> result = await compare_providers(image_bytes, providers)
        >>> print(f"Best: {result.selected_provider}")
        >>> print(f"Confidence scores: {result.confidence_scores}")
    """
    comparator = ProviderComparator(
        providers=providers,
        selection_strategy=strategy,
    )
    return await comparator.compare(image_data)
