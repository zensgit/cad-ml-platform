"""Feature Flag Decorators for CAD ML Platform."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Optional, TypeVar

from .client import FlagContext, get_feature_client

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def feature_flag(
    flag_name: str,
    default: bool = False,
    fallback: Optional[Callable[..., Any]] = None,
    context_extractor: Optional[Callable[..., FlagContext]] = None,
) -> Callable[[F], F]:
    """Decorator to gate function execution behind a feature flag.

    Args:
        flag_name: Name of the feature flag
        default: Default value if flag not found
        fallback: Fallback function to call if flag is disabled
        context_extractor: Function to extract FlagContext from args

    Example:
        @feature_flag("new_algorithm", fallback=old_algorithm)
        def new_algorithm(data):
            return process_v2(data)

        @feature_flag("beta_feature", context_extractor=lambda req: FlagContext(user_id=req.user_id))
        def beta_endpoint(request):
            return beta_response(request)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract context if extractor provided
            context = None
            if context_extractor:
                try:
                    context = context_extractor(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Failed to extract flag context: {e}")

            # Check if flag is enabled
            client = get_feature_client()
            if client.is_enabled(flag_name, context, default):
                return func(*args, **kwargs)
            elif fallback:
                return fallback(*args, **kwargs)
            else:
                logger.debug(f"Feature flag '{flag_name}' is disabled, skipping {func.__name__}")
                return None

        return wrapper  # type: ignore

    return decorator


def feature_variant(
    flag_name: str,
    variants: dict[str, Callable[..., Any]],
    default_variant: str = "control",
    context_extractor: Optional[Callable[..., FlagContext]] = None,
) -> Callable[[F], F]:
    """Decorator for A/B testing with multiple variants.

    Args:
        flag_name: Name of the feature flag
        variants: Dict of variant name to implementation
        default_variant: Default variant if flag evaluation fails
        context_extractor: Function to extract FlagContext from args

    Example:
        @feature_variant(
            "checkout_flow",
            variants={
                "control": checkout_v1,
                "variant_a": checkout_v2,
                "variant_b": checkout_v3,
            }
        )
        def checkout(cart):
            pass  # Implementation selected by decorator
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            context = None
            if context_extractor:
                try:
                    context = context_extractor(*args, **kwargs)
                except Exception:
                    pass

            # Get variant based on user bucket
            if context:
                bucket = context.get_percentage_bucket()
                variant_count = len(variants)
                variant_index = bucket % variant_count
                variant_name = list(variants.keys())[variant_index]
            else:
                variant_name = default_variant

            # Call selected variant
            variant_func = variants.get(variant_name, variants.get(default_variant))
            if variant_func:
                logger.debug(f"A/B test '{flag_name}': using variant '{variant_name}'")
                return variant_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class FeatureFlagMiddleware:
    """ASGI middleware for feature flag context injection."""

    def __init__(self, app: Any, client: Optional[Any] = None):
        self.app = app
        self.client = client or get_feature_client()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            # Extract user/tenant from headers or auth
            headers = dict(scope.get("headers", []))
            user_id = headers.get(b"x-user-id", b"").decode()
            tenant_id = headers.get(b"x-tenant-id", b"").decode()

            # Store context in scope state
            scope["state"] = scope.get("state", {})
            scope["state"]["feature_context"] = FlagContext(
                user_id=user_id or None,
                tenant_id=tenant_id or None,
                environment=scope.get("app", {}).get("environment"),
            )

        await self.app(scope, receive, send)
