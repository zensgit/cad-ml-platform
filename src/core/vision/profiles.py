"""Provider configuration profiles for vision analysis.

Provides:
- Pre-configured provider settings
- Environment-based configuration
- Profile inheritance and composition
- Easy provider setup
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional



class ProfileType(Enum):
    """Types of configuration profiles."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    HIGH_PERFORMANCE = "high_performance"
    COST_OPTIMIZED = "cost_optimized"
    HIGH_ACCURACY = "high_accuracy"
    CUSTOM = "custom"


class ProviderType(Enum):
    """Supported provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    MOCK = "mock"


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_retries: int = 3
    initial_delay_ms: float = 1000.0
    max_delay_ms: float = 30000.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    enable_throttling: bool = True


@dataclass
class CacheConfig:
    """Caching configuration."""

    enabled: bool = True
    ttl_seconds: int = 3600
    max_entries: int = 1000
    cache_backend: str = "memory"  # memory, redis, file


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_requests: int = 3


@dataclass
class TimeoutConfig:
    """Timeout configuration."""

    connect_timeout_ms: float = 5000.0
    read_timeout_ms: float = 30000.0
    total_timeout_ms: float = 60000.0


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""

    provider_type: ProviderType
    api_key_env_var: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    weight: float = 1.0
    enabled: bool = True
    max_connections: int = 100

    # Nested configs
    retry: RetryConfig = field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)

    # Provider-specific options
    options: Dict[str, Any] = field(default_factory=dict)

    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        if self.api_key_env_var:
            return os.environ.get(self.api_key_env_var)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider_type": self.provider_type.value,
            "api_key_env_var": self.api_key_env_var,
            "base_url": self.base_url,
            "model": self.model,
            "weight": self.weight,
            "enabled": self.enabled,
            "max_connections": self.max_connections,
            "retry": {
                "max_retries": self.retry.max_retries,
                "initial_delay_ms": self.retry.initial_delay_ms,
                "max_delay_ms": self.retry.max_delay_ms,
            },
            "rate_limit": {
                "requests_per_minute": self.rate_limit.requests_per_minute,
                "requests_per_hour": self.rate_limit.requests_per_hour,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "ttl_seconds": self.cache.ttl_seconds,
            },
            "circuit_breaker": {
                "enabled": self.circuit_breaker.enabled,
                "failure_threshold": self.circuit_breaker.failure_threshold,
            },
            "timeout": {
                "connect_timeout_ms": self.timeout.connect_timeout_ms,
                "read_timeout_ms": self.timeout.read_timeout_ms,
            },
            "options": self.options,
        }


@dataclass
class ProfileConfig:
    """Complete profile configuration."""

    name: str
    profile_type: ProfileType
    description: str = ""

    # Provider configurations
    providers: List[ProviderConfig] = field(default_factory=list)
    primary_provider: Optional[ProviderType] = None
    fallback_providers: List[ProviderType] = field(default_factory=list)

    # Load balancing
    enable_load_balancing: bool = False
    load_balancing_algorithm: str = "weighted_round_robin"

    # Global settings
    enable_preprocessing: bool = True
    enable_logging: bool = True
    enable_persistence: bool = False
    enable_analytics: bool = False

    # Preprocessing defaults
    max_image_size_mb: float = 20.0
    max_image_dimension: int = 4096
    auto_resize: bool = True

    # Logging defaults
    log_level: str = "info"
    slow_request_threshold_ms: float = 5000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "profile_type": self.profile_type.value,
            "description": self.description,
            "providers": [p.to_dict() for p in self.providers],
            "primary_provider": (
                self.primary_provider.value if self.primary_provider else None
            ),
            "fallback_providers": [p.value for p in self.fallback_providers],
            "enable_load_balancing": self.enable_load_balancing,
            "load_balancing_algorithm": self.load_balancing_algorithm,
            "enable_preprocessing": self.enable_preprocessing,
            "enable_logging": self.enable_logging,
            "enable_persistence": self.enable_persistence,
            "enable_analytics": self.enable_analytics,
        }


class ProfileManager:
    """
    Manages configuration profiles for vision providers.

    Features:
    - Pre-built profiles for common scenarios
    - Profile inheritance and customization
    - Environment-aware configuration
    - Profile validation
    """

    def __init__(self) -> None:
        """Initialize profile manager."""
        self._profiles: Dict[str, ProfileConfig] = {}
        self._register_builtin_profiles()

    def _register_builtin_profiles(self) -> None:
        """Register built-in profiles."""
        # Development profile
        self.register_profile(
            ProfileConfig(
                name="development",
                profile_type=ProfileType.DEVELOPMENT,
                description="Development environment with relaxed limits",
                providers=[
                    ProviderConfig(
                        provider_type=ProviderType.OPENAI,
                        api_key_env_var="OPENAI_API_KEY",
                        model="gpt-4-vision-preview",
                        rate_limit=RateLimitConfig(
                            requests_per_minute=20,
                            requests_per_hour=200,
                        ),
                        cache=CacheConfig(enabled=True, ttl_seconds=7200),
                    ),
                ],
                primary_provider=ProviderType.OPENAI,
                enable_load_balancing=False,
                enable_preprocessing=True,
                enable_logging=True,
                log_level="debug",
            )
        )

        # Testing profile
        self.register_profile(
            ProfileConfig(
                name="testing",
                profile_type=ProfileType.TESTING,
                description="Testing environment with mock provider",
                providers=[
                    ProviderConfig(
                        provider_type=ProviderType.MOCK,
                        enabled=True,
                        cache=CacheConfig(enabled=False),
                        circuit_breaker=CircuitBreakerConfig(enabled=False),
                    ),
                ],
                primary_provider=ProviderType.MOCK,
                enable_load_balancing=False,
                enable_preprocessing=False,
                enable_logging=True,
                log_level="debug",
            )
        )

        # Production profile
        self.register_profile(
            ProfileConfig(
                name="production",
                profile_type=ProfileType.PRODUCTION,
                description="Production environment with all features enabled",
                providers=[
                    ProviderConfig(
                        provider_type=ProviderType.OPENAI,
                        api_key_env_var="OPENAI_API_KEY",
                        model="gpt-4-vision-preview",
                        weight=2.0,
                        retry=RetryConfig(max_retries=3),
                        rate_limit=RateLimitConfig(
                            requests_per_minute=60,
                            requests_per_hour=1000,
                        ),
                        circuit_breaker=CircuitBreakerConfig(
                            enabled=True,
                            failure_threshold=5,
                        ),
                    ),
                    ProviderConfig(
                        provider_type=ProviderType.ANTHROPIC,
                        api_key_env_var="ANTHROPIC_API_KEY",
                        model="claude-3-opus-20240229",
                        weight=1.5,
                        retry=RetryConfig(max_retries=3),
                    ),
                    ProviderConfig(
                        provider_type=ProviderType.DEEPSEEK,
                        api_key_env_var="DEEPSEEK_API_KEY",
                        weight=1.0,
                    ),
                ],
                primary_provider=ProviderType.OPENAI,
                fallback_providers=[ProviderType.ANTHROPIC, ProviderType.DEEPSEEK],
                enable_load_balancing=True,
                load_balancing_algorithm="weighted_round_robin",
                enable_preprocessing=True,
                enable_logging=True,
                enable_persistence=True,
                enable_analytics=True,
                log_level="info",
                slow_request_threshold_ms=5000.0,
            )
        )

        # High performance profile
        self.register_profile(
            ProfileConfig(
                name="high_performance",
                profile_type=ProfileType.HIGH_PERFORMANCE,
                description="Optimized for speed with aggressive caching",
                providers=[
                    ProviderConfig(
                        provider_type=ProviderType.OPENAI,
                        api_key_env_var="OPENAI_API_KEY",
                        model="gpt-4-vision-preview",
                        weight=1.0,
                        max_connections=200,
                        timeout=TimeoutConfig(
                            connect_timeout_ms=3000.0,
                            read_timeout_ms=15000.0,
                        ),
                        cache=CacheConfig(
                            enabled=True,
                            ttl_seconds=86400,  # 24 hours
                            max_entries=10000,
                        ),
                    ),
                ],
                primary_provider=ProviderType.OPENAI,
                enable_load_balancing=False,
                enable_preprocessing=True,
                enable_logging=False,  # Disabled for performance
                max_image_dimension=2048,  # Smaller for speed
                auto_resize=True,
            )
        )

        # Cost optimized profile
        self.register_profile(
            ProfileConfig(
                name="cost_optimized",
                profile_type=ProfileType.COST_OPTIMIZED,
                description="Minimizes API costs with aggressive caching and rate limiting",
                providers=[
                    ProviderConfig(
                        provider_type=ProviderType.DEEPSEEK,
                        api_key_env_var="DEEPSEEK_API_KEY",
                        weight=2.0,  # Prefer cheaper provider
                        rate_limit=RateLimitConfig(
                            requests_per_minute=30,
                            requests_per_hour=500,
                        ),
                        cache=CacheConfig(
                            enabled=True,
                            ttl_seconds=604800,  # 1 week
                            max_entries=50000,
                        ),
                    ),
                    ProviderConfig(
                        provider_type=ProviderType.OPENAI,
                        api_key_env_var="OPENAI_API_KEY",
                        model="gpt-4-vision-preview",
                        weight=0.5,  # Lower weight
                        rate_limit=RateLimitConfig(
                            requests_per_minute=10,
                            requests_per_hour=100,
                        ),
                    ),
                ],
                primary_provider=ProviderType.DEEPSEEK,
                fallback_providers=[ProviderType.OPENAI],
                enable_load_balancing=True,
                load_balancing_algorithm="weighted_random",
                enable_preprocessing=True,
                max_image_size_mb=10.0,  # Smaller to reduce tokens
                max_image_dimension=2048,
                auto_resize=True,
            )
        )

        # High accuracy profile
        self.register_profile(
            ProfileConfig(
                name="high_accuracy",
                profile_type=ProfileType.HIGH_ACCURACY,
                description="Maximum accuracy with multiple providers and comparison",
                providers=[
                    ProviderConfig(
                        provider_type=ProviderType.OPENAI,
                        api_key_env_var="OPENAI_API_KEY",
                        model="gpt-4-vision-preview",
                        weight=1.0,
                    ),
                    ProviderConfig(
                        provider_type=ProviderType.ANTHROPIC,
                        api_key_env_var="ANTHROPIC_API_KEY",
                        model="claude-3-opus-20240229",
                        weight=1.0,
                    ),
                ],
                primary_provider=ProviderType.OPENAI,
                fallback_providers=[ProviderType.ANTHROPIC],
                enable_load_balancing=False,  # Use comparison instead
                enable_preprocessing=True,
                enable_logging=True,
                enable_persistence=True,
                enable_analytics=True,
                max_image_dimension=4096,  # Full resolution
                auto_resize=False,  # Keep original quality
            )
        )

    def register_profile(self, profile: ProfileConfig) -> None:
        """Register a profile."""
        self._profiles[profile.name] = profile

    def get_profile(self, name: str) -> Optional[ProfileConfig]:
        """Get a profile by name."""
        return self._profiles.get(name)

    def list_profiles(self) -> List[str]:
        """List all registered profile names."""
        return list(self._profiles.keys())

    def get_profile_info(self) -> List[Dict[str, Any]]:
        """Get information about all profiles."""
        return [
            {
                "name": p.name,
                "type": p.profile_type.value,
                "description": p.description,
                "providers": [pr.provider_type.value for pr in p.providers],
            }
            for p in self._profiles.values()
        ]

    def create_custom_profile(
        self,
        name: str,
        base_profile: Optional[str] = None,
        **overrides: Any,
    ) -> ProfileConfig:
        """
        Create a custom profile, optionally based on an existing one.

        Args:
            name: Name for the new profile
            base_profile: Optional base profile to inherit from
            **overrides: Configuration overrides

        Returns:
            New ProfileConfig instance
        """
        if base_profile and base_profile in self._profiles:
            base = self._profiles[base_profile]
            # Create new config with overrides
            config_dict = base.to_dict()
            config_dict.update(overrides)
            config_dict["name"] = name
            config_dict["profile_type"] = ProfileType.CUSTOM

            # Reconstruct ProfileConfig
            profile = ProfileConfig(
                name=name,
                profile_type=ProfileType.CUSTOM,
                description=overrides.get("description", f"Custom profile based on {base_profile}"),
                providers=base.providers.copy(),
                primary_provider=base.primary_provider,
                fallback_providers=base.fallback_providers.copy(),
                enable_load_balancing=overrides.get(
                    "enable_load_balancing", base.enable_load_balancing
                ),
                enable_preprocessing=overrides.get(
                    "enable_preprocessing", base.enable_preprocessing
                ),
                enable_logging=overrides.get("enable_logging", base.enable_logging),
                enable_persistence=overrides.get(
                    "enable_persistence", base.enable_persistence
                ),
                enable_analytics=overrides.get(
                    "enable_analytics", base.enable_analytics
                ),
            )
        else:
            # Create from scratch
            profile = ProfileConfig(
                name=name,
                profile_type=ProfileType.CUSTOM,
                **overrides,
            )

        self._profiles[name] = profile
        return profile

    def validate_profile(self, name: str) -> Dict[str, Any]:
        """
        Validate a profile configuration.

        Args:
            name: Profile name to validate

        Returns:
            Validation result with any issues found
        """
        profile = self._profiles.get(name)
        if not profile:
            return {"valid": False, "errors": [f"Profile '{name}' not found"]}

        errors: List[str] = []
        warnings: List[str] = []

        # Check providers
        if not profile.providers:
            errors.append("No providers configured")

        for provider in profile.providers:
            if provider.enabled and not provider.get_api_key():
                warnings.append(
                    f"Provider {provider.provider_type.value}: "
                    f"API key not found (env: {provider.api_key_env_var})"
                )

        # Check primary provider
        if profile.primary_provider:
            primary_found = any(
                p.provider_type == profile.primary_provider and p.enabled
                for p in profile.providers
            )
            if not primary_found:
                errors.append(
                    f"Primary provider {profile.primary_provider.value} "
                    "not found or disabled"
                )

        # Check fallback providers
        for fallback in profile.fallback_providers:
            fallback_found = any(
                p.provider_type == fallback and p.enabled for p in profile.providers
            )
            if not fallback_found:
                warnings.append(
                    f"Fallback provider {fallback.value} not found or disabled"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


class ProfileEnvironmentLoader:
    """
    Loads profile configuration from environment variables.

    Supports loading provider configs and overrides from environment.
    """

    def __init__(self, prefix: str = "VISION_"):
        """
        Initialize loader.

        Args:
            prefix: Environment variable prefix
        """
        self._prefix = prefix

    def get_profile_name(self) -> str:
        """Get profile name from environment."""
        return os.environ.get(f"{self._prefix}PROFILE", "development")

    def get_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment."""
        overrides: Dict[str, Any] = {}

        # Check for common overrides
        if f"{self._prefix}LOG_LEVEL" in os.environ:
            overrides["log_level"] = os.environ[f"{self._prefix}LOG_LEVEL"]

        if f"{self._prefix}ENABLE_CACHING" in os.environ:
            value = os.environ[f"{self._prefix}ENABLE_CACHING"].lower()
            overrides["enable_caching"] = value in ("true", "1", "yes")

        if f"{self._prefix}ENABLE_LOGGING" in os.environ:
            value = os.environ[f"{self._prefix}ENABLE_LOGGING"].lower()
            overrides["enable_logging"] = value in ("true", "1", "yes")

        if f"{self._prefix}MAX_IMAGE_SIZE_MB" in os.environ:
            try:
                overrides["max_image_size_mb"] = float(
                    os.environ[f"{self._prefix}MAX_IMAGE_SIZE_MB"]
                )
            except ValueError:
                pass

        return overrides


# Global profile manager
_profile_manager: Optional[ProfileManager] = None


def get_profile_manager() -> ProfileManager:
    """
    Get the global profile manager instance.

    Returns:
        ProfileManager singleton
    """
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
    return _profile_manager


def get_profile(name: Optional[str] = None) -> Optional[ProfileConfig]:
    """
    Get a profile by name or from environment.

    Args:
        name: Optional profile name (defaults to environment setting)

    Returns:
        ProfileConfig or None if not found
    """
    manager = get_profile_manager()
    if name is None:
        loader = ProfileEnvironmentLoader()
        name = loader.get_profile_name()
    return manager.get_profile(name)


def create_profile(
    name: str,
    providers: List[Dict[str, Any]],
    **options: Any,
) -> ProfileConfig:
    """
    Factory to create a custom profile.

    Args:
        name: Profile name
        providers: List of provider configurations
        **options: Additional profile options

    Returns:
        ProfileConfig instance

    Example:
        >>> profile = create_profile(
        ...     name="my_profile",
        ...     providers=[
        ...         {"provider_type": "openai", "api_key_env_var": "MY_OPENAI_KEY"},
        ...         {"provider_type": "anthropic", "weight": 0.5},
        ...     ],
        ...     enable_load_balancing=True,
        ...     enable_logging=True,
        ... )
    """
    provider_configs = []
    for p in providers:
        provider_type = ProviderType(p.get("provider_type", "openai"))
        provider_configs.append(
            ProviderConfig(
                provider_type=provider_type,
                api_key_env_var=p.get("api_key_env_var", ""),
                api_key=p.get("api_key"),
                base_url=p.get("base_url"),
                model=p.get("model"),
                weight=p.get("weight", 1.0),
                enabled=p.get("enabled", True),
                options=p.get("options", {}),
            )
        )

    profile = ProfileConfig(
        name=name,
        profile_type=ProfileType.CUSTOM,
        providers=provider_configs,
        **options,
    )

    get_profile_manager().register_profile(profile)
    return profile
