"""Provider capability discovery module for Vision Provider system.

This module provides capability discovery including:
- Provider capability enumeration
- Feature detection
- Capability requirements matching
- Dynamic capability updates
- Capability-based routing
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from .base import VisionDescription, VisionProvider


class Capability(Enum):
    """Provider capabilities."""

    # Core capabilities
    IMAGE_ANALYSIS = auto()
    OCR = auto()
    OBJECT_DETECTION = auto()
    SCENE_DESCRIPTION = auto()

    # Format support
    FORMAT_JPEG = auto()
    FORMAT_PNG = auto()
    FORMAT_WEBP = auto()
    FORMAT_GIF = auto()
    FORMAT_BMP = auto()
    FORMAT_TIFF = auto()
    FORMAT_PDF = auto()

    # Size capabilities
    SIZE_SMALL = auto()  # < 1MB
    SIZE_MEDIUM = auto()  # 1-5MB
    SIZE_LARGE = auto()  # 5-20MB
    SIZE_XLARGE = auto()  # > 20MB

    # Resolution capabilities
    RESOLUTION_LOW = auto()  # < 1MP
    RESOLUTION_MEDIUM = auto()  # 1-4MP
    RESOLUTION_HIGH = auto()  # 4-12MP
    RESOLUTION_ULTRA = auto()  # > 12MP

    # Advanced features
    BATCH_PROCESSING = auto()
    STREAMING = auto()
    ASYNC_PROCESSING = auto()
    WEBHOOK_CALLBACKS = auto()

    # Domain-specific
    ENGINEERING_DRAWINGS = auto()
    MEDICAL_IMAGES = auto()
    DOCUMENT_ANALYSIS = auto()
    FACE_DETECTION = auto()

    # Quality features
    CONFIDENCE_SCORES = auto()
    BOUNDING_BOXES = auto()
    MULTI_LANGUAGE = auto()


class CapabilityStatus(Enum):
    """Status of a capability."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class CapabilityInfo:
    """Information about a capability."""

    capability: Capability
    status: CapabilityStatus = CapabilityStatus.UNKNOWN
    version: Optional[str] = None
    limits: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_checked: Optional[datetime] = None

    def is_available(self) -> bool:
        """Check if capability is available."""
        return self.status in (CapabilityStatus.AVAILABLE, CapabilityStatus.DEGRADED)


@dataclass
class CapabilityRequirement:
    """Requirement for a capability."""

    capability: Capability
    required: bool = True  # False means optional/preferred
    min_version: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderCapabilities:
    """Capabilities of a provider."""

    provider_name: str
    capabilities: Dict[Capability, CapabilityInfo] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_capability(self, capability: Capability) -> bool:
        """Check if provider has a capability."""
        info = self.capabilities.get(capability)
        return info is not None and info.is_available()

    def get_capability(self, capability: Capability) -> Optional[CapabilityInfo]:
        """Get capability info."""
        return self.capabilities.get(capability)

    def add_capability(
        self,
        capability: Capability,
        status: CapabilityStatus = CapabilityStatus.AVAILABLE,
        version: Optional[str] = None,
        limits: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update a capability."""
        self.capabilities[capability] = CapabilityInfo(
            capability=capability,
            status=status,
            version=version,
            limits=limits or {},
            last_checked=datetime.now(),
        )
        self.last_updated = datetime.now()

    def remove_capability(self, capability: Capability) -> None:
        """Remove a capability."""
        self.capabilities.pop(capability, None)
        self.last_updated = datetime.now()

    def get_all_available(self) -> List[Capability]:
        """Get all available capabilities."""
        return [c for c, info in self.capabilities.items() if info.is_available()]


@dataclass
class MatchResult:
    """Result of capability matching."""

    provider_name: str
    matches: bool
    score: float  # 0.0 to 1.0
    matched_requirements: List[CapabilityRequirement] = field(default_factory=list)
    missing_requirements: List[CapabilityRequirement] = field(default_factory=list)
    optional_matches: List[CapabilityRequirement] = field(default_factory=list)


class CapabilityDiscovery:
    """Discovers and tracks provider capabilities."""

    def __init__(self) -> None:
        """Initialize capability discovery."""
        self._providers: Dict[str, ProviderCapabilities] = {}
        self._capability_tests: Dict[
            Capability, Callable[[VisionProvider], bool]
        ] = {}

    def register_provider(
        self,
        provider_name: str,
        capabilities: Optional[List[Capability]] = None,
    ) -> ProviderCapabilities:
        """Register a provider with capabilities.

        Args:
            provider_name: Provider name
            capabilities: Initial capabilities

        Returns:
            ProviderCapabilities instance
        """
        provider_caps = ProviderCapabilities(provider_name=provider_name)

        if capabilities:
            for cap in capabilities:
                provider_caps.add_capability(cap)

        self._providers[provider_name] = provider_caps
        return provider_caps

    def unregister_provider(self, provider_name: str) -> None:
        """Unregister a provider."""
        self._providers.pop(provider_name, None)

    def get_provider_capabilities(
        self, provider_name: str
    ) -> Optional[ProviderCapabilities]:
        """Get capabilities for a provider."""
        return self._providers.get(provider_name)

    def update_capability(
        self,
        provider_name: str,
        capability: Capability,
        status: CapabilityStatus,
        version: Optional[str] = None,
        limits: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a provider's capability."""
        if provider_name not in self._providers:
            self.register_provider(provider_name)

        self._providers[provider_name].add_capability(
            capability, status, version, limits
        )

    def register_capability_test(
        self,
        capability: Capability,
        test_fn: Callable[[VisionProvider], bool],
    ) -> None:
        """Register a test function for a capability."""
        self._capability_tests[capability] = test_fn

    async def discover_capabilities(
        self,
        provider: VisionProvider,
    ) -> ProviderCapabilities:
        """Discover capabilities of a provider.

        Args:
            provider: Vision provider to discover

        Returns:
            Discovered capabilities
        """
        provider_name = provider.provider_name
        caps = self._providers.get(
            provider_name,
            ProviderCapabilities(provider_name=provider_name),
        )

        # Run capability tests
        for capability, test_fn in self._capability_tests.items():
            try:
                if test_fn(provider):
                    caps.add_capability(capability, CapabilityStatus.AVAILABLE)
                else:
                    caps.add_capability(capability, CapabilityStatus.UNAVAILABLE)
            except Exception:
                caps.add_capability(capability, CapabilityStatus.UNKNOWN)

        self._providers[provider_name] = caps
        return caps

    def match_requirements(
        self,
        provider_name: str,
        requirements: List[CapabilityRequirement],
    ) -> MatchResult:
        """Match a provider against requirements.

        Args:
            provider_name: Provider to check
            requirements: Requirements to match

        Returns:
            MatchResult
        """
        caps = self._providers.get(provider_name)
        if not caps:
            return MatchResult(
                provider_name=provider_name,
                matches=False,
                score=0.0,
                missing_requirements=requirements,
            )

        matched: List[CapabilityRequirement] = []
        missing: List[CapabilityRequirement] = []
        optional_matches: List[CapabilityRequirement] = []

        for req in requirements:
            has_cap = caps.has_capability(req.capability)

            if has_cap:
                if req.required:
                    matched.append(req)
                else:
                    optional_matches.append(req)
            elif req.required:
                missing.append(req)

        # Calculate score
        required_count = sum(1 for r in requirements if r.required)
        optional_count = len(requirements) - required_count

        if required_count > 0:
            required_score = len(matched) / required_count
        else:
            required_score = 1.0

        if optional_count > 0:
            optional_score = len(optional_matches) / optional_count
        else:
            optional_score = 1.0

        # Weight required capabilities more heavily
        score = required_score * 0.8 + optional_score * 0.2

        return MatchResult(
            provider_name=provider_name,
            matches=len(missing) == 0,
            score=score,
            matched_requirements=matched,
            missing_requirements=missing,
            optional_matches=optional_matches,
        )

    def find_matching_providers(
        self,
        requirements: List[CapabilityRequirement],
        min_score: float = 0.0,
    ) -> List[MatchResult]:
        """Find providers matching requirements.

        Args:
            requirements: Requirements to match
            min_score: Minimum score threshold

        Returns:
            List of matching providers sorted by score
        """
        results: List[MatchResult] = []

        for provider_name in self._providers:
            result = self.match_requirements(provider_name, requirements)
            if result.score >= min_score:
                results.append(result)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def get_providers_with_capability(
        self, capability: Capability
    ) -> List[str]:
        """Get providers with a specific capability."""
        return [
            name
            for name, caps in self._providers.items()
            if caps.has_capability(capability)
        ]

    def get_all_providers(self) -> List[str]:
        """Get all registered providers."""
        return list(self._providers.keys())


class CapabilityAwareVisionProvider(VisionProvider):
    """Vision provider with capability awareness."""

    def __init__(
        self,
        provider: VisionProvider,
        discovery: CapabilityDiscovery,
        capabilities: Optional[List[Capability]] = None,
    ) -> None:
        """Initialize capability-aware provider.

        Args:
            provider: Underlying vision provider
            discovery: Capability discovery instance
            capabilities: Initial capabilities to register
        """
        self._provider = provider
        self._discovery = discovery

        # Register with discovery
        self._discovery.register_provider(
            provider.provider_name,
            capabilities or [],
        )

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return self._provider.provider_name

    @property
    def capabilities(self) -> Optional[ProviderCapabilities]:
        """Get provider capabilities."""
        return self._discovery.get_provider_capabilities(self._provider.provider_name)

    def has_capability(self, capability: Capability) -> bool:
        """Check if provider has a capability."""
        caps = self.capabilities
        return caps is not None and caps.has_capability(capability)

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        return await self._provider.analyze_image(image_data, include_description)


# Global discovery instance
_discovery: Optional[CapabilityDiscovery] = None


def get_capability_discovery() -> CapabilityDiscovery:
    """Get global capability discovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = CapabilityDiscovery()
    return _discovery


def create_capability_aware_provider(
    provider: VisionProvider,
    capabilities: Optional[List[Capability]] = None,
    discovery: Optional[CapabilityDiscovery] = None,
) -> CapabilityAwareVisionProvider:
    """Create a capability-aware provider.

    Args:
        provider: Underlying vision provider
        capabilities: Initial capabilities
        discovery: Optional discovery instance

    Returns:
        CapabilityAwareVisionProvider instance
    """
    if discovery is None:
        discovery = get_capability_discovery()

    return CapabilityAwareVisionProvider(
        provider=provider,
        discovery=discovery,
        capabilities=capabilities,
    )
