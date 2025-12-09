"""
Data Masking Module for Vision Provider System.

Provides PII detection, data masking, anonymization, and redaction
capabilities for vision analysis results.
"""

import hashlib
import re
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple

from .base import VisionDescription, VisionProvider


# ============================================================================
# Enums and Types
# ============================================================================


class PIIType(Enum):
    """Types of Personally Identifiable Information."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    IP_ADDRESS = "ip_address"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    BANK_ACCOUNT = "bank_account"
    MEDICAL_ID = "medical_id"
    BIOMETRIC = "biometric"
    CUSTOM = "custom"


class MaskingStrategy(Enum):
    """Data masking strategies."""

    REDACT = "redact"  # Replace with [REDACTED]
    HASH = "hash"  # Replace with hash
    PARTIAL = "partial"  # Show partial data (e.g., ***-**-1234)
    PSEUDONYMIZE = "pseudonymize"  # Replace with fake but consistent data
    ENCRYPT = "encrypt"  # Encrypt the data
    TOKENIZE = "tokenize"  # Replace with a token
    GENERALIZE = "generalize"  # Replace with a general category
    NULL = "null"  # Replace with null/empty


class SensitivityLevel(Enum):
    """Data sensitivity levels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PIIMatch:
    """A detected PII match."""

    match_id: str
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskingRule:
    """A data masking rule."""

    rule_id: str
    pii_type: PIIType
    strategy: MaskingStrategy
    pattern: Optional[str] = None
    replacement: Optional[str] = None
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskingResult:
    """Result of masking operation."""

    original_text: str
    masked_text: str
    detections: List[PIIMatch] = field(default_factory=list)
    masks_applied: int = 0
    processing_time_ms: float = 0.0


@dataclass
class DetectionReport:
    """PII detection report."""

    report_id: str
    source: str
    total_detections: int
    detections_by_type: Dict[PIIType, int] = field(default_factory=dict)
    high_confidence_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# PII Detectors
# ============================================================================


class PIIDetector(ABC):
    """Abstract base for PII detection."""

    @abstractmethod
    def detect(self, text: str) -> List[PIIMatch]:
        """Detect PII in text."""
        pass


class RegexPIIDetector(PIIDetector):
    """Regex-based PII detector."""

    def __init__(self) -> None:
        self._patterns: Dict[PIIType, List[Tuple[Pattern[str], float]]] = {
            PIIType.EMAIL: [
                (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), 0.95),
            ],
            PIIType.PHONE: [
                (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), 0.85),
                (re.compile(r"\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b"), 0.90),
                (re.compile(r"\b\+\d{1,3}[-.\s]?\d{1,14}\b"), 0.80),
            ],
            PIIType.SSN: [
                (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0.95),
                (re.compile(r"\b\d{9}\b"), 0.50),  # Lower confidence for just 9 digits
            ],
            PIIType.CREDIT_CARD: [
                (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), 0.90),
                (re.compile(r"\b\d{16}\b"), 0.70),
            ],
            PIIType.IP_ADDRESS: [
                (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), 0.90),
            ],
            PIIType.DATE_OF_BIRTH: [
                (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), 0.60),
                (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), 0.70),
            ],
        }

    def detect(self, text: str) -> List[PIIMatch]:
        """Detect PII using regex patterns."""
        matches: List[PIIMatch] = []

        for pii_type, patterns in self._patterns.items():
            for pattern, confidence in patterns:
                for match in pattern.finditer(text):
                    # Get context around match
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end]

                    pii_match = PIIMatch(
                        match_id=str(uuid.uuid4()),
                        pii_type=pii_type,
                        value=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=confidence,
                        context=context,
                    )
                    matches.append(pii_match)

        return matches

    def add_pattern(
        self,
        pii_type: PIIType,
        pattern: str,
        confidence: float = 0.80,
    ) -> None:
        """Add a custom pattern."""
        if pii_type not in self._patterns:
            self._patterns[pii_type] = []
        self._patterns[pii_type].append((re.compile(pattern), confidence))


class HeuristicPIIDetector(PIIDetector):
    """Heuristic-based PII detector for names and addresses."""

    def __init__(self) -> None:
        # Common name prefixes and patterns
        self._name_prefixes = {"mr", "mrs", "ms", "dr", "prof", "sir", "lady"}
        self._address_keywords = {
            "street", "st", "avenue", "ave", "road", "rd", "boulevard",
            "blvd", "drive", "dr", "lane", "ln", "way", "court", "ct",
            "apartment", "apt", "suite", "ste", "floor", "fl",
        }

    def detect(self, text: str) -> List[PIIMatch]:
        """Detect PII using heuristics."""
        matches: List[PIIMatch] = []
        words = text.lower().split()

        # Detect potential names (capitalized words after prefixes)
        for i, word in enumerate(words):
            clean_word = word.strip(".,!?")
            if clean_word in self._name_prefixes and i + 1 < len(words):
                # Potential name after prefix
                name_match = PIIMatch(
                    match_id=str(uuid.uuid4()),
                    pii_type=PIIType.NAME,
                    value=f"{word} {words[i + 1]}",
                    start_pos=text.lower().find(word),
                    end_pos=text.lower().find(words[i + 1]) + len(words[i + 1]),
                    confidence=0.70,
                )
                matches.append(name_match)

        # Detect potential addresses
        for keyword in self._address_keywords:
            idx = text.lower().find(keyword)
            if idx != -1:
                # Find surrounding context
                start = max(0, idx - 30)
                end = min(len(text), idx + len(keyword) + 30)
                context = text[start:end]

                address_match = PIIMatch(
                    match_id=str(uuid.uuid4()),
                    pii_type=PIIType.ADDRESS,
                    value=context.strip(),
                    start_pos=start,
                    end_pos=end,
                    confidence=0.60,
                )
                matches.append(address_match)
                break  # Only report once per text

        return matches


# ============================================================================
# Data Maskers
# ============================================================================


class DataMasker(ABC):
    """Abstract base for data masking."""

    @abstractmethod
    def mask(self, value: str, pii_type: PIIType) -> str:
        """Mask a value."""
        pass


class RedactionMasker(DataMasker):
    """Redacts data with a placeholder."""

    def __init__(self, placeholder: str = "[REDACTED]") -> None:
        self._placeholder = placeholder

    def mask(self, value: str, pii_type: PIIType) -> str:
        """Redact value."""
        return self._placeholder


class PartialMasker(DataMasker):
    """Partially masks data, showing only some characters."""

    def __init__(self, show_chars: int = 4, mask_char: str = "*") -> None:
        self._show_chars = show_chars
        self._mask_char = mask_char

    def mask(self, value: str, pii_type: PIIType) -> str:
        """Partially mask value."""
        if len(value) <= self._show_chars:
            return self._mask_char * len(value)

        # Different masking based on PII type
        if pii_type == PIIType.EMAIL:
            parts = value.split("@")
            if len(parts) == 2:
                masked_local = self._mask_char * (len(parts[0]) - 1) + parts[0][-1]
                return f"{masked_local}@{parts[1]}"

        if pii_type == PIIType.SSN:
            return f"{self._mask_char * 3}-{self._mask_char * 2}-{value[-4:]}"

        if pii_type == PIIType.CREDIT_CARD:
            return f"{self._mask_char * 12}{value[-4:]}"

        if pii_type == PIIType.PHONE:
            return f"{self._mask_char * (len(value) - 4)}{value[-4:]}"

        # Default: mask all but last N characters
        masked_len = len(value) - self._show_chars
        return self._mask_char * masked_len + value[-self._show_chars:]


class HashMasker(DataMasker):
    """Masks data by hashing."""

    def __init__(self, algorithm: str = "sha256", truncate: int = 8) -> None:
        self._algorithm = algorithm
        self._truncate = truncate

    def mask(self, value: str, pii_type: PIIType) -> str:
        """Hash value."""
        hash_obj = hashlib.new(self._algorithm, value.encode())
        hashed = hash_obj.hexdigest()
        if self._truncate:
            return hashed[: self._truncate]
        return hashed


class TokenizeMasker(DataMasker):
    """Tokenizes data with a consistent token."""

    def __init__(self) -> None:
        self._token_map: Dict[str, str] = {}
        self._lock = threading.Lock()

    def mask(self, value: str, pii_type: PIIType) -> str:
        """Tokenize value."""
        with self._lock:
            if value not in self._token_map:
                token = f"TOKEN_{pii_type.value.upper()}_{len(self._token_map):06d}"
                self._token_map[value] = token
            return self._token_map[value]

    def get_original(self, token: str) -> Optional[str]:
        """Get original value from token (for authorized de-tokenization)."""
        for original, tok in self._token_map.items():
            if tok == token:
                return original
        return None


class PseudonymizeMasker(DataMasker):
    """Replaces data with consistent pseudonyms."""

    def __init__(self) -> None:
        self._pseudonym_map: Dict[str, str] = {}
        self._counters: Dict[PIIType, int] = {}
        self._lock = threading.Lock()

        # Pseudonym templates
        self._templates = {
            PIIType.NAME: "Person_{:04d}",
            PIIType.EMAIL: "user{:04d}@example.com",
            PIIType.PHONE: "555-000-{:04d}",
            PIIType.ADDRESS: "123 Fake Street #{:04d}",
            PIIType.SSN: "000-00-{:04d}",
        }

    def mask(self, value: str, pii_type: PIIType) -> str:
        """Pseudonymize value."""
        with self._lock:
            if value not in self._pseudonym_map:
                if pii_type not in self._counters:
                    self._counters[pii_type] = 0
                self._counters[pii_type] += 1

                template = self._templates.get(
                    pii_type, f"{pii_type.value}_{{:04d}}"
                )
                pseudonym = template.format(self._counters[pii_type])
                self._pseudonym_map[value] = pseudonym

            return self._pseudonym_map[value]


# ============================================================================
# Masking Engine
# ============================================================================


class MaskingEngine:
    """Comprehensive data masking engine."""

    def __init__(self) -> None:
        self._detectors: List[PIIDetector] = [
            RegexPIIDetector(),
            HeuristicPIIDetector(),
        ]
        self._maskers: Dict[MaskingStrategy, DataMasker] = {
            MaskingStrategy.REDACT: RedactionMasker(),
            MaskingStrategy.PARTIAL: PartialMasker(),
            MaskingStrategy.HASH: HashMasker(),
            MaskingStrategy.TOKENIZE: TokenizeMasker(),
            MaskingStrategy.PSEUDONYMIZE: PseudonymizeMasker(),
        }
        self._rules: Dict[str, MaskingRule] = {}
        self._default_strategy = MaskingStrategy.REDACT
        self._lock = threading.Lock()

    def add_detector(self, detector: PIIDetector) -> None:
        """Add a PII detector."""
        self._detectors.append(detector)

    def add_masker(self, strategy: MaskingStrategy, masker: DataMasker) -> None:
        """Add or replace a masker."""
        self._maskers[strategy] = masker

    def add_rule(self, rule: MaskingRule) -> None:
        """Add a masking rule."""
        with self._lock:
            self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a masking rule."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                return True
            return False

    def set_default_strategy(self, strategy: MaskingStrategy) -> None:
        """Set the default masking strategy."""
        self._default_strategy = strategy

    def detect_pii(self, text: str) -> List[PIIMatch]:
        """Detect all PII in text."""
        all_matches: List[PIIMatch] = []

        for detector in self._detectors:
            matches = detector.detect(text)
            all_matches.extend(matches)

        # Remove duplicates based on position
        seen_positions: Set[Tuple[int, int]] = set()
        unique_matches: List[PIIMatch] = []

        for match in sorted(all_matches, key=lambda m: -m.confidence):
            pos = (match.start_pos, match.end_pos)
            if pos not in seen_positions:
                seen_positions.add(pos)
                unique_matches.append(match)

        return unique_matches

    def mask_text(
        self,
        text: str,
        strategy: Optional[MaskingStrategy] = None,
        pii_types: Optional[Set[PIIType]] = None,
    ) -> MaskingResult:
        """Mask all detected PII in text."""
        import time

        start_time = time.time()
        detections = self.detect_pii(text)
        masked_text = text
        masks_applied = 0

        # Filter by PII types if specified
        if pii_types:
            detections = [d for d in detections if d.pii_type in pii_types]

        # Sort by position descending to replace from end
        detections.sort(key=lambda m: m.start_pos, reverse=True)

        for detection in detections:
            # Get strategy from rules or default
            mask_strategy = self._get_strategy_for_pii(detection.pii_type)
            if strategy:
                mask_strategy = strategy

            masker = self._maskers.get(mask_strategy)
            if masker:
                masked_value = masker.mask(detection.value, detection.pii_type)
                masked_text = (
                    masked_text[: detection.start_pos]
                    + masked_value
                    + masked_text[detection.end_pos:]
                )
                masks_applied += 1

        processing_time = (time.time() - start_time) * 1000

        return MaskingResult(
            original_text=text,
            masked_text=masked_text,
            detections=detections,
            masks_applied=masks_applied,
            processing_time_ms=processing_time,
        )

    def _get_strategy_for_pii(self, pii_type: PIIType) -> MaskingStrategy:
        """Get the masking strategy for a PII type."""
        for rule in sorted(self._rules.values(), key=lambda r: -r.priority):
            if rule.enabled and rule.pii_type == pii_type:
                return rule.strategy
        return self._default_strategy

    def generate_report(
        self,
        texts: List[str],
        source: str = "unknown",
    ) -> DetectionReport:
        """Generate a PII detection report."""
        detections_by_type: Dict[PIIType, int] = {}
        high_confidence_count = 0
        total_detections = 0

        for text in texts:
            matches = self.detect_pii(text)
            for match in matches:
                total_detections += 1
                if match.pii_type not in detections_by_type:
                    detections_by_type[match.pii_type] = 0
                detections_by_type[match.pii_type] += 1
                if match.confidence >= 0.80:
                    high_confidence_count += 1

        return DetectionReport(
            report_id=str(uuid.uuid4()),
            source=source,
            total_detections=total_detections,
            detections_by_type=detections_by_type,
            high_confidence_count=high_confidence_count,
        )


# ============================================================================
# Data Masking Vision Provider
# ============================================================================


class DataMaskingVisionProvider(VisionProvider):
    """Vision provider with data masking for results."""

    def __init__(
        self,
        provider: VisionProvider,
        masking_engine: Optional[MaskingEngine] = None,
        strategy: MaskingStrategy = MaskingStrategy.REDACT,
    ) -> None:
        self._provider = provider
        self._engine = masking_engine or MaskingEngine()
        self._strategy = strategy

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"masked_{self._provider.provider_name}"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        **kwargs: Any,
    ) -> VisionDescription:
        """Analyze image and mask PII in results."""
        result = await self._provider.analyze_image(image_data, include_description)

        # Mask PII in summary
        masked_summary = self._engine.mask_text(result.summary, self._strategy)

        # Mask PII in details
        masked_details = []
        for detail in result.details:
            masked_detail = self._engine.mask_text(detail, self._strategy)
            masked_details.append(masked_detail.masked_text)

        return VisionDescription(
            summary=masked_summary.masked_text,
            details=masked_details,
            confidence=result.confidence,
        )

    def get_masking_engine(self) -> MaskingEngine:
        """Get the masking engine."""
        return self._engine


# ============================================================================
# Factory Functions
# ============================================================================


def create_masking_engine() -> MaskingEngine:
    """Create a masking engine."""
    return MaskingEngine()


def create_regex_detector() -> RegexPIIDetector:
    """Create a regex PII detector."""
    return RegexPIIDetector()


def create_heuristic_detector() -> HeuristicPIIDetector:
    """Create a heuristic PII detector."""
    return HeuristicPIIDetector()


def create_masking_provider(
    provider: VisionProvider,
    engine: Optional[MaskingEngine] = None,
    strategy: MaskingStrategy = MaskingStrategy.REDACT,
) -> DataMaskingVisionProvider:
    """Create a data masking vision provider."""
    return DataMaskingVisionProvider(provider, engine, strategy)


def create_masking_rule(
    pii_type: PIIType,
    strategy: MaskingStrategy,
    priority: int = 0,
) -> MaskingRule:
    """Create a masking rule."""
    return MaskingRule(
        rule_id=str(uuid.uuid4()),
        pii_type=pii_type,
        strategy=strategy,
        priority=priority,
    )


def create_redaction_masker(placeholder: str = "[REDACTED]") -> RedactionMasker:
    """Create a redaction masker."""
    return RedactionMasker(placeholder)


def create_partial_masker(
    show_chars: int = 4,
    mask_char: str = "*",
) -> PartialMasker:
    """Create a partial masker."""
    return PartialMasker(show_chars, mask_char)


def create_hash_masker(
    algorithm: str = "sha256",
    truncate: int = 8,
) -> HashMasker:
    """Create a hash masker."""
    return HashMasker(algorithm, truncate)


def create_tokenize_masker() -> TokenizeMasker:
    """Create a tokenize masker."""
    return TokenizeMasker()


def create_pseudonymize_masker() -> PseudonymizeMasker:
    """Create a pseudonymize masker."""
    return PseudonymizeMasker()
