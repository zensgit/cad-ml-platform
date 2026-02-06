"""Schema Registry.

Provides schema registry functionality:
- Schema registration
- Version management
- Subject management
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from src.core.schema_registry.schema import (
    CompatibilityMode,
    Schema,
    SchemaMetadata,
    SchemaReference,
    SchemaType,
    get_parser,
)
from src.core.schema_registry.compatibility import (
    CompatibilityResult,
    get_compatibility_checker,
)

logger = logging.getLogger(__name__)


class SchemaRegistryError(Exception):
    """Base exception for schema registry errors."""
    pass


class SchemaNotFoundError(SchemaRegistryError):
    """Schema not found."""
    pass


class IncompatibleSchemaError(SchemaRegistryError):
    """Schema is incompatible with previous versions."""

    def __init__(self, message: str, result: CompatibilityResult):
        super().__init__(message)
        self.result = result


class InvalidSchemaError(SchemaRegistryError):
    """Schema is invalid."""

    def __init__(self, message: str, errors: List[str]):
        super().__init__(message)
        self.errors = errors


@dataclass
class SubjectConfig:
    """Configuration for a subject."""
    compatibility_mode: CompatibilityMode = CompatibilityMode.BACKWARD
    normalize: bool = True


class SchemaRegistry(ABC):
    """Abstract base class for schema registries."""

    @abstractmethod
    async def register_schema(
        self,
        subject: str,
        schema: Schema,
        normalize: bool = True,
    ) -> int:
        """Register a schema under a subject. Returns schema ID."""
        pass

    @abstractmethod
    async def get_schema(self, schema_id: int) -> Optional[Schema]:
        """Get schema by ID."""
        pass

    @abstractmethod
    async def get_latest_schema(self, subject: str) -> Optional[Schema]:
        """Get latest schema for a subject."""
        pass

    @abstractmethod
    async def get_schema_by_version(
        self,
        subject: str,
        version: int,
    ) -> Optional[Schema]:
        """Get schema by subject and version."""
        pass

    @abstractmethod
    async def get_versions(self, subject: str) -> List[int]:
        """Get all versions for a subject."""
        pass

    @abstractmethod
    async def get_subjects(self) -> List[str]:
        """Get all subjects."""
        pass

    @abstractmethod
    async def delete_schema(self, subject: str, version: int) -> bool:
        """Delete a specific schema version."""
        pass

    @abstractmethod
    async def delete_subject(self, subject: str) -> List[int]:
        """Delete all schemas for a subject. Returns deleted versions."""
        pass

    @abstractmethod
    async def check_compatibility(
        self,
        subject: str,
        schema: Schema,
    ) -> CompatibilityResult:
        """Check if schema is compatible with existing versions."""
        pass

    @abstractmethod
    async def get_compatibility(self, subject: str) -> CompatibilityMode:
        """Get compatibility mode for a subject."""
        pass

    @abstractmethod
    async def set_compatibility(
        self,
        subject: str,
        mode: CompatibilityMode,
    ) -> None:
        """Set compatibility mode for a subject."""
        pass


class InMemorySchemaRegistry(SchemaRegistry):
    """In-memory schema registry implementation."""

    def __init__(self, default_compatibility: CompatibilityMode = CompatibilityMode.BACKWARD):
        self.default_compatibility = default_compatibility
        self._schemas: Dict[int, Schema] = {}
        self._subjects: Dict[str, Dict[int, int]] = {}  # subject -> {version: schema_id}
        self._subject_configs: Dict[str, SubjectConfig] = {}
        self._fingerprints: Dict[str, int] = {}  # fingerprint -> schema_id
        self._next_id = 1
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_compatibility_unlocked(self, subject: str) -> CompatibilityMode:
        """Read compatibility mode without re-acquiring the async lock."""
        if subject in self._subject_configs:
            return self._subject_configs[subject].compatibility_mode
        return self.default_compatibility

    async def register_schema(
        self,
        subject: str,
        schema: Schema,
        normalize: bool = True,
    ) -> int:
        async with self._get_lock():
            # Validate schema
            parser = get_parser(schema.schema_type)
            errors = parser.validate(schema.schema_str)
            if errors:
                raise InvalidSchemaError("Invalid schema", errors)

            # Normalize if requested
            if normalize:
                schema.schema_str = parser.normalize(schema.schema_str)

            # Check if schema already exists (by fingerprint)
            fingerprint = schema.fingerprint
            if fingerprint in self._fingerprints:
                existing_id = self._fingerprints[fingerprint]
                # Check if already registered for this subject
                if subject in self._subjects:
                    for version, schema_id in self._subjects[subject].items():
                        if schema_id == existing_id:
                            return existing_id
                # Register existing schema under new subject/version
                schema_id = existing_id
            else:
                # New schema - check compatibility
                existing_schemas = await self._get_all_schemas(subject)
                if existing_schemas:
                    mode = self._get_compatibility_unlocked(subject)
                    if mode != CompatibilityMode.NONE:
                        checker = get_compatibility_checker(schema.schema_type)
                        result = checker.check(schema, existing_schemas, mode)
                        if not result.compatible:
                            raise IncompatibleSchemaError(
                                f"Schema is incompatible: {', '.join(result.errors)}",
                                result,
                            )

                # Assign new ID
                schema_id = self._next_id
                self._next_id += 1
                self._fingerprints[fingerprint] = schema_id

            # Determine version
            if subject not in self._subjects:
                self._subjects[subject] = {}
            version = max(self._subjects[subject].keys(), default=0) + 1

            # Create metadata
            schema.metadata = SchemaMetadata(
                schema_id=schema_id,
                subject=subject,
                version=version,
                schema_type=schema.schema_type,
                fingerprint=fingerprint,
            )

            # Store
            self._schemas[schema_id] = schema
            self._subjects[subject][version] = schema_id

            logger.info(f"Registered schema {subject} v{version} (ID: {schema_id})")
            return schema_id

    async def _get_all_schemas(self, subject: str) -> List[Schema]:
        """Get all schemas for a subject in version order."""
        if subject not in self._subjects:
            return []

        schemas = []
        for version in sorted(self._subjects[subject].keys()):
            schema_id = self._subjects[subject][version]
            if schema_id in self._schemas:
                schemas.append(self._schemas[schema_id])
        return schemas

    async def get_schema(self, schema_id: int) -> Optional[Schema]:
        async with self._get_lock():
            return self._schemas.get(schema_id)

    async def get_latest_schema(self, subject: str) -> Optional[Schema]:
        async with self._get_lock():
            if subject not in self._subjects or not self._subjects[subject]:
                return None
            latest_version = max(self._subjects[subject].keys())
            schema_id = self._subjects[subject][latest_version]
            return self._schemas.get(schema_id)

    async def get_schema_by_version(
        self,
        subject: str,
        version: int,
    ) -> Optional[Schema]:
        async with self._get_lock():
            if subject not in self._subjects:
                return None
            schema_id = self._subjects[subject].get(version)
            if schema_id is None:
                return None
            return self._schemas.get(schema_id)

    async def get_versions(self, subject: str) -> List[int]:
        async with self._get_lock():
            if subject not in self._subjects:
                return []
            return sorted(self._subjects[subject].keys())

    async def get_subjects(self) -> List[str]:
        async with self._get_lock():
            return list(self._subjects.keys())

    async def delete_schema(self, subject: str, version: int) -> bool:
        async with self._get_lock():
            if subject not in self._subjects:
                return False
            if version not in self._subjects[subject]:
                return False

            schema_id = self._subjects[subject].pop(version)
            # Don't delete from _schemas as other subjects might reference it
            logger.info(f"Deleted schema {subject} v{version}")
            return True

    async def delete_subject(self, subject: str) -> List[int]:
        async with self._get_lock():
            if subject not in self._subjects:
                return []

            versions = sorted(self._subjects[subject].keys())
            del self._subjects[subject]
            if subject in self._subject_configs:
                del self._subject_configs[subject]

            logger.info(f"Deleted subject {subject} ({len(versions)} versions)")
            return versions

    async def check_compatibility(
        self,
        subject: str,
        schema: Schema,
    ) -> CompatibilityResult:
        async with self._get_lock():
            existing_schemas = await self._get_all_schemas(subject)
            if not existing_schemas:
                return CompatibilityResult(compatible=True)

            mode = self._get_compatibility_unlocked(subject)
            if mode == CompatibilityMode.NONE:
                return CompatibilityResult(compatible=True)

            checker = get_compatibility_checker(schema.schema_type)
            return checker.check(schema, existing_schemas, mode)

    async def get_compatibility(self, subject: str) -> CompatibilityMode:
        async with self._get_lock():
            if subject in self._subject_configs:
                return self._subject_configs[subject].compatibility_mode
            return self.default_compatibility

    async def set_compatibility(
        self,
        subject: str,
        mode: CompatibilityMode,
    ) -> None:
        async with self._get_lock():
            if subject not in self._subject_configs:
                self._subject_configs[subject] = SubjectConfig()
            self._subject_configs[subject].compatibility_mode = mode
            logger.info(f"Set compatibility for {subject} to {mode.value}")


class SchemaCache:
    """Cache for schema lookups."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[int, Schema] = {}
        self._subject_cache: Dict[str, Dict[int, Schema]] = {}

    def get(self, schema_id: int) -> Optional[Schema]:
        return self._cache.get(schema_id)

    def put(self, schema: Schema) -> None:
        if schema.metadata:
            self._cache[schema.metadata.schema_id] = schema

            # Also cache by subject/version
            subject = schema.metadata.subject
            version = schema.metadata.version
            if subject not in self._subject_cache:
                self._subject_cache[subject] = {}
            self._subject_cache[subject][version] = schema

        # Evict if too large
        if len(self._cache) > self.max_size:
            # Remove oldest entries
            to_remove = list(self._cache.keys())[:self.max_size // 4]
            for key in to_remove:
                del self._cache[key]

    def get_by_subject_version(
        self,
        subject: str,
        version: int,
    ) -> Optional[Schema]:
        if subject in self._subject_cache:
            return self._subject_cache[subject].get(version)
        return None

    def invalidate(self, schema_id: Optional[int] = None, subject: Optional[str] = None) -> None:
        if schema_id:
            self._cache.pop(schema_id, None)
        if subject:
            self._subject_cache.pop(subject, None)

    def clear(self) -> None:
        self._cache.clear()
        self._subject_cache.clear()


class CachingSchemaRegistry(SchemaRegistry):
    """Schema registry wrapper with caching."""

    def __init__(
        self,
        delegate: SchemaRegistry,
        cache_size: int = 1000,
    ):
        self.delegate = delegate
        self.cache = SchemaCache(cache_size)

    async def register_schema(
        self,
        subject: str,
        schema: Schema,
        normalize: bool = True,
    ) -> int:
        schema_id = await self.delegate.register_schema(subject, schema, normalize)
        # Refresh cache
        registered = await self.delegate.get_schema(schema_id)
        if registered:
            self.cache.put(registered)
        return schema_id

    async def get_schema(self, schema_id: int) -> Optional[Schema]:
        # Check cache
        cached = self.cache.get(schema_id)
        if cached:
            return cached

        # Fetch from delegate
        schema = await self.delegate.get_schema(schema_id)
        if schema:
            self.cache.put(schema)
        return schema

    async def get_latest_schema(self, subject: str) -> Optional[Schema]:
        schema = await self.delegate.get_latest_schema(subject)
        if schema:
            self.cache.put(schema)
        return schema

    async def get_schema_by_version(
        self,
        subject: str,
        version: int,
    ) -> Optional[Schema]:
        # Check cache
        cached = self.cache.get_by_subject_version(subject, version)
        if cached:
            return cached

        schema = await self.delegate.get_schema_by_version(subject, version)
        if schema:
            self.cache.put(schema)
        return schema

    async def get_versions(self, subject: str) -> List[int]:
        return await self.delegate.get_versions(subject)

    async def get_subjects(self) -> List[str]:
        return await self.delegate.get_subjects()

    async def delete_schema(self, subject: str, version: int) -> bool:
        result = await self.delegate.delete_schema(subject, version)
        if result:
            self.cache.invalidate(subject=subject)
        return result

    async def delete_subject(self, subject: str) -> List[int]:
        result = await self.delegate.delete_subject(subject)
        self.cache.invalidate(subject=subject)
        return result

    async def check_compatibility(
        self,
        subject: str,
        schema: Schema,
    ) -> CompatibilityResult:
        return await self.delegate.check_compatibility(subject, schema)

    async def get_compatibility(self, subject: str) -> CompatibilityMode:
        return await self.delegate.get_compatibility(subject)

    async def set_compatibility(
        self,
        subject: str,
        mode: CompatibilityMode,
    ) -> None:
        await self.delegate.set_compatibility(subject, mode)
