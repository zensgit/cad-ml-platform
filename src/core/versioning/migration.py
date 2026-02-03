"""API Migration Support.

Provides tools for API migration:
- Request/response transformation
- Schema migration
- Deprecation warnings
- Migration guides
"""

from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from src.core.versioning.version import SemanticVersion

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MigrationStep:
    """A single migration step between versions."""
    from_version: SemanticVersion
    to_version: SemanticVersion
    description: str
    breaking: bool = False
    transform_request: Optional[Callable[[Dict], Dict]] = None
    transform_response: Optional[Callable[[Dict], Dict]] = None


class SchemaMigrator:
    """Handles schema migration between API versions."""

    def __init__(self):
        self._migrations: Dict[str, List[MigrationStep]] = {}  # schema_name -> steps

    def register_migration(
        self,
        schema_name: str,
        from_version: Union[str, SemanticVersion],
        to_version: Union[str, SemanticVersion],
        description: str,
        breaking: bool = False,
        transform_request: Optional[Callable[[Dict], Dict]] = None,
        transform_response: Optional[Callable[[Dict], Dict]] = None,
    ) -> MigrationStep:
        """Register a migration step."""
        if isinstance(from_version, str):
            from_version = SemanticVersion.parse(from_version)
        if isinstance(to_version, str):
            to_version = SemanticVersion.parse(to_version)

        step = MigrationStep(
            from_version=from_version,
            to_version=to_version,
            description=description,
            breaking=breaking,
            transform_request=transform_request,
            transform_response=transform_response,
        )

        if schema_name not in self._migrations:
            self._migrations[schema_name] = []

        self._migrations[schema_name].append(step)
        # Sort by from_version
        self._migrations[schema_name].sort(key=lambda s: s.from_version)

        return step

    def migrate_request(
        self,
        schema_name: str,
        data: Dict,
        from_version: SemanticVersion,
        to_version: SemanticVersion,
    ) -> Dict:
        """Migrate request data from one version to another."""
        if from_version >= to_version:
            return data

        steps = self._get_migration_path(schema_name, from_version, to_version)
        result = copy.deepcopy(data)

        for step in steps:
            if step.transform_request:
                result = step.transform_request(result)
                logger.debug(
                    f"Applied request migration {schema_name}: "
                    f"v{step.from_version} -> v{step.to_version}"
                )

        return result

    def migrate_response(
        self,
        schema_name: str,
        data: Dict,
        from_version: SemanticVersion,
        to_version: SemanticVersion,
    ) -> Dict:
        """Migrate response data (downgrade for older clients)."""
        if from_version <= to_version:
            return data

        # Get reverse path for response migration
        steps = self._get_migration_path(schema_name, to_version, from_version)
        steps = list(reversed(steps))

        result = copy.deepcopy(data)

        for step in steps:
            if step.transform_response:
                result = step.transform_response(result)
                logger.debug(
                    f"Applied response migration {schema_name}: "
                    f"v{step.to_version} -> v{step.from_version}"
                )

        return result

    def _get_migration_path(
        self,
        schema_name: str,
        from_version: SemanticVersion,
        to_version: SemanticVersion,
    ) -> List[MigrationStep]:
        """Get the sequence of migrations to apply."""
        if schema_name not in self._migrations:
            return []

        path = []
        current = from_version

        for step in self._migrations[schema_name]:
            if step.from_version >= current and step.to_version <= to_version:
                path.append(step)
                current = step.to_version

        return path

    def get_breaking_changes(
        self,
        schema_name: str,
        from_version: SemanticVersion,
        to_version: SemanticVersion,
    ) -> List[MigrationStep]:
        """Get all breaking changes in a version range."""
        path = self._get_migration_path(schema_name, from_version, to_version)
        return [step for step in path if step.breaking]


@dataclass
class FieldChange:
    """Represents a field change in schema migration."""
    change_type: str  # "added", "removed", "renamed", "type_changed"
    field_name: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    description: str = ""


class SchemaTransformBuilder:
    """Builder for creating schema transformation functions."""

    def __init__(self):
        self._request_transforms: List[Callable[[Dict], Dict]] = []
        self._response_transforms: List[Callable[[Dict], Dict]] = []

    def rename_field(
        self,
        old_name: str,
        new_name: str,
        in_request: bool = True,
        in_response: bool = True,
    ) -> "SchemaTransformBuilder":
        """Rename a field."""
        def transform(data: Dict) -> Dict:
            if old_name in data:
                data[new_name] = data.pop(old_name)
            return data

        def reverse_transform(data: Dict) -> Dict:
            if new_name in data:
                data[old_name] = data.pop(new_name)
            return data

        if in_request:
            self._request_transforms.append(transform)
        if in_response:
            self._response_transforms.append(reverse_transform)

        return self

    def add_field(
        self,
        field_name: str,
        default_value: Any,
        in_request: bool = True,
    ) -> "SchemaTransformBuilder":
        """Add a new field with default value."""
        def transform(data: Dict) -> Dict:
            if field_name not in data:
                data[field_name] = default_value
            return data

        if in_request:
            self._request_transforms.append(transform)

        return self

    def remove_field(
        self,
        field_name: str,
        in_response: bool = True,
    ) -> "SchemaTransformBuilder":
        """Remove a field (for response downgrade)."""
        def transform(data: Dict) -> Dict:
            data.pop(field_name, None)
            return data

        if in_response:
            self._response_transforms.append(transform)

        return self

    def transform_field(
        self,
        field_name: str,
        request_transform: Optional[Callable[[Any], Any]] = None,
        response_transform: Optional[Callable[[Any], Any]] = None,
    ) -> "SchemaTransformBuilder":
        """Transform a field value."""
        if request_transform:
            def req_transform(data: Dict) -> Dict:
                if field_name in data:
                    data[field_name] = request_transform(data[field_name])
                return data
            self._request_transforms.append(req_transform)

        if response_transform:
            def res_transform(data: Dict) -> Dict:
                if field_name in data:
                    data[field_name] = response_transform(data[field_name])
                return data
            self._response_transforms.append(res_transform)

        return self

    def nest_fields(
        self,
        parent_field: str,
        child_fields: List[str],
    ) -> "SchemaTransformBuilder":
        """Nest flat fields under a parent object."""
        def transform(data: Dict) -> Dict:
            nested = {}
            for field_name in child_fields:
                if field_name in data:
                    nested[field_name] = data.pop(field_name)
            if nested:
                data[parent_field] = nested
            return data

        def reverse_transform(data: Dict) -> Dict:
            if parent_field in data and isinstance(data[parent_field], dict):
                nested = data.pop(parent_field)
                data.update(nested)
            return data

        self._request_transforms.append(transform)
        self._response_transforms.append(reverse_transform)

        return self

    def flatten_fields(
        self,
        parent_field: str,
        prefix: str = "",
    ) -> "SchemaTransformBuilder":
        """Flatten nested object to flat fields."""
        def transform(data: Dict) -> Dict:
            if parent_field in data and isinstance(data[parent_field], dict):
                nested = data.pop(parent_field)
                for key, value in nested.items():
                    data[f"{prefix}{key}"] = value
            return data

        self._request_transforms.append(transform)

        return self

    def build(self) -> tuple[Callable[[Dict], Dict], Callable[[Dict], Dict]]:
        """Build the transformation functions."""
        def request_transform(data: Dict) -> Dict:
            result = copy.deepcopy(data)
            for transform in self._request_transforms:
                result = transform(result)
            return result

        def response_transform(data: Dict) -> Dict:
            result = copy.deepcopy(data)
            for transform in self._response_transforms:
                result = transform(result)
            return result

        return request_transform, response_transform


@dataclass
class DeprecationNotice:
    """Deprecation notice for API elements."""
    element_type: str  # "endpoint", "field", "parameter"
    element_name: str
    deprecated_in: SemanticVersion
    removed_in: Optional[SemanticVersion] = None
    replacement: Optional[str] = None
    message: str = ""
    documentation_url: Optional[str] = None


class DeprecationRegistry:
    """Registry for deprecation notices."""

    def __init__(self):
        self._notices: List[DeprecationNotice] = []

    def deprecate_endpoint(
        self,
        endpoint: str,
        deprecated_in: Union[str, SemanticVersion],
        removed_in: Optional[Union[str, SemanticVersion]] = None,
        replacement: Optional[str] = None,
        message: str = "",
    ) -> DeprecationNotice:
        """Register endpoint deprecation."""
        if isinstance(deprecated_in, str):
            deprecated_in = SemanticVersion.parse(deprecated_in)
        if isinstance(removed_in, str):
            removed_in = SemanticVersion.parse(removed_in)

        notice = DeprecationNotice(
            element_type="endpoint",
            element_name=endpoint,
            deprecated_in=deprecated_in,
            removed_in=removed_in,
            replacement=replacement,
            message=message,
        )
        self._notices.append(notice)
        return notice

    def deprecate_field(
        self,
        field: str,
        deprecated_in: Union[str, SemanticVersion],
        removed_in: Optional[Union[str, SemanticVersion]] = None,
        replacement: Optional[str] = None,
        message: str = "",
    ) -> DeprecationNotice:
        """Register field deprecation."""
        if isinstance(deprecated_in, str):
            deprecated_in = SemanticVersion.parse(deprecated_in)
        if isinstance(removed_in, str):
            removed_in = SemanticVersion.parse(removed_in)

        notice = DeprecationNotice(
            element_type="field",
            element_name=field,
            deprecated_in=deprecated_in,
            removed_in=removed_in,
            replacement=replacement,
            message=message,
        )
        self._notices.append(notice)
        return notice

    def get_notices_for_version(
        self,
        version: SemanticVersion,
    ) -> List[DeprecationNotice]:
        """Get all deprecation notices affecting a version."""
        return [
            n for n in self._notices
            if n.deprecated_in <= version and (
                n.removed_in is None or n.removed_in > version
            )
        ]

    def get_removed_in_version(
        self,
        version: SemanticVersion,
    ) -> List[DeprecationNotice]:
        """Get elements removed in a specific version."""
        return [
            n for n in self._notices
            if n.removed_in and n.removed_in == version
        ]

    def format_warnings(
        self,
        version: SemanticVersion,
    ) -> List[str]:
        """Format deprecation warnings for a version."""
        warnings = []
        for notice in self.get_notices_for_version(version):
            msg = f"{notice.element_type.capitalize()} '{notice.element_name}' is deprecated"
            if notice.removed_in:
                msg += f" and will be removed in v{notice.removed_in}"
            if notice.replacement:
                msg += f". Use '{notice.replacement}' instead"
            if notice.message:
                msg += f". {notice.message}"
            warnings.append(msg)
        return warnings


@dataclass
class MigrationGuide:
    """Migration guide between versions."""
    from_version: SemanticVersion
    to_version: SemanticVersion
    title: str
    summary: str
    breaking_changes: List[str] = field(default_factory=list)
    new_features: List[str] = field(default_factory=list)
    deprecations: List[str] = field(default_factory=list)
    migration_steps: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown documentation."""
        lines = [
            f"# {self.title}",
            "",
            f"**From:** v{self.from_version}",
            f"**To:** v{self.to_version}",
            "",
            "## Summary",
            "",
            self.summary,
            "",
        ]

        if self.breaking_changes:
            lines.extend([
                "## Breaking Changes",
                "",
            ])
            for change in self.breaking_changes:
                lines.append(f"- ⚠️ {change}")
            lines.append("")

        if self.new_features:
            lines.extend([
                "## New Features",
                "",
            ])
            for feature in self.new_features:
                lines.append(f"- ✨ {feature}")
            lines.append("")

        if self.deprecations:
            lines.extend([
                "## Deprecations",
                "",
            ])
            for deprecation in self.deprecations:
                lines.append(f"- ⏰ {deprecation}")
            lines.append("")

        if self.migration_steps:
            lines.extend([
                "## Migration Steps",
                "",
            ])
            for i, step in enumerate(self.migration_steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")

        return "\n".join(lines)


class MigrationGuideBuilder:
    """Builder for creating migration guides."""

    def __init__(
        self,
        from_version: Union[str, SemanticVersion],
        to_version: Union[str, SemanticVersion],
    ):
        if isinstance(from_version, str):
            from_version = SemanticVersion.parse(from_version)
        if isinstance(to_version, str):
            to_version = SemanticVersion.parse(to_version)

        self._from_version = from_version
        self._to_version = to_version
        self._title = f"Migration Guide: v{from_version} to v{to_version}"
        self._summary = ""
        self._breaking_changes: List[str] = []
        self._new_features: List[str] = []
        self._deprecations: List[str] = []
        self._migration_steps: List[str] = []

    def title(self, title: str) -> "MigrationGuideBuilder":
        """Set guide title."""
        self._title = title
        return self

    def summary(self, summary: str) -> "MigrationGuideBuilder":
        """Set guide summary."""
        self._summary = summary
        return self

    def breaking_change(self, change: str) -> "MigrationGuideBuilder":
        """Add a breaking change."""
        self._breaking_changes.append(change)
        return self

    def new_feature(self, feature: str) -> "MigrationGuideBuilder":
        """Add a new feature."""
        self._new_features.append(feature)
        return self

    def deprecation(self, deprecation: str) -> "MigrationGuideBuilder":
        """Add a deprecation notice."""
        self._deprecations.append(deprecation)
        return self

    def migration_step(self, step: str) -> "MigrationGuideBuilder":
        """Add a migration step."""
        self._migration_steps.append(step)
        return self

    def build(self) -> MigrationGuide:
        """Build the migration guide."""
        return MigrationGuide(
            from_version=self._from_version,
            to_version=self._to_version,
            title=self._title,
            summary=self._summary,
            breaking_changes=self._breaking_changes,
            new_features=self._new_features,
            deprecations=self._deprecations,
            migration_steps=self._migration_steps,
        )
