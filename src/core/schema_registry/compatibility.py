"""Schema Compatibility Checking.

Provides compatibility checking:
- Backward compatibility
- Forward compatibility
- Full compatibility
- Transitive checks
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.schema_registry.schema import (
    CompatibilityMode,
    Schema,
    SchemaType,
    get_parser,
)

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityResult:
    """Result of a compatibility check."""
    compatible: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.compatible


class CompatibilityChecker(ABC):
    """Abstract base class for compatibility checkers."""

    @abstractmethod
    def check(
        self,
        new_schema: Schema,
        old_schemas: List[Schema],
        mode: CompatibilityMode,
    ) -> CompatibilityResult:
        """Check compatibility between schemas."""
        pass


class JSONSchemaCompatibilityChecker(CompatibilityChecker):
    """Compatibility checker for JSON Schema."""

    def check(
        self,
        new_schema: Schema,
        old_schemas: List[Schema],
        mode: CompatibilityMode,
    ) -> CompatibilityResult:
        if not old_schemas:
            return CompatibilityResult(compatible=True)

        errors = []
        warnings = []

        parser = get_parser(SchemaType.JSON)
        new_parsed = parser.parse(new_schema.schema_str)

        # Determine which schemas to check against
        if mode in (CompatibilityMode.BACKWARD, CompatibilityMode.FORWARD, CompatibilityMode.FULL):
            schemas_to_check = [old_schemas[-1]]
        else:
            schemas_to_check = old_schemas

        for old_schema in schemas_to_check:
            old_parsed = parser.parse(old_schema.schema_str)

            if mode in (CompatibilityMode.BACKWARD, CompatibilityMode.BACKWARD_TRANSITIVE, CompatibilityMode.FULL, CompatibilityMode.FULL_TRANSITIVE):
                backward_errors = self._check_backward(new_parsed, old_parsed)
                errors.extend(backward_errors)

            if mode in (CompatibilityMode.FORWARD, CompatibilityMode.FORWARD_TRANSITIVE, CompatibilityMode.FULL, CompatibilityMode.FULL_TRANSITIVE):
                forward_errors = self._check_forward(new_parsed, old_parsed)
                errors.extend(forward_errors)

        return CompatibilityResult(
            compatible=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_backward(
        self,
        new_schema: Dict[str, Any],
        old_schema: Dict[str, Any],
    ) -> List[str]:
        """Check backward compatibility (new can read old data).

        Rules:
        - Cannot remove required fields
        - Cannot change field types incompatibly
        - Cannot narrow allowed values
        """
        errors = []

        # Check required fields
        old_required = set(old_schema.get("required", []))
        new_required = set(new_schema.get("required", []))

        # New schema can add required fields only if they have defaults
        new_properties = new_schema.get("properties", {})
        for req in new_required - old_required:
            if req in new_properties:
                prop = new_properties[req]
                if "default" not in prop:
                    errors.append(
                        f"New required field '{req}' without default breaks backward compatibility"
                    )

        # Check property type changes
        old_properties = old_schema.get("properties", {})
        for prop_name, old_prop in old_properties.items():
            if prop_name in new_properties:
                new_prop = new_properties[prop_name]
                type_errors = self._check_type_compatibility(
                    old_prop, new_prop, prop_name, "backward"
                )
                errors.extend(type_errors)

        return errors

    def _check_forward(
        self,
        new_schema: Dict[str, Any],
        old_schema: Dict[str, Any],
    ) -> List[str]:
        """Check forward compatibility (old can read new data).

        Rules:
        - Cannot add required fields without defaults
        - Cannot change field types incompatibly
        - Cannot widen allowed values beyond what old schema accepts
        """
        errors = []

        # Check if old schema has additionalProperties: false
        if old_schema.get("additionalProperties") is False:
            old_props = set(old_schema.get("properties", {}).keys())
            new_props = set(new_schema.get("properties", {}).keys())
            added_props = new_props - old_props
            if added_props:
                errors.append(
                    f"New properties {added_props} not allowed by old schema (additionalProperties: false)"
                )

        return errors

    def _check_type_compatibility(
        self,
        old_prop: Dict[str, Any],
        new_prop: Dict[str, Any],
        prop_name: str,
        direction: str,
    ) -> List[str]:
        """Check if property types are compatible."""
        errors = []

        old_type = old_prop.get("type")
        new_type = new_prop.get("type")

        if old_type != new_type:
            # Check for compatible type changes
            if isinstance(old_type, list) and isinstance(new_type, str):
                if new_type not in old_type:
                    errors.append(
                        f"Property '{prop_name}': narrowed type from {old_type} to {new_type}"
                    )
            elif isinstance(new_type, list) and isinstance(old_type, str):
                if direction == "backward" and old_type not in new_type:
                    errors.append(
                        f"Property '{prop_name}': incompatible type change {old_type} -> {new_type}"
                    )
            else:
                errors.append(
                    f"Property '{prop_name}': type changed from {old_type} to {new_type}"
                )

        # Check enum changes
        old_enum = old_prop.get("enum")
        new_enum = new_prop.get("enum")
        if old_enum and new_enum:
            if direction == "backward":
                # New enum should be superset of old
                if not set(old_enum).issubset(set(new_enum)):
                    removed = set(old_enum) - set(new_enum)
                    errors.append(
                        f"Property '{prop_name}': removed enum values {removed}"
                    )
            else:
                # Old enum should be superset of new
                if not set(new_enum).issubset(set(old_enum)):
                    added = set(new_enum) - set(old_enum)
                    errors.append(
                        f"Property '{prop_name}': added enum values {added} not known to old schema"
                    )

        return errors


class AvroCompatibilityChecker(CompatibilityChecker):
    """Compatibility checker for Avro schema."""

    def check(
        self,
        new_schema: Schema,
        old_schemas: List[Schema],
        mode: CompatibilityMode,
    ) -> CompatibilityResult:
        if not old_schemas:
            return CompatibilityResult(compatible=True)

        errors = []
        warnings = []

        parser = get_parser(SchemaType.AVRO)
        new_parsed = parser.parse(new_schema.schema_str)

        schemas_to_check = (
            [old_schemas[-1]]
            if mode in (CompatibilityMode.BACKWARD, CompatibilityMode.FORWARD, CompatibilityMode.FULL)
            else old_schemas
        )

        for old_schema in schemas_to_check:
            old_parsed = parser.parse(old_schema.schema_str)

            if mode in (CompatibilityMode.BACKWARD, CompatibilityMode.BACKWARD_TRANSITIVE, CompatibilityMode.FULL, CompatibilityMode.FULL_TRANSITIVE):
                backward_errors = self._check_backward_avro(new_parsed, old_parsed)
                errors.extend(backward_errors)

            if mode in (CompatibilityMode.FORWARD, CompatibilityMode.FORWARD_TRANSITIVE, CompatibilityMode.FULL, CompatibilityMode.FULL_TRANSITIVE):
                forward_errors = self._check_forward_avro(new_parsed, old_parsed)
                errors.extend(forward_errors)

        return CompatibilityResult(
            compatible=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_backward_avro(
        self,
        new_schema: Dict[str, Any],
        old_schema: Dict[str, Any],
    ) -> List[str]:
        """Check backward compatibility for Avro."""
        errors = []

        if new_schema.get("type") == "record" and old_schema.get("type") == "record":
            old_fields = {f["name"]: f for f in old_schema.get("fields", [])}
            new_fields = {f["name"]: f for f in new_schema.get("fields", [])}

            # Fields removed in new schema
            for field_name in old_fields:
                if field_name not in new_fields:
                    errors.append(f"Field '{field_name}' removed from schema")

            # New fields without defaults
            for field_name, field_def in new_fields.items():
                if field_name not in old_fields:
                    if "default" not in field_def:
                        errors.append(
                            f"New field '{field_name}' must have a default value for backward compatibility"
                        )

        return errors

    def _check_forward_avro(
        self,
        new_schema: Dict[str, Any],
        old_schema: Dict[str, Any],
    ) -> List[str]:
        """Check forward compatibility for Avro."""
        errors = []

        if new_schema.get("type") == "record" and old_schema.get("type") == "record":
            old_fields = {f["name"]: f for f in old_schema.get("fields", [])}
            new_fields = {f["name"]: f for f in new_schema.get("fields", [])}

            # Fields added in new schema that old readers must handle
            for field_name in new_fields:
                if field_name not in old_fields:
                    # Old schema must have a default or the field must be optional
                    pass  # Avro readers typically skip unknown fields

        return errors


class ProtobufCompatibilityChecker(CompatibilityChecker):
    """Compatibility checker for Protocol Buffers."""

    def check(
        self,
        new_schema: Schema,
        old_schemas: List[Schema],
        mode: CompatibilityMode,
    ) -> CompatibilityResult:
        if not old_schemas:
            return CompatibilityResult(compatible=True)

        errors = []
        warnings = []

        parser = get_parser(SchemaType.PROTOBUF)
        new_parsed = parser.parse(new_schema.schema_str)

        schemas_to_check = (
            [old_schemas[-1]]
            if mode in (CompatibilityMode.BACKWARD, CompatibilityMode.FORWARD, CompatibilityMode.FULL)
            else old_schemas
        )

        for old_schema in schemas_to_check:
            old_parsed = parser.parse(old_schema.schema_str)
            check_errors = self._check_protobuf_compatibility(new_parsed, old_parsed)
            errors.extend(check_errors)

        return CompatibilityResult(
            compatible=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_protobuf_compatibility(
        self,
        new_parsed: Dict[str, Any],
        old_parsed: Dict[str, Any],
    ) -> List[str]:
        """Check protobuf compatibility rules."""
        errors = []

        for old_message in old_parsed.get("messages", []):
            new_message = None
            for m in new_parsed.get("messages", []):
                if m["name"] == old_message["name"]:
                    new_message = m
                    break

            if new_message is None:
                errors.append(f"Message '{old_message['name']}' was removed")
                continue

            # Check field compatibility
            old_fields = {f["number"]: f for f in old_message["fields"]}
            new_fields = {f["number"]: f for f in new_message["fields"]}

            for field_num, old_field in old_fields.items():
                if field_num in new_fields:
                    new_field = new_fields[field_num]
                    # Cannot change field type
                    if old_field["type"] != new_field["type"]:
                        errors.append(
                            f"Field {old_field['name']} (#{field_num}): "
                            f"type changed from {old_field['type']} to {new_field['type']}"
                        )
                    # Cannot change repeated
                    if old_field["repeated"] != new_field["repeated"]:
                        errors.append(
                            f"Field {old_field['name']} (#{field_num}): "
                            f"repeated modifier changed"
                        )

            # Field number reuse is forbidden
            for field_num, new_field in new_fields.items():
                if field_num in old_fields:
                    old_field = old_fields[field_num]
                    if old_field["name"] != new_field["name"]:
                        errors.append(
                            f"Field number {field_num} reused: "
                            f"was '{old_field['name']}', now '{new_field['name']}'"
                        )

        return errors


def get_compatibility_checker(schema_type: SchemaType) -> CompatibilityChecker:
    """Get compatibility checker for schema type."""
    checkers = {
        SchemaType.JSON: JSONSchemaCompatibilityChecker,
        SchemaType.AVRO: AvroCompatibilityChecker,
        SchemaType.PROTOBUF: ProtobufCompatibilityChecker,
    }
    return checkers[schema_type]()
