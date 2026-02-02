"""Schema Definition and Types.

Provides schema type definitions:
- JSON Schema support
- Avro schema support
- Protobuf schema support
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type, Union


class SchemaType(Enum):
    """Supported schema types."""
    JSON = "json"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class CompatibilityMode(Enum):
    """Schema compatibility modes."""
    NONE = "none"               # No compatibility checks
    BACKWARD = "backward"       # New schema can read old data
    FORWARD = "forward"         # Old schema can read new data
    FULL = "full"              # Both backward and forward
    BACKWARD_TRANSITIVE = "backward_transitive"  # Backward with all versions
    FORWARD_TRANSITIVE = "forward_transitive"    # Forward with all versions
    FULL_TRANSITIVE = "full_transitive"          # Full with all versions


@dataclass
class SchemaMetadata:
    """Metadata for a schema version."""
    schema_id: int
    subject: str
    version: int
    schema_type: SchemaType
    fingerprint: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    deprecated: bool = False
    tags: Set[str] = field(default_factory=set)
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class Schema:
    """A schema definition."""
    schema_type: SchemaType
    schema_str: str
    metadata: Optional[SchemaMetadata] = None
    references: List["SchemaReference"] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        """Get schema fingerprint (hash)."""
        return hashlib.sha256(self.schema_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_type": self.schema_type.value,
            "schema": self.schema_str,
            "fingerprint": self.fingerprint,
            "metadata": {
                "schema_id": self.metadata.schema_id,
                "subject": self.metadata.subject,
                "version": self.metadata.version,
            } if self.metadata else None,
            "references": [r.to_dict() for r in self.references],
        }


@dataclass
class SchemaReference:
    """Reference to another schema."""
    name: str
    subject: str
    version: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "subject": self.subject,
            "version": self.version,
        }


class SchemaParser(ABC):
    """Abstract base class for schema parsers."""

    @abstractmethod
    def parse(self, schema_str: str) -> Dict[str, Any]:
        """Parse schema string to internal representation."""
        pass

    @abstractmethod
    def validate(self, schema_str: str) -> List[str]:
        """Validate schema syntax. Returns list of errors."""
        pass

    @abstractmethod
    def normalize(self, schema_str: str) -> str:
        """Normalize schema to canonical form."""
        pass


class JSONSchemaParser(SchemaParser):
    """Parser for JSON Schema."""

    SUPPORTED_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}

    def parse(self, schema_str: str) -> Dict[str, Any]:
        return json.loads(schema_str)

    def validate(self, schema_str: str) -> List[str]:
        errors = []
        try:
            schema = json.loads(schema_str)

            # Basic validation
            if not isinstance(schema, dict):
                errors.append("Schema must be an object")
                return errors

            # Validate type if present
            if "type" in schema:
                if isinstance(schema["type"], str):
                    if schema["type"] not in self.SUPPORTED_TYPES:
                        errors.append(f"Unsupported type: {schema['type']}")
                elif isinstance(schema["type"], list):
                    for t in schema["type"]:
                        if t not in self.SUPPORTED_TYPES:
                            errors.append(f"Unsupported type: {t}")

            # Validate properties
            if "properties" in schema:
                if not isinstance(schema["properties"], dict):
                    errors.append("'properties' must be an object")

            # Validate required
            if "required" in schema:
                if not isinstance(schema["required"], list):
                    errors.append("'required' must be an array")
                elif "properties" in schema:
                    for req in schema["required"]:
                        if req not in schema.get("properties", {}):
                            errors.append(f"Required property '{req}' not in properties")

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")

        return errors

    def normalize(self, schema_str: str) -> str:
        """Normalize JSON schema to canonical form."""
        schema = json.loads(schema_str)
        return json.dumps(schema, sort_keys=True, separators=(',', ':'))


class AvroSchemaParser(SchemaParser):
    """Parser for Avro schema."""

    PRIMITIVE_TYPES = {"null", "boolean", "int", "long", "float", "double", "bytes", "string"}

    def parse(self, schema_str: str) -> Dict[str, Any]:
        return json.loads(schema_str)

    def validate(self, schema_str: str) -> List[str]:
        errors = []
        try:
            schema = json.loads(schema_str)
            errors.extend(self._validate_type(schema, "root"))
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
        return errors

    def _validate_type(self, schema: Any, path: str) -> List[str]:
        errors = []

        if isinstance(schema, str):
            # Primitive type or named type reference
            if schema not in self.PRIMITIVE_TYPES:
                # Could be a named type reference - allow it
                pass
        elif isinstance(schema, list):
            # Union type
            for i, item in enumerate(schema):
                errors.extend(self._validate_type(item, f"{path}[{i}]"))
        elif isinstance(schema, dict):
            schema_type = schema.get("type")
            if schema_type == "record":
                if "name" not in schema:
                    errors.append(f"{path}: record type requires 'name'")
                if "fields" not in schema:
                    errors.append(f"{path}: record type requires 'fields'")
                elif isinstance(schema["fields"], list):
                    for i, field_def in enumerate(schema["fields"]):
                        if "name" not in field_def:
                            errors.append(f"{path}.fields[{i}]: field requires 'name'")
                        if "type" not in field_def:
                            errors.append(f"{path}.fields[{i}]: field requires 'type'")
                        else:
                            errors.extend(self._validate_type(
                                field_def["type"],
                                f"{path}.fields[{i}].type"
                            ))
            elif schema_type == "array":
                if "items" not in schema:
                    errors.append(f"{path}: array type requires 'items'")
                else:
                    errors.extend(self._validate_type(schema["items"], f"{path}.items"))
            elif schema_type == "map":
                if "values" not in schema:
                    errors.append(f"{path}: map type requires 'values'")
                else:
                    errors.extend(self._validate_type(schema["values"], f"{path}.values"))
            elif schema_type == "enum":
                if "name" not in schema:
                    errors.append(f"{path}: enum type requires 'name'")
                if "symbols" not in schema:
                    errors.append(f"{path}: enum type requires 'symbols'")
            elif schema_type == "fixed":
                if "name" not in schema:
                    errors.append(f"{path}: fixed type requires 'name'")
                if "size" not in schema:
                    errors.append(f"{path}: fixed type requires 'size'")
            elif schema_type in self.PRIMITIVE_TYPES:
                pass
            elif schema_type is None:
                errors.append(f"{path}: schema object requires 'type'")
            else:
                errors.append(f"{path}: unknown type '{schema_type}'")
        else:
            errors.append(f"{path}: invalid schema type")

        return errors

    def normalize(self, schema_str: str) -> str:
        schema = json.loads(schema_str)
        return json.dumps(schema, sort_keys=True, separators=(',', ':'))


class ProtobufSchemaParser(SchemaParser):
    """Parser for Protocol Buffers schema."""

    def parse(self, schema_str: str) -> Dict[str, Any]:
        """Parse protobuf schema to internal representation."""
        # Simple parser - in production use protobuf compiler
        result = {
            "syntax": "proto3",
            "messages": [],
            "enums": [],
        }

        # Extract syntax
        syntax_match = re.search(r'syntax\s*=\s*"(proto2|proto3)"\s*;', schema_str)
        if syntax_match:
            result["syntax"] = syntax_match.group(1)

        # Extract messages
        message_pattern = r'message\s+(\w+)\s*\{([^}]*)\}'
        for match in re.finditer(message_pattern, schema_str):
            message_name = match.group(1)
            message_body = match.group(2)
            fields = self._parse_fields(message_body)
            result["messages"].append({
                "name": message_name,
                "fields": fields,
            })

        # Extract enums
        enum_pattern = r'enum\s+(\w+)\s*\{([^}]*)\}'
        for match in re.finditer(enum_pattern, schema_str):
            enum_name = match.group(1)
            enum_body = match.group(2)
            values = self._parse_enum_values(enum_body)
            result["enums"].append({
                "name": enum_name,
                "values": values,
            })

        return result

    def _parse_fields(self, body: str) -> List[Dict[str, Any]]:
        """Parse message fields."""
        fields = []
        field_pattern = r'(repeated\s+)?(\w+)\s+(\w+)\s*=\s*(\d+)\s*;'
        for match in re.finditer(field_pattern, body):
            repeated = match.group(1) is not None
            field_type = match.group(2)
            field_name = match.group(3)
            field_number = int(match.group(4))
            fields.append({
                "name": field_name,
                "type": field_type,
                "number": field_number,
                "repeated": repeated,
            })
        return fields

    def _parse_enum_values(self, body: str) -> List[Dict[str, Any]]:
        """Parse enum values."""
        values = []
        value_pattern = r'(\w+)\s*=\s*(\d+)\s*;'
        for match in re.finditer(value_pattern, body):
            value_name = match.group(1)
            value_number = int(match.group(2))
            values.append({
                "name": value_name,
                "number": value_number,
            })
        return values

    def validate(self, schema_str: str) -> List[str]:
        errors = []
        try:
            parsed = self.parse(schema_str)

            # Validate messages have unique names
            message_names = [m["name"] for m in parsed["messages"]]
            if len(message_names) != len(set(message_names)):
                errors.append("Duplicate message names found")

            # Validate field numbers are unique within each message
            for message in parsed["messages"]:
                field_numbers = [f["number"] for f in message["fields"]]
                if len(field_numbers) != len(set(field_numbers)):
                    errors.append(f"Duplicate field numbers in message {message['name']}")

        except Exception as e:
            errors.append(f"Parse error: {e}")

        return errors

    def normalize(self, schema_str: str) -> str:
        # For protobuf, we return a canonical representation
        # In production, use protobuf compiler for proper normalization
        return re.sub(r'\s+', ' ', schema_str.strip())


def get_parser(schema_type: SchemaType) -> SchemaParser:
    """Get parser for schema type."""
    parsers = {
        SchemaType.JSON: JSONSchemaParser,
        SchemaType.AVRO: AvroSchemaParser,
        SchemaType.PROTOBUF: ProtobufSchemaParser,
    }
    return parsers[schema_type]()
