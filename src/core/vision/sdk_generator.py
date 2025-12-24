"""SDK Generator Module for Vision System.

This module provides client SDK generation capabilities including:
- Multi-language SDK generation (Python, TypeScript, Java, Go)
- OpenAPI/Swagger specification generation
- GraphQL schema generation
- Code templates and client libraries
- API documentation generation
- Type definitions and interfaces

Phase 16: Advanced Integration & Extensibility
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import VisionDescription, VisionProvider

# ========================
# Enums
# ========================


class SDKLanguage(str, Enum):
    """Supported SDK languages."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    GO = "go"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    RUST = "rust"
    SWIFT = "swift"
    KOTLIN = "kotlin"


class SpecFormat(str, Enum):
    """API specification formats."""

    OPENAPI_3_0 = "openapi_3_0"
    OPENAPI_3_1 = "openapi_3_1"
    SWAGGER_2_0 = "swagger_2_0"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    ASYNCAPI = "asyncapi"
    JSON_SCHEMA = "json_schema"


class HTTPMethod(str, Enum):
    """HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ParameterLocation(str, Enum):
    """Parameter location in request."""

    PATH = "path"
    QUERY = "query"
    HEADER = "header"
    COOKIE = "cookie"
    BODY = "body"


class DataType(str, Enum):
    """Data types for schema definitions."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    BINARY = "binary"
    FILE = "file"


# ========================
# Data Classes
# ========================


@dataclass
class SchemaProperty:
    """Schema property definition."""

    name: str
    data_type: DataType
    description: str = ""
    required: bool = False
    default: Any = None
    enum_values: Optional[List[Any]] = None
    format: Optional[str] = None
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    items_type: Optional[DataType] = None  # For arrays
    ref: Optional[str] = None  # Reference to another schema


@dataclass
class SchemaDefinition:
    """Schema definition for API types."""

    name: str
    description: str = ""
    properties: List[SchemaProperty] = field(default_factory=list)
    required_properties: List[str] = field(default_factory=list)
    additional_properties: bool = False
    example: Optional[Dict[str, Any]] = None


@dataclass
class ParameterDefinition:
    """API parameter definition."""

    name: str
    location: ParameterLocation
    data_type: DataType
    description: str = ""
    required: bool = False
    default: Any = None
    schema_ref: Optional[str] = None


@dataclass
class ResponseDefinition:
    """API response definition."""

    status_code: int
    description: str
    content_type: str = "application/json"
    schema_ref: Optional[str] = None
    schema: Optional[SchemaDefinition] = None
    headers: Dict[str, str] = field(default_factory=dict)
    example: Optional[Dict[str, Any]] = None


@dataclass
class EndpointDefinition:
    """API endpoint definition."""

    path: str
    method: HTTPMethod
    operation_id: str
    summary: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: List[ParameterDefinition] = field(default_factory=list)
    request_body: Optional[SchemaDefinition] = None
    responses: List[ResponseDefinition] = field(default_factory=list)
    security: List[str] = field(default_factory=list)
    deprecated: bool = False


@dataclass
class APIDefinition:
    """Complete API definition."""

    title: str
    version: str
    description: str = ""
    base_url: str = ""
    endpoints: List[EndpointDefinition] = field(default_factory=list)
    schemas: List[SchemaDefinition] = field(default_factory=list)
    security_schemes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tags: List[Dict[str, str]] = field(default_factory=list)
    servers: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class GeneratedFile:
    """Generated SDK file."""

    path: str
    content: str
    language: SDKLanguage
    file_type: str  # "client", "model", "util", "test", "config"
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class SDKConfig:
    """SDK generation configuration."""

    language: SDKLanguage
    package_name: str
    version: str = "1.0.0"
    author: str = ""
    license: str = "MIT"
    include_tests: bool = True
    include_docs: bool = True
    async_support: bool = True
    type_hints: bool = True
    output_dir: str = "generated"
    custom_templates: Dict[str, str] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """SDK generation result."""

    success: bool
    files: List[GeneratedFile] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)


# ========================
# Code Generators
# ========================


class CodeGenerator(ABC):
    """Abstract base class for code generators."""

    @abstractmethod
    def generate_client(self, api_def: APIDefinition, config: SDKConfig) -> List[GeneratedFile]:
        """Generate client code."""
        pass

    @abstractmethod
    def generate_models(
        self, schemas: List[SchemaDefinition], config: SDKConfig
    ) -> List[GeneratedFile]:
        """Generate model/type definitions."""
        pass

    @abstractmethod
    def generate_utils(self, config: SDKConfig) -> List[GeneratedFile]:
        """Generate utility code."""
        pass

    def _to_pascal_case(self, name: str) -> str:
        """Convert string to PascalCase."""
        parts = re.split(r"[-_\s]", name)
        return "".join(part.capitalize() for part in parts)

    def _to_camel_case(self, name: str) -> str:
        """Convert string to camelCase."""
        pascal = self._to_pascal_case(name)
        return pascal[0].lower() + pascal[1:] if pascal else ""

    def _to_snake_case(self, name: str) -> str:
        """Convert string to snake_case."""
        # Handle camelCase and PascalCase
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
        return name.lower().replace("-", "_").replace(" ", "_")


class PythonGenerator(CodeGenerator):
    """Python SDK generator."""

    TYPE_MAPPING = {
        DataType.STRING: "str",
        DataType.INTEGER: "int",
        DataType.NUMBER: "float",
        DataType.BOOLEAN: "bool",
        DataType.ARRAY: "List",
        DataType.OBJECT: "Dict[str, Any]",
        DataType.NULL: "None",
        DataType.BINARY: "bytes",
        DataType.FILE: "BinaryIO",
    }

    def generate_client(self, api_def: APIDefinition, config: SDKConfig) -> List[GeneratedFile]:
        """Generate Python client code."""
        files = []

        # Generate main client class
        client_code = self._generate_client_class(api_def, config)
        files.append(
            GeneratedFile(
                path=f"{config.output_dir}/{config.package_name}/client.py",
                content=client_code,
                language=SDKLanguage.PYTHON,
                file_type="client",
            )
        )

        # Generate __init__.py
        init_code = self._generate_init(api_def, config)
        files.append(
            GeneratedFile(
                path=f"{config.output_dir}/{config.package_name}/__init__.py",
                content=init_code,
                language=SDKLanguage.PYTHON,
                file_type="config",
            )
        )

        return files

    def generate_models(
        self, schemas: List[SchemaDefinition], config: SDKConfig
    ) -> List[GeneratedFile]:
        """Generate Python model classes."""
        model_code = self._generate_models_code(schemas, config)
        return [
            GeneratedFile(
                path=f"{config.output_dir}/{config.package_name}/models.py",
                content=model_code,
                language=SDKLanguage.PYTHON,
                file_type="model",
            )
        ]

    def generate_utils(self, config: SDKConfig) -> List[GeneratedFile]:
        """Generate Python utility code."""
        utils_code = self._generate_utils_code(config)
        return [
            GeneratedFile(
                path=f"{config.output_dir}/{config.package_name}/utils.py",
                content=utils_code,
                language=SDKLanguage.PYTHON,
                file_type="util",
            )
        ]

    def _generate_client_class(self, api_def: APIDefinition, config: SDKConfig) -> str:
        """Generate the main client class."""
        class_name = self._to_pascal_case(config.package_name) + "Client"

        # Group endpoints by tag
        tagged_endpoints: Dict[str, List[EndpointDefinition]] = {}
        for endpoint in api_def.endpoints:
            tag = endpoint.tags[0] if endpoint.tags else "default"
            if tag not in tagged_endpoints:
                tagged_endpoints[tag] = []
            tagged_endpoints[tag].append(endpoint)

        methods = []
        for tag, endpoints in tagged_endpoints.items():
            for endpoint in endpoints:
                method = self._generate_method(endpoint, config)
                methods.append(method)

        async_prefix = "async " if config.async_support else ""
        await_keyword = "await " if config.async_support else ""
        http_lib = "httpx" if config.async_support else "requests"

        return f'''"""
{api_def.title} SDK Client

Auto-generated Python client for {api_def.title} API.
Version: {api_def.version}
"""

from __future__ import annotations

import {http_lib}
from typing import Any, Dict, List, Optional, Union
{"from typing import BinaryIO" if any(e.request_body for e in api_def.endpoints) else ""}

from .models import *
from .utils import APIError, handle_response


class {class_name}:
    """Client for {api_def.title} API."""

    def __init__(
        self,
        base_url: str = "{api_def.base_url or 'http://localhost:8000'}",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the API client.

        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
            timeout: Request timeout in seconds
            headers: Additional headers to include in requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._headers = headers or {{}}
        {"self._client = httpx.AsyncClient(timeout=timeout)" if config.async_support else "self._session = requests.Session()"}

    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers."""
        h = {{"Content-Type": "application/json", **self._headers}}
        if self.api_key:
            h["Authorization"] = f"Bearer {{self.api_key}}"
        return h

    {async_prefix}def close(self) -> None:
        """Close the client session."""
        {await_keyword}{"self._client.aclose()" if config.async_support else "self._session.close()"}

    async def __aenter__(self) -> "{class_name}":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

{chr(10).join(methods)}
'''

    def _generate_method(self, endpoint: EndpointDefinition, config: SDKConfig) -> str:
        """Generate a method for an endpoint."""
        method_name = self._to_snake_case(endpoint.operation_id)
        async_prefix = "async " if config.async_support else ""
        await_keyword = "await " if config.async_support else ""

        # Build parameters
        params = []
        path_params = []
        query_params = []

        for param in endpoint.parameters:
            param_type = self.TYPE_MAPPING.get(param.data_type, "Any")
            if param.required:
                params.append(f"{param.name}: {param_type}")
            else:
                default = repr(param.default) if param.default is not None else "None"
                params.append(f"{param.name}: Optional[{param_type}] = {default}")

            if param.location == ParameterLocation.PATH:
                path_params.append(param.name)
            elif param.location == ParameterLocation.QUERY:
                query_params.append(param.name)

        # Add request body if present
        if endpoint.request_body:
            params.append("body: Dict[str, Any]")

        params_str = ", ".join(["self"] + params)

        # Build URL
        path = endpoint.path
        for p in path_params:
            path = path.replace(f"{{{p}}}", f"{{{p}}}")

        # Build query string
        query_build = ""
        if query_params:
            query_build = f"""
        params = {{}}
        {chr(10).join(f'        if {p} is not None: params["{p}"] = {p}' for p in query_params)}"""

        # Build request
        http_method = endpoint.method.value.lower()

        return f'''
    {async_prefix}def {method_name}({params_str}) -> Dict[str, Any]:
        """
        {endpoint.summary}

        {endpoint.description}

        Returns:
            API response data
        """
        url = f"{{self.base_url}}{path}"{query_build}
        response = {await_keyword}{"self._client" if config.async_support else "self._session"}.{http_method}(
            url,
            headers=self.headers,
            {"json=body," if endpoint.request_body else ""}
            {"params=params," if query_params else ""}
            timeout=self.timeout
        )
        return handle_response(response)
'''

    def _generate_models_code(self, schemas: List[SchemaDefinition], config: SDKConfig) -> str:
        """Generate model classes code."""
        models = []

        for schema in schemas:
            class_name = self._to_pascal_case(schema.name)

            # Build fields
            fields = []
            for prop in schema.properties:
                type_str = self.TYPE_MAPPING.get(prop.data_type, "Any")
                if prop.items_type:
                    items_type = self.TYPE_MAPPING.get(prop.items_type, "Any")
                    type_str = f"List[{items_type}]"

                if not prop.required:
                    type_str = f"Optional[{type_str}]"
                    fields.append(f"    {prop.name}: {type_str} = None")
                else:
                    fields.append(f"    {prop.name}: {type_str}")

            fields_str = "\n".join(fields) if fields else "    pass"

            model = f'''
@dataclass
class {class_name}:
    """
    {schema.description or f'{class_name} model.'}
    """
{fields_str}
'''
            models.append(model)

        return f'''"""
Model definitions for the API.

Auto-generated from API schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
{"".join(models)}
'''

    def _generate_utils_code(self, config: SDKConfig) -> str:
        """Generate utility code."""
        return '''"""
Utility functions for the SDK.
"""

from typing import Any, Dict, Union


class APIError(Exception):
    """API error exception."""

    def __init__(
        self,
        message: str,
        status_code: int = 0,
        response_body: Any = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def handle_response(response: Any) -> Dict[str, Any]:
    """
    Handle API response and raise errors if needed.

    Args:
        response: HTTP response object

    Returns:
        Parsed JSON response

    Raises:
        APIError: If response indicates an error
    """
    if hasattr(response, 'status_code'):
        status = response.status_code
    else:
        status = getattr(response, 'status', 200)

    if status >= 400:
        try:
            body = response.json()
        except Exception:
            body = response.text if hasattr(response, 'text') else str(response)

        raise APIError(
            f"API request failed with status {status}",
            status_code=status,
            response_body=body
        )

    try:
        return response.json()
    except Exception:
        return {"data": response.text if hasattr(response, 'text') else str(response)}
'''

    def _generate_init(self, api_def: APIDefinition, config: SDKConfig) -> str:
        """Generate __init__.py content."""
        class_name = self._to_pascal_case(config.package_name) + "Client"

        return f'''"""
{api_def.title} SDK

Auto-generated SDK for {api_def.title} API.
Version: {config.version}
"""

from .client import {class_name}
from .models import *
from .utils import APIError

__version__ = "{config.version}"
__all__ = ["{class_name}", "APIError"]
'''


class TypeScriptGenerator(CodeGenerator):
    """TypeScript SDK generator."""

    TYPE_MAPPING = {
        DataType.STRING: "string",
        DataType.INTEGER: "number",
        DataType.NUMBER: "number",
        DataType.BOOLEAN: "boolean",
        DataType.ARRAY: "Array",
        DataType.OBJECT: "Record<string, any>",
        DataType.NULL: "null",
        DataType.BINARY: "Blob",
        DataType.FILE: "File",
    }

    def generate_client(self, api_def: APIDefinition, config: SDKConfig) -> List[GeneratedFile]:
        """Generate TypeScript client code."""
        files = []

        # Generate main client
        client_code = self._generate_client_class(api_def, config)
        files.append(
            GeneratedFile(
                path=f"{config.output_dir}/src/client.ts",
                content=client_code,
                language=SDKLanguage.TYPESCRIPT,
                file_type="client",
            )
        )

        # Generate index.ts
        index_code = self._generate_index(api_def, config)
        files.append(
            GeneratedFile(
                path=f"{config.output_dir}/src/index.ts",
                content=index_code,
                language=SDKLanguage.TYPESCRIPT,
                file_type="config",
            )
        )

        # Generate package.json
        package_json = self._generate_package_json(api_def, config)
        files.append(
            GeneratedFile(
                path=f"{config.output_dir}/package.json",
                content=package_json,
                language=SDKLanguage.TYPESCRIPT,
                file_type="config",
            )
        )

        return files

    def generate_models(
        self, schemas: List[SchemaDefinition], config: SDKConfig
    ) -> List[GeneratedFile]:
        """Generate TypeScript interfaces."""
        types_code = self._generate_types(schemas)
        return [
            GeneratedFile(
                path=f"{config.output_dir}/src/types.ts",
                content=types_code,
                language=SDKLanguage.TYPESCRIPT,
                file_type="model",
            )
        ]

    def generate_utils(self, config: SDKConfig) -> List[GeneratedFile]:
        """Generate TypeScript utilities."""
        utils_code = self._generate_utils_code()
        return [
            GeneratedFile(
                path=f"{config.output_dir}/src/utils.ts",
                content=utils_code,
                language=SDKLanguage.TYPESCRIPT,
                file_type="util",
            )
        ]

    def _generate_client_class(self, api_def: APIDefinition, config: SDKConfig) -> str:
        """Generate TypeScript client class."""
        class_name = self._to_pascal_case(config.package_name) + "Client"

        methods = []
        for endpoint in api_def.endpoints:
            method = self._generate_method(endpoint)
            methods.append(method)

        return f"""/**
 * {api_def.title} SDK Client
 *
 * Auto-generated TypeScript client for {api_def.title} API.
 * Version: {api_def.version}
 */

import {{ APIError, handleResponse }} from './utils';
import * as Types from './types';

export interface ClientConfig {{
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  headers?: Record<string, string>;
}}

export class {class_name} {{
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;
  private headers: Record<string, string>;

  constructor(config: ClientConfig = {{}}) {{
    this.baseUrl = (config.baseUrl || '{api_def.base_url or "http://localhost:8000"}').replace(/\\/$/, '');
    this.apiKey = config.apiKey;
    this.timeout = config.timeout || 30000;
    this.headers = config.headers || {{}};
  }}

  private getHeaders(): Record<string, string> {{
    const headers: Record<string, string> = {{
      'Content-Type': 'application/json',
      ...this.headers,
    }};
    if (this.apiKey) {{
      headers['Authorization'] = `Bearer ${{this.apiKey}}`;
    }}
    return headers;
  }}

  private async request<T>(
    method: string,
    path: string,
    options: {{
      params?: Record<string, any>;
      body?: any;
    }} = {{}}
  ): Promise<T> {{
    const url = new URL(`${{this.baseUrl}}${{path}}`);

    if (options.params) {{
      Object.entries(options.params).forEach(([key, value]) => {{
        if (value !== undefined && value !== null) {{
          url.searchParams.append(key, String(value));
        }}
      }});
    }}

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {{
      const response = await fetch(url.toString(), {{
        method,
        headers: this.getHeaders(),
        body: options.body ? JSON.stringify(options.body) : undefined,
        signal: controller.signal,
      }});

      return handleResponse<T>(response);
    }} finally {{
      clearTimeout(timeoutId);
    }}
  }}

{chr(10).join(methods)}
}}
"""

    def _generate_method(self, endpoint: EndpointDefinition) -> str:
        """Generate a TypeScript method."""
        method_name = self._to_camel_case(endpoint.operation_id)

        # Build parameters
        params = []
        path_params = []
        query_params = []

        for param in endpoint.parameters:
            ts_type = self.TYPE_MAPPING.get(param.data_type, "any")
            optional = "" if param.required else "?"
            params.append(f"{param.name}{optional}: {ts_type}")

            if param.location == ParameterLocation.PATH:
                path_params.append(param.name)
            elif param.location == ParameterLocation.QUERY:
                query_params.append(param.name)

        if endpoint.request_body:
            params.append("body: Record<string, any>")

        params_str = ", ".join(params) if params else ""

        # Build path with substitutions
        path = endpoint.path
        for p in path_params:
            path = path.replace(f"{{{p}}}", f"${{{p}}}")

        # Build request options
        options_parts = []
        if query_params:
            options_parts.append(f"params: {{ {', '.join(query_params)} }}")
        if endpoint.request_body:
            options_parts.append("body")
        options = f"{{ {', '.join(options_parts)} }}" if options_parts else ""

        return f"""
  /**
   * {endpoint.summary}
   *
   * {endpoint.description}
   */
  async {method_name}({params_str}): Promise<any> {{
    return this.request('{endpoint.method.value}', `{path}`{f', {options}' if options else ''});
  }}
"""

    def _generate_types(self, schemas: List[SchemaDefinition]) -> str:
        """Generate TypeScript interfaces."""
        interfaces = []

        for schema in schemas:
            interface_name = self._to_pascal_case(schema.name)

            fields = []
            for prop in schema.properties:
                ts_type = self.TYPE_MAPPING.get(prop.data_type, "any")
                if prop.data_type == DataType.ARRAY and prop.items_type:
                    items_type = self.TYPE_MAPPING.get(prop.items_type, "any")
                    ts_type = f"Array<{items_type}>"

                optional = "" if prop.required else "?"
                fields.append(f"  {prop.name}{optional}: {ts_type};")

            fields_str = "\n".join(fields) if fields else "  [key: string]: any;"

            interface = f"""
/**
 * {schema.description or f'{interface_name} interface.'}
 */
export interface {interface_name} {{
{fields_str}
}}
"""
            interfaces.append(interface)

        return f"""/**
 * Type definitions for the API.
 *
 * Auto-generated from API schemas.
 */
{"".join(interfaces)}
"""

    def _generate_utils_code(self) -> str:
        """Generate TypeScript utilities."""
        return """/**
 * Utility functions for the SDK.
 */

export class APIError extends Error {
  statusCode: number;
  responseBody: any;

  constructor(message: string, statusCode: number = 0, responseBody?: any) {
    super(message);
    this.name = 'APIError';
    this.statusCode = statusCode;
    this.responseBody = responseBody;
  }
}

export async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let body: any;
    try {
      body = await response.json();
    } catch {
      body = await response.text();
    }
    throw new APIError(
      `API request failed with status ${response.status}`,
      response.status,
      body
    );
  }

  try {
    return await response.json();
  } catch {
    return { data: await response.text() } as T;
  }
}
"""

    def _generate_index(self, api_def: APIDefinition, config: SDKConfig) -> str:
        """Generate index.ts content."""
        class_name = self._to_pascal_case(config.package_name) + "Client"

        return f"""/**
 * {api_def.title} SDK
 *
 * Auto-generated SDK for {api_def.title} API.
 * Version: {config.version}
 */

export {{ {class_name}, ClientConfig }} from './client';
export * from './types';
export {{ APIError }} from './utils';
"""

    def _generate_package_json(self, api_def: APIDefinition, config: SDKConfig) -> str:
        """Generate package.json."""
        return json.dumps(
            {
                "name": config.package_name,
                "version": config.version,
                "description": api_def.description or f"SDK for {api_def.title}",
                "main": "dist/index.js",
                "types": "dist/index.d.ts",
                "scripts": {"build": "tsc", "test": "jest"},
                "author": config.author,
                "license": config.license,
                "devDependencies": {"typescript": "^5.0.0", "@types/node": "^20.0.0"},
            },
            indent=2,
        )


# ========================
# Specification Generators
# ========================


class OpenAPIGenerator:
    """OpenAPI specification generator."""

    def generate(
        self, api_def: APIDefinition, format: SpecFormat = SpecFormat.OPENAPI_3_0
    ) -> Dict[str, Any]:
        """Generate OpenAPI specification."""
        if format == SpecFormat.OPENAPI_3_1:
            return self._generate_3_1(api_def)
        elif format == SpecFormat.SWAGGER_2_0:
            return self._generate_swagger_2(api_def)
        return self._generate_3_0(api_def)

    def _generate_3_0(self, api_def: APIDefinition) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 spec."""
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": api_def.title,
                "version": api_def.version,
                "description": api_def.description,
            },
            "servers": api_def.servers or [{"url": api_def.base_url or "http://localhost:8000"}],
            "paths": {},
            "components": {"schemas": {}, "securitySchemes": api_def.security_schemes},
        }

        if api_def.tags:
            spec["tags"] = api_def.tags

        # Generate paths
        for endpoint in api_def.endpoints:
            path = endpoint.path
            if path not in spec["paths"]:
                spec["paths"][path] = {}

            operation = {
                "operationId": endpoint.operation_id,
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "parameters": [],
                "responses": {},
            }

            if endpoint.deprecated:
                operation["deprecated"] = True

            # Add parameters
            for param in endpoint.parameters:
                if param.location != ParameterLocation.BODY:
                    operation["parameters"].append(
                        {
                            "name": param.name,
                            "in": param.location.value,
                            "description": param.description,
                            "required": param.required,
                            "schema": {"type": param.data_type.value},
                        }
                    )

            # Add request body
            if endpoint.request_body:
                operation["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": self._schema_to_openapi(endpoint.request_body)
                        }
                    },
                }

            # Add responses
            for response in endpoint.responses:
                resp_obj: Dict[str, Any] = {"description": response.description}
                if response.schema:
                    resp_obj["content"] = {
                        response.content_type: {"schema": self._schema_to_openapi(response.schema)}
                    }
                operation["responses"][str(response.status_code)] = resp_obj

            if not operation["responses"]:
                operation["responses"]["200"] = {"description": "Successful response"}

            spec["paths"][path][endpoint.method.value.lower()] = operation

        # Generate schemas
        for schema in api_def.schemas:
            spec["components"]["schemas"][schema.name] = self._schema_to_openapi(schema)

        return spec

    def _generate_3_1(self, api_def: APIDefinition) -> Dict[str, Any]:
        """Generate OpenAPI 3.1 spec."""
        spec = self._generate_3_0(api_def)
        spec["openapi"] = "3.1.0"
        return spec

    def _generate_swagger_2(self, api_def: APIDefinition) -> Dict[str, Any]:
        """Generate Swagger 2.0 spec."""
        spec = {
            "swagger": "2.0",
            "info": {
                "title": api_def.title,
                "version": api_def.version,
                "description": api_def.description,
            },
            "host": api_def.base_url.replace("http://", "").replace("https://", "")
            if api_def.base_url
            else "localhost:8000",
            "basePath": "/",
            "schemes": ["https", "http"],
            "paths": {},
            "definitions": {},
        }

        # Convert endpoints (simplified)
        for endpoint in api_def.endpoints:
            path = endpoint.path
            if path not in spec["paths"]:
                spec["paths"][path] = {}

            spec["paths"][path][endpoint.method.value.lower()] = {
                "operationId": endpoint.operation_id,
                "summary": endpoint.summary,
                "responses": {"200": {"description": "Success"}},
            }

        return spec

    def _schema_to_openapi(self, schema: SchemaDefinition) -> Dict[str, Any]:
        """Convert schema to OpenAPI format."""
        result: Dict[str, Any] = {"type": "object", "description": schema.description}

        if schema.properties:
            result["properties"] = {}
            for prop in schema.properties:
                prop_def: Dict[str, Any] = {"type": prop.data_type.value}
                if prop.description:
                    prop_def["description"] = prop.description
                if prop.format:
                    prop_def["format"] = prop.format
                if prop.enum_values:
                    prop_def["enum"] = prop.enum_values
                if prop.data_type == DataType.ARRAY and prop.items_type:
                    prop_def["items"] = {"type": prop.items_type.value}
                result["properties"][prop.name] = prop_def

        if schema.required_properties:
            result["required"] = schema.required_properties

        if schema.example:
            result["example"] = schema.example

        return result

    def to_json(self, spec: Dict[str, Any], indent: int = 2) -> str:
        """Convert spec to JSON string."""
        return json.dumps(spec, indent=indent)

    def to_yaml(self, spec: Dict[str, Any]) -> str:
        """Convert spec to YAML string."""
        # Simple YAML conversion without external dependency
        return self._dict_to_yaml(spec)

    def _dict_to_yaml(self, data: Any, indent: int = 0) -> str:
        """Simple dict to YAML converter."""
        lines = []
        prefix = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)) and value:
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._dict_to_yaml(value, indent + 1))
                else:
                    if isinstance(value, str) and ("\n" in value or ":" in value):
                        lines.append(f"{prefix}{key}: |")
                        for line in value.split("\n"):
                            lines.append(f"{prefix}  {line}")
                    else:
                        lines.append(f"{prefix}{key}: {json.dumps(value)}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    first = True
                    for key, value in item.items():
                        if first:
                            lines.append(f"{prefix}- {key}: {json.dumps(value)}")
                            first = False
                        else:
                            lines.append(f"{prefix}  {key}: {json.dumps(value)}")
                else:
                    lines.append(f"{prefix}- {json.dumps(item)}")

        return "\n".join(lines)


class GraphQLSchemaGenerator:
    """GraphQL schema generator."""

    TYPE_MAPPING = {
        DataType.STRING: "String",
        DataType.INTEGER: "Int",
        DataType.NUMBER: "Float",
        DataType.BOOLEAN: "Boolean",
        DataType.ARRAY: "[",  # Will be completed with item type
        DataType.OBJECT: "JSON",
        DataType.BINARY: "String",
        DataType.FILE: "Upload",
    }

    def generate(self, api_def: APIDefinition) -> str:
        """Generate GraphQL schema."""
        lines = [
            '"""',
            f"{api_def.title} GraphQL Schema",
            f"Version: {api_def.version}",
            '"""',
            "",
            "scalar JSON",
            "scalar DateTime",
            "scalar Upload",
            "",
        ]

        # Generate types from schemas
        for schema in api_def.schemas:
            lines.extend(self._generate_type(schema))

        # Generate Query type
        queries = [e for e in api_def.endpoints if e.method == HTTPMethod.GET]
        if queries:
            lines.append("type Query {")
            for endpoint in queries:
                query = self._endpoint_to_field(endpoint)
                lines.append(f"  {query}")
            lines.append("}")
            lines.append("")

        # Generate Mutation type
        mutations = [
            e
            for e in api_def.endpoints
            if e.method in (HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.DELETE)
        ]
        if mutations:
            lines.append("type Mutation {")
            for endpoint in mutations:
                mutation = self._endpoint_to_field(endpoint)
                lines.append(f"  {mutation}")
            lines.append("}")

        return "\n".join(lines)

    def _generate_type(self, schema: SchemaDefinition) -> List[str]:
        """Generate GraphQL type from schema."""
        lines = []

        type_name = self._to_pascal_case(schema.name)

        if schema.description:
            lines.append(f'"""{schema.description}"""')

        lines.append(f"type {type_name} {{")

        for prop in schema.properties:
            gql_type = self._get_graphql_type(prop)
            required = "!" if prop.required else ""
            desc = f'  """{prop.description}"""' if prop.description else ""
            if desc:
                lines.append(desc)
            lines.append(f"  {prop.name}: {gql_type}{required}")

        if not schema.properties:
            lines.append("  _empty: String")

        lines.append("}")
        lines.append("")

        return lines

    def _get_graphql_type(self, prop: SchemaProperty) -> str:
        """Get GraphQL type for property."""
        base_type = self.TYPE_MAPPING.get(prop.data_type, "JSON")

        if prop.data_type == DataType.ARRAY:
            items_type = (
                self.TYPE_MAPPING.get(prop.items_type, "JSON") if prop.items_type else "JSON"
            )
            return f"[{items_type}]"

        return base_type

    def _endpoint_to_field(self, endpoint: EndpointDefinition) -> str:
        """Convert endpoint to GraphQL field."""
        name = self._to_camel_case(endpoint.operation_id)

        args = []
        for param in endpoint.parameters:
            gql_type = self.TYPE_MAPPING.get(param.data_type, "String")
            required = "!" if param.required else ""
            args.append(f"{param.name}: {gql_type}{required}")

        args_str = f"({', '.join(args)})" if args else ""

        return f"{name}{args_str}: JSON"

    def _to_pascal_case(self, name: str) -> str:
        """Convert to PascalCase."""
        parts = re.split(r"[-_\s]", name)
        return "".join(part.capitalize() for part in parts)

    def _to_camel_case(self, name: str) -> str:
        """Convert to camelCase."""
        pascal = self._to_pascal_case(name)
        return pascal[0].lower() + pascal[1:] if pascal else ""


# ========================
# SDK Generator Manager
# ========================


class SDKGenerator:
    """Main SDK generator orchestrator."""

    def __init__(self):
        """Initialize SDK generator."""
        self._lock = threading.Lock()
        self._generators: Dict[SDKLanguage, CodeGenerator] = {
            SDKLanguage.PYTHON: PythonGenerator(),
            SDKLanguage.TYPESCRIPT: TypeScriptGenerator(),
        }
        self._openapi_generator = OpenAPIGenerator()
        self._graphql_generator = GraphQLSchemaGenerator()
        self._generation_history: List[GenerationResult] = []

    def register_generator(self, language: SDKLanguage, generator: CodeGenerator) -> None:
        """Register a code generator."""
        with self._lock:
            self._generators[language] = generator

    def generate_sdk(self, api_def: APIDefinition, config: SDKConfig) -> GenerationResult:
        """Generate complete SDK for a language."""
        errors = []
        warnings = []
        files: List[GeneratedFile] = []

        generator = self._generators.get(config.language)
        if not generator:
            return GenerationResult(
                success=False, errors=[f"No generator available for language: {config.language}"]
            )

        try:
            # Generate client
            client_files = generator.generate_client(api_def, config)
            files.extend(client_files)

            # Generate models
            if api_def.schemas:
                model_files = generator.generate_models(api_def.schemas, config)
                files.extend(model_files)

            # Generate utils
            util_files = generator.generate_utils(config)
            files.extend(util_files)

            stats = {
                "total_files": len(files),
                "client_files": len([f for f in files if f.file_type == "client"]),
                "model_files": len([f for f in files if f.file_type == "model"]),
                "util_files": len([f for f in files if f.file_type == "util"]),
                "total_size": sum(len(f.content) for f in files),
            }

            result = GenerationResult(
                success=True, files=files, errors=errors, warnings=warnings, stats=stats
            )

        except Exception as e:
            result = GenerationResult(success=False, files=files, errors=[str(e)])

        with self._lock:
            self._generation_history.append(result)

        return result

    def generate_openapi_spec(
        self, api_def: APIDefinition, format: SpecFormat = SpecFormat.OPENAPI_3_0
    ) -> str:
        """Generate OpenAPI specification."""
        spec = self._openapi_generator.generate(api_def, format)
        return self._openapi_generator.to_json(spec)

    def generate_graphql_schema(self, api_def: APIDefinition) -> str:
        """Generate GraphQL schema."""
        return self._graphql_generator.generate(api_def)

    def get_supported_languages(self) -> List[SDKLanguage]:
        """Get list of supported SDK languages."""
        return list(self._generators.keys())

    def get_generation_history(self) -> List[GenerationResult]:
        """Get generation history."""
        with self._lock:
            return list(self._generation_history)


# ========================
# API Definition Builder
# ========================


class APIDefinitionBuilder:
    """Builder for API definitions."""

    def __init__(self):
        """Initialize builder."""
        self._title = ""
        self._version = "1.0.0"
        self._description = ""
        self._base_url = ""
        self._endpoints: List[EndpointDefinition] = []
        self._schemas: List[SchemaDefinition] = []
        self._security_schemes: Dict[str, Dict[str, Any]] = {}
        self._tags: List[Dict[str, str]] = []
        self._servers: List[Dict[str, str]] = []

    def title(self, title: str) -> "APIDefinitionBuilder":
        """Set API title."""
        self._title = title
        return self

    def version(self, version: str) -> "APIDefinitionBuilder":
        """Set API version."""
        self._version = version
        return self

    def description(self, description: str) -> "APIDefinitionBuilder":
        """Set API description."""
        self._description = description
        return self

    def base_url(self, url: str) -> "APIDefinitionBuilder":
        """Set base URL."""
        self._base_url = url
        return self

    def add_endpoint(self, endpoint: EndpointDefinition) -> "APIDefinitionBuilder":
        """Add an endpoint."""
        self._endpoints.append(endpoint)
        return self

    def add_schema(self, schema: SchemaDefinition) -> "APIDefinitionBuilder":
        """Add a schema."""
        self._schemas.append(schema)
        return self

    def add_security_scheme(
        self, name: str, scheme_type: str, **kwargs: Any
    ) -> "APIDefinitionBuilder":
        """Add a security scheme."""
        self._security_schemes[name] = {"type": scheme_type, **kwargs}
        return self

    def add_tag(self, name: str, description: str = "") -> "APIDefinitionBuilder":
        """Add a tag."""
        self._tags.append({"name": name, "description": description})
        return self

    def add_server(self, url: str, description: str = "") -> "APIDefinitionBuilder":
        """Add a server."""
        self._servers.append({"url": url, "description": description})
        return self

    def build(self) -> APIDefinition:
        """Build the API definition."""
        return APIDefinition(
            title=self._title,
            version=self._version,
            description=self._description,
            base_url=self._base_url,
            endpoints=self._endpoints,
            schemas=self._schemas,
            security_schemes=self._security_schemes,
            tags=self._tags,
            servers=self._servers,
        )


# ========================
# Vision Provider
# ========================


class SDKGeneratorVisionProvider(VisionProvider):
    """Vision provider with SDK generation capabilities."""

    def __init__(self, base_provider: VisionProvider, sdk_generator: Optional[SDKGenerator] = None):
        """Initialize SDK generator vision provider."""
        self._base_provider = base_provider
        self._sdk_generator = sdk_generator or SDKGenerator()
        self._api_definition: Optional[APIDefinition] = None

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"sdk_generator_{self._base_provider.provider_name}"

    async def analyze_image(
        self, image_data: bytes, include_description: bool = True, **kwargs: Any
    ) -> VisionDescription:
        """Analyze image with SDK generation support."""
        result = await self._base_provider.analyze_image(
            image_data, include_description=include_description, **kwargs
        )
        return result

    def set_api_definition(self, api_def: APIDefinition) -> None:
        """Set the API definition for SDK generation."""
        self._api_definition = api_def

    def generate_sdk(self, config: SDKConfig) -> GenerationResult:
        """Generate SDK for the configured API."""
        if not self._api_definition:
            return GenerationResult(success=False, errors=["No API definition set"])
        return self._sdk_generator.generate_sdk(self._api_definition, config)

    def generate_openapi(self, format: SpecFormat = SpecFormat.OPENAPI_3_0) -> Optional[str]:
        """Generate OpenAPI specification."""
        if not self._api_definition:
            return None
        return self._sdk_generator.generate_openapi_spec(self._api_definition, format)

    def generate_graphql(self) -> Optional[str]:
        """Generate GraphQL schema."""
        if not self._api_definition:
            return None
        return self._sdk_generator.generate_graphql_schema(self._api_definition)


# ========================
# Factory Functions
# ========================


def create_sdk_generator() -> SDKGenerator:
    """Create a new SDK generator instance."""
    return SDKGenerator()


def create_api_definition_builder() -> APIDefinitionBuilder:
    """Create a new API definition builder."""
    return APIDefinitionBuilder()


def create_sdk_generator_provider(
    base_provider: VisionProvider, sdk_generator: Optional[SDKGenerator] = None
) -> SDKGeneratorVisionProvider:
    """Create an SDK generator vision provider."""
    return SDKGeneratorVisionProvider(base_provider, sdk_generator)


def create_python_generator() -> PythonGenerator:
    """Create a Python code generator."""
    return PythonGenerator()


def create_typescript_generator() -> TypeScriptGenerator:
    """Create a TypeScript code generator."""
    return TypeScriptGenerator()


def create_openapi_generator() -> OpenAPIGenerator:
    """Create an OpenAPI specification generator."""
    return OpenAPIGenerator()


def create_graphql_generator() -> GraphQLSchemaGenerator:
    """Create a GraphQL schema generator."""
    return GraphQLSchemaGenerator()
