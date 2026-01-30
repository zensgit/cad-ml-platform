"""OpenAPI documentation configuration and enhancements.

Features:
- Custom OpenAPI schema
- API documentation with examples
- Response models
- Authentication documentation
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="CAD ML Platform API",
        version="2.0.0",
        description="""
# CAD ML Platform API

Enterprise-grade machine learning platform for CAD analysis and processing.

## Features

- **OCR Processing**: Extract text and dimensions from CAD drawings
- **Vision Analysis**: Analyze CAD drawings using computer vision
- **Material Classification**: Identify and classify materials
- **Digital Twin**: Real-time asset monitoring and synchronization
- **Batch Processing**: Bulk operations with job tracking
- **Intelligent Assistant**: AI-powered CAD assistance

## Authentication

All API endpoints require authentication via API key:

```
X-API-Key: your-api-key
```

For admin operations:
```
X-Admin-Token: your-admin-token
```

## Rate Limiting

Rate limits are applied per-tier:

| Tier | Requests/min | Burst |
|------|-------------|-------|
| Anonymous | 60 | 5 |
| Free | 300 | 10 |
| Pro | 1200 | 50 |
| Enterprise | 6000 | 200 |

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`

## Versioning

API version can be specified via:
- Path prefix: `/api/v1/...` or `/api/v2/...`
- Header: `X-API-Version: v2`

## Error Handling

All errors follow this format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {}
  }
}
```

## WebSocket

Real-time notifications available at:
- `ws://host/api/v1/ws`
- `ws://host/api/v1/ws/notifications/{channel}`

## Support

For API support, contact: api-support@example.com
        """,
        routes=app.routes,
        tags=[
            {
                "name": "健康",
                "description": "Health check endpoints",
            },
            {
                "name": "OCR",
                "description": "Optical Character Recognition for CAD drawings",
            },
            {
                "name": "视觉",
                "description": "Computer vision analysis endpoints",
            },
            {
                "name": "分析",
                "description": "CAD analysis and processing",
            },
            {
                "name": "向量",
                "description": "Vector database operations",
            },
            {
                "name": "材料",
                "description": "Material classification and lookup",
            },
            {
                "name": "数字孪生",
                "description": "Digital twin real-time synchronization",
            },
            {
                "name": "智能助手",
                "description": "AI-powered CAD assistant",
            },
            {
                "name": "batch",
                "description": "Batch processing operations",
            },
            {
                "name": "websocket",
                "description": "WebSocket real-time notifications",
            },
        ],
    )

    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication",
        },
        "AdminToken": {
            "type": "apiKey",
            "in": "header",
            "name": "X-Admin-Token",
            "description": "Admin token for privileged operations",
        },
    }

    # Apply security globally
    openapi_schema["security"] = [{"ApiKeyAuth": []}]

    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Local development server",
        },
        {
            "url": "https://api.example.com",
            "description": "Production server",
        },
    ]

    # Add contact and license info
    openapi_schema["info"]["contact"] = {
        "name": "API Support",
        "email": "api-support@example.com",
        "url": "https://docs.example.com",
    }
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    # Add external docs
    openapi_schema["externalDocs"] = {
        "description": "Full documentation",
        "url": "https://docs.example.com/api",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Common response examples
RESPONSE_EXAMPLES = {
    "success": {
        "summary": "Successful response",
        "value": {
            "success": True,
            "data": {},
            "meta": {"version": "v2"},
        },
    },
    "error_400": {
        "summary": "Bad Request",
        "value": {
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid input data",
                "details": {"field": "error description"},
            },
        },
    },
    "error_401": {
        "summary": "Unauthorized",
        "value": {
            "success": False,
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Missing or invalid API key",
            },
        },
    },
    "error_403": {
        "summary": "Forbidden",
        "value": {
            "success": False,
            "error": {
                "code": "FORBIDDEN",
                "message": "Insufficient permissions",
            },
        },
    },
    "error_404": {
        "summary": "Not Found",
        "value": {
            "success": False,
            "error": {
                "code": "NOT_FOUND",
                "message": "Resource not found",
            },
        },
    },
    "error_429": {
        "summary": "Rate Limited",
        "value": {
            "success": False,
            "error": {
                "code": "RATE_LIMITED",
                "message": "Rate limit exceeded. Please retry later.",
            },
        },
    },
    "error_500": {
        "summary": "Internal Server Error",
        "value": {
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
            },
        },
    },
}


def setup_openapi(app: FastAPI) -> None:
    """Setup OpenAPI documentation for the app."""
    app.openapi = lambda: custom_openapi(app)
