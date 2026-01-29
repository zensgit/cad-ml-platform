"""
REST API Service for CAD Assistant.

Provides HTTP endpoints for the CAD assistant functionality,
enabling integration with web applications and external services.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Callable
from functools import wraps


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class ValidationError(APIError):
    """Request validation error."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
        )
        self.field = field


class NotFoundError(APIError):
    """Resource not found error."""

    def __init__(self, resource: str, resource_id: str):
        super().__init__(
            message=f"{resource} not found: {resource_id}",
            status_code=404,
            error_code="NOT_FOUND",
        )


class RateLimitError(APIError):
    """Rate limit exceeded error."""

    def __init__(self, retry_after: int = 60):
        super().__init__(
            message=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
        )
        self.retry_after = retry_after


@dataclass
class APIRequest:
    """Base API request structure."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


@dataclass
class APIResponse:
    """Base API response structure."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "success": self.success,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class AskRequest(APIRequest):
    """Request for /ask endpoint."""

    query: str = ""
    conversation_id: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AskRequest":
        """Create from dictionary."""
        return cls(
            query=data.get("query", ""),
            conversation_id=data.get("conversation_id"),
            options=data.get("options", {}),
        )

    def validate(self) -> None:
        """Validate request data."""
        if not self.query or not self.query.strip():
            raise ValidationError("Query cannot be empty", field="query")
        if len(self.query) > 2000:
            raise ValidationError(
                "Query too long (max 2000 characters)", field="query"
            )


@dataclass
class ConversationRequest(APIRequest):
    """Request for conversation operations."""

    action: str = ""  # "create", "get", "delete", "list"
    conversation_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationRequest":
        """Create from dictionary."""
        return cls(
            action=data.get("action", ""),
            conversation_id=data.get("conversation_id"),
        )

    def validate(self) -> None:
        """Validate request data."""
        valid_actions = ["create", "get", "delete", "list", "clear"]
        if self.action not in valid_actions:
            raise ValidationError(
                f"Invalid action. Must be one of: {valid_actions}",
                field="action",
            )
        if self.action in ["get", "delete", "clear"] and not self.conversation_id:
            raise ValidationError(
                "conversation_id required for this action",
                field="conversation_id",
            )


@dataclass
class EvaluationRequest(APIRequest):
    """Request for response evaluation."""

    query: str = ""
    response: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationRequest":
        """Create from dictionary."""
        return cls(
            query=data.get("query", ""),
            response=data.get("response", ""),
        )

    def validate(self) -> None:
        """Validate request data."""
        if not self.query:
            raise ValidationError("Query cannot be empty", field="query")
        if not self.response:
            raise ValidationError("Response cannot be empty", field="response")


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute
            requests_per_hour: Max requests per hour
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self._requests: Dict[str, List[float]] = {}

    def check(self, client_id: str) -> bool:
        """
        Check if request is allowed.

        Args:
            client_id: Client identifier

        Returns:
            True if request is allowed
        """
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        if client_id not in self._requests:
            self._requests[client_id] = []

        # Clean old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > hour_ago
        ]

        # Check limits
        recent_minute = sum(1 for t in self._requests[client_id] if t > minute_ago)
        recent_hour = len(self._requests[client_id])

        if recent_minute >= self.requests_per_minute:
            return False
        if recent_hour >= self.requests_per_hour:
            return False

        # Record request
        self._requests[client_id].append(now)
        return True

    def get_retry_after(self, client_id: str) -> int:
        """Get seconds until rate limit resets."""
        if client_id not in self._requests:
            return 0

        now = time.time()
        minute_ago = now - 60

        recent = [t for t in self._requests[client_id] if t > minute_ago]
        if len(recent) >= self.requests_per_minute and recent:
            return int(60 - (now - recent[0]))

        return 60


class CADAssistantAPI:
    """
    REST API wrapper for CAD Assistant.

    Provides a clean API interface with request/response handling,
    validation, rate limiting, and error management.

    Example:
        >>> api = CADAssistantAPI()
        >>> response = api.ask({"query": "304不锈钢的强度是多少？"})
        >>> print(response.to_json())
    """

    def __init__(
        self,
        rate_limit_enabled: bool = True,
        requests_per_minute: int = 60,
    ):
        """
        Initialize API.

        Args:
            rate_limit_enabled: Enable rate limiting
            requests_per_minute: Rate limit per minute per client
        """
        self.rate_limit_enabled = rate_limit_enabled
        self._rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute
        )

        # Lazy imports to avoid circular dependencies
        self._assistant = None
        self._evaluator = None
        self._persistence = None

    def _get_assistant(self):
        """Lazy load assistant."""
        if self._assistant is None:
            from src.core.assistant import CADAssistant
            self._assistant = CADAssistant()
        return self._assistant

    def _get_evaluator(self):
        """Lazy load evaluator."""
        if self._evaluator is None:
            from src.core.assistant.quality_evaluation import ResponseQualityEvaluator
            self._evaluator = ResponseQualityEvaluator()
        return self._evaluator

    def _get_persistence(self):
        """Lazy load persistence."""
        if self._persistence is None:
            from src.core.assistant.persistence import ConversationPersistence
            self._persistence = ConversationPersistence()
        return self._persistence

    def _check_rate_limit(self, client_id: str) -> None:
        """Check rate limit and raise error if exceeded."""
        if not self.rate_limit_enabled:
            return

        if not self._rate_limiter.check(client_id):
            retry_after = self._rate_limiter.get_retry_after(client_id)
            raise RateLimitError(retry_after=retry_after)

    def _make_response(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[APIError] = None,
        request_id: str = "",
    ) -> APIResponse:
        """Create API response."""
        error_dict = None
        if error:
            error_dict = {
                "code": error.error_code,
                "message": error.message,
            }
            if isinstance(error, ValidationError) and error.field:
                error_dict["field"] = error.field

        return APIResponse(
            success=success,
            data=data,
            error=error_dict,
            request_id=request_id,
        )

    def ask(
        self,
        request_data: Dict[str, Any],
        client_id: str = "default",
    ) -> APIResponse:
        """
        Handle /ask endpoint - query the assistant.

        Args:
            request_data: Request data dictionary
            client_id: Client identifier for rate limiting

        Returns:
            API response
        """
        request = AskRequest.from_dict(request_data)

        try:
            self._check_rate_limit(client_id)
            request.validate()

            assistant = self._get_assistant()

            # Handle conversation context
            conv_id = request.conversation_id
            if not conv_id:
                conv_id = assistant.start_conversation()

            # Get response
            result = assistant.ask(
                request.query,
                conversation_id=conv_id,
            )

            return self._make_response(
                success=True,
                data={
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "sources": [s.to_dict() if hasattr(s, 'to_dict') else str(s)
                               for s in (result.sources or [])],
                    "conversation_id": conv_id,
                    "intent": result.intent.value if result.intent else None,
                },
                request_id=request.request_id,
            )

        except APIError as e:
            return self._make_response(
                success=False,
                error=e,
                request_id=request.request_id,
            )
        except Exception as e:
            return self._make_response(
                success=False,
                error=APIError(str(e)),
                request_id=request.request_id,
            )

    def conversation(
        self,
        request_data: Dict[str, Any],
        client_id: str = "default",
    ) -> APIResponse:
        """
        Handle /conversation endpoint - manage conversations.

        Args:
            request_data: Request data dictionary
            client_id: Client identifier for rate limiting

        Returns:
            API response
        """
        request = ConversationRequest.from_dict(request_data)

        try:
            self._check_rate_limit(client_id)
            request.validate()

            assistant = self._get_assistant()

            if request.action == "create":
                conv_id = assistant.start_conversation()
                return self._make_response(
                    success=True,
                    data={"conversation_id": conv_id},
                    request_id=request.request_id,
                )

            elif request.action == "get":
                history = assistant.get_conversation_history(request.conversation_id)
                return self._make_response(
                    success=True,
                    data={
                        "conversation_id": request.conversation_id,
                        "messages": history,
                    },
                    request_id=request.request_id,
                )

            elif request.action == "delete":
                success = assistant.end_conversation(request.conversation_id)
                return self._make_response(
                    success=success,
                    data={"deleted": success},
                    request_id=request.request_id,
                )

            elif request.action == "clear":
                if assistant._conversation_manager:
                    success = assistant._conversation_manager.clear_conversation(
                        request.conversation_id
                    )
                    return self._make_response(
                        success=success,
                        data={"cleared": success},
                        request_id=request.request_id,
                    )
                return self._make_response(
                    success=False,
                    error=APIError("Conversation manager not available"),
                    request_id=request.request_id,
                )

            elif request.action == "list":
                if assistant._conversation_manager:
                    conversations = assistant._conversation_manager.list_conversations()
                    return self._make_response(
                        success=True,
                        data={"conversations": conversations},
                        request_id=request.request_id,
                    )
                return self._make_response(
                    success=True,
                    data={"conversations": []},
                    request_id=request.request_id,
                )

        except APIError as e:
            return self._make_response(
                success=False,
                error=e,
                request_id=request.request_id,
            )
        except Exception as e:
            return self._make_response(
                success=False,
                error=APIError(str(e)),
                request_id=request.request_id,
            )

    def evaluate(
        self,
        request_data: Dict[str, Any],
        client_id: str = "default",
    ) -> APIResponse:
        """
        Handle /evaluate endpoint - evaluate response quality.

        Args:
            request_data: Request data dictionary
            client_id: Client identifier for rate limiting

        Returns:
            API response
        """
        request = EvaluationRequest.from_dict(request_data)

        try:
            self._check_rate_limit(client_id)
            request.validate()

            evaluator = self._get_evaluator()
            result = evaluator.evaluate(request.query, request.response)

            return self._make_response(
                success=True,
                data=result.to_dict(),
                request_id=request.request_id,
            )

        except APIError as e:
            return self._make_response(
                success=False,
                error=e,
                request_id=request.request_id,
            )
        except Exception as e:
            return self._make_response(
                success=False,
                error=APIError(str(e)),
                request_id=request.request_id,
            )

    def health(self) -> APIResponse:
        """
        Handle /health endpoint - health check.

        Returns:
            API response with health status
        """
        return self._make_response(
            success=True,
            data={
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": time.time(),
            },
        )

    def info(self) -> APIResponse:
        """
        Handle /info endpoint - API information.

        Returns:
            API response with API info
        """
        return self._make_response(
            success=True,
            data={
                "name": "CAD Assistant API",
                "version": "1.0.0",
                "endpoints": [
                    {
                        "path": "/ask",
                        "method": "POST",
                        "description": "Query the CAD assistant",
                    },
                    {
                        "path": "/conversation",
                        "method": "POST",
                        "description": "Manage conversations",
                    },
                    {
                        "path": "/evaluate",
                        "method": "POST",
                        "description": "Evaluate response quality",
                    },
                    {
                        "path": "/health",
                        "method": "GET",
                        "description": "Health check",
                    },
                    {
                        "path": "/info",
                        "method": "GET",
                        "description": "API information",
                    },
                ],
            },
        )


def create_flask_app(api: Optional[CADAssistantAPI] = None):
    """
    Create Flask application with API routes.

    Args:
        api: CADAssistantAPI instance (creates new if None)

    Returns:
        Flask application

    Note:
        Requires Flask to be installed.
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise ImportError(
            "Flask is required for the web server. "
            "Install with: pip install flask"
        )

    app = Flask(__name__)
    api = api or CADAssistantAPI()

    @app.route("/health", methods=["GET"])
    def health():
        response = api.health()
        return jsonify(response.to_dict())

    @app.route("/info", methods=["GET"])
    def info():
        response = api.info()
        return jsonify(response.to_dict())

    @app.route("/ask", methods=["POST"])
    def ask():
        data = request.get_json() or {}
        client_id = request.headers.get("X-Client-ID", "default")
        response = api.ask(data, client_id)
        return jsonify(response.to_dict()), (
            200 if response.success else
            response.error.get("status_code", 500) if response.error else 500
        )

    @app.route("/conversation", methods=["POST"])
    def conversation():
        data = request.get_json() or {}
        client_id = request.headers.get("X-Client-ID", "default")
        response = api.conversation(data, client_id)
        return jsonify(response.to_dict()), (
            200 if response.success else 400
        )

    @app.route("/evaluate", methods=["POST"])
    def evaluate():
        data = request.get_json() or {}
        client_id = request.headers.get("X-Client-ID", "default")
        response = api.evaluate(data, client_id)
        return jsonify(response.to_dict()), (
            200 if response.success else 400
        )

    return app


def create_fastapi_app(api: Optional[CADAssistantAPI] = None):
    """
    Create FastAPI application with API routes.

    Args:
        api: CADAssistantAPI instance (creates new if None)

    Returns:
        FastAPI application

    Note:
        Requires FastAPI to be installed.
    """
    try:
        from fastapi import FastAPI, Request, Header
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError(
            "FastAPI is required for the web server. "
            "Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="CAD Assistant API",
        description="REST API for CAD/manufacturing knowledge assistant",
        version="1.0.0",
    )
    api_instance = api or CADAssistantAPI()

    @app.get("/health")
    async def health():
        response = api_instance.health()
        return response.to_dict()

    @app.get("/info")
    async def info():
        response = api_instance.info()
        return response.to_dict()

    @app.post("/ask")
    async def ask(
        request: Request,
        x_client_id: str = Header(default="default"),
    ):
        data = await request.json()
        response = api_instance.ask(data, x_client_id)
        return response.to_dict()

    @app.post("/conversation")
    async def conversation(
        request: Request,
        x_client_id: str = Header(default="default"),
    ):
        data = await request.json()
        response = api_instance.conversation(data, x_client_id)
        return response.to_dict()

    @app.post("/evaluate")
    async def evaluate(
        request: Request,
        x_client_id: str = Header(default="default"),
    ):
        data = await request.json()
        response = api_instance.evaluate(data, x_client_id)
        return response.to_dict()

    return app
