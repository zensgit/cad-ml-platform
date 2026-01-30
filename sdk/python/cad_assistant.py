"""
CAD Assistant Python SDK.

A Python client library for interacting with the CAD Assistant API.

Usage:
    from cad_assistant import CADAssistant

    client = CADAssistant(api_key="your-api-key")

    # Simple query
    response = client.ask("304不锈钢的抗拉强度是多少？")
    print(response.answer)

    # Streaming query
    for chunk in client.ask_stream("解释焊接工艺"):
        print(chunk, end="", flush=True)
"""

import json
import time
from typing import Optional, Dict, Any, Iterator, List
from dataclasses import dataclass, field
from enum import Enum
import urllib.request
import urllib.error


class CADAssistantError(Exception):
    """Base exception for CAD Assistant SDK."""

    def __init__(self, message: str, code: Optional[str] = None, status: int = 0):
        self.message = message
        self.code = code
        self.status = status
        super().__init__(message)


class RateLimitError(CADAssistantError):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(CADAssistantError):
    """Raised when authentication fails."""
    pass


class ValidationError(CADAssistantError):
    """Raised when request validation fails."""
    pass


@dataclass
class Source:
    """A source reference in the response."""
    type: str
    content: str
    relevance: float = 0.0


@dataclass
class AskResponse:
    """Response from ask endpoint."""
    answer: str
    confidence: float
    conversation_id: Optional[str] = None
    sources: List[Source] = field(default_factory=list)
    model_used: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class Conversation:
    """A conversation session."""
    id: str
    created_at: float
    updated_at: float
    message_count: int = 0


@dataclass
class Message:
    """A message in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float


@dataclass
class KnowledgeResult:
    """A knowledge search result."""
    id: str
    content: str
    category: str
    relevance: float


class CADAssistant:
    """
    CAD Assistant API Client.

    Args:
        api_key: Your API key
        base_url: API base URL (default: http://localhost:8000)
        timeout: Request timeout in seconds (default: 30)
        tenant_id: Optional tenant ID for multi-tenant setups
    """

    DEFAULT_BASE_URL = "http://localhost:8000"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: int = 30,
        tenant_id: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.tenant_id = tenant_id
        self._conversation_id: Optional[str] = None

    def _headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.tenant_id:
            headers["X-Tenant-ID"] = self.tenant_id
        return headers

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request."""
        url = f"{self.base_url}{path}"

        body = None
        if data:
            body = json.dumps(data).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method=method,
        )

        try:
            start_time = time.time()
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                latency = (time.time() - start_time) * 1000
                result = json.loads(response.read().decode("utf-8"))
                result["_latency_ms"] = latency
                return result
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            try:
                error_data = json.loads(body)
                error_msg = error_data.get("error", {}).get("message", str(e))
                error_code = error_data.get("error", {}).get("code", "unknown")
            except json.JSONDecodeError:
                error_msg = str(e)
                error_code = "unknown"

            if e.code == 401:
                raise AuthenticationError(error_msg, error_code, e.code)
            elif e.code == 429:
                raise RateLimitError(error_msg, error_code, e.code)
            elif e.code == 400:
                raise ValidationError(error_msg, error_code, e.code)
            else:
                raise CADAssistantError(error_msg, error_code, e.code)
        except urllib.error.URLError as e:
            raise CADAssistantError(f"Connection error: {e.reason}")

    # ========== Health ==========

    def health(self) -> Dict[str, Any]:
        """
        Check API health status.

        Returns:
            Health status dict with 'status', 'version', 'timestamp'
        """
        return self._request("GET", "/health")

    # ========== Ask ==========

    def ask(
        self,
        query: str,
        conversation_id: Optional[str] = None,
    ) -> AskResponse:
        """
        Ask a question.

        Args:
            query: The question to ask
            conversation_id: Optional conversation ID for multi-turn

        Returns:
            AskResponse with answer and metadata
        """
        data = {"query": query}
        if conversation_id or self._conversation_id:
            data["conversation_id"] = conversation_id or self._conversation_id

        result = self._request("POST", "/ask", data)

        response_data = result.get("data", {})

        # Parse sources
        sources = []
        for s in response_data.get("sources", []):
            sources.append(Source(
                type=s.get("type", "unknown"),
                content=s.get("content", ""),
                relevance=s.get("relevance", 0.0),
            ))

        response = AskResponse(
            answer=response_data.get("answer", ""),
            confidence=response_data.get("confidence", 0.0),
            conversation_id=response_data.get("conversation_id"),
            sources=sources,
            latency_ms=result.get("_latency_ms", 0),
        )

        # Store conversation ID for multi-turn
        if response.conversation_id:
            self._conversation_id = response.conversation_id

        return response

    def ask_stream(
        self,
        query: str,
        conversation_id: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Ask a question with streaming response.

        Args:
            query: The question to ask
            conversation_id: Optional conversation ID

        Yields:
            Text chunks as they arrive
        """
        url = f"{self.base_url}/ask/stream"
        data = {"query": query}
        if conversation_id or self._conversation_id:
            data["conversation_id"] = conversation_id or self._conversation_id

        body = json.dumps(data).encode("utf-8")
        headers = self._headers()
        headers["Accept"] = "text/event-stream"

        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                            if event.get("type") == "chunk":
                                yield event.get("data", {}).get("text", "")
                        except json.JSONDecodeError:
                            continue
        except urllib.error.HTTPError as e:
            raise CADAssistantError(f"Stream error: {e}")

    # ========== Conversations ==========

    def create_conversation(self) -> Conversation:
        """
        Create a new conversation.

        Returns:
            New Conversation object
        """
        result = self._request("POST", "/conversations")
        return Conversation(
            id=result.get("id", ""),
            created_at=result.get("created_at", 0),
            updated_at=result.get("updated_at", 0),
            message_count=result.get("message_count", 0),
        )

    def list_conversations(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Conversation]:
        """
        List conversations.

        Args:
            limit: Max results to return
            offset: Pagination offset

        Returns:
            List of Conversation objects
        """
        result = self._request(
            "GET",
            f"/conversations?limit={limit}&offset={offset}"
        )

        conversations = []
        for c in result:
            conversations.append(Conversation(
                id=c.get("id", ""),
                created_at=c.get("created_at", 0),
                updated_at=c.get("updated_at", 0),
                message_count=c.get("message_count", 0),
            ))
        return conversations

    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation details including messages.

        Args:
            conversation_id: The conversation ID

        Returns:
            Conversation details with messages
        """
        return self._request("GET", f"/conversations/{conversation_id}")

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            True if deleted
        """
        self._request("DELETE", f"/conversations/{conversation_id}")
        return True

    # ========== Knowledge ==========

    def search_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[KnowledgeResult]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            category: Optional category filter
            limit: Max results

        Returns:
            List of KnowledgeResult objects
        """
        data = {"query": query, "limit": limit}
        if category:
            data["category"] = category

        result = self._request("POST", "/knowledge/search", data)

        results = []
        for r in result.get("results", []):
            results.append(KnowledgeResult(
                id=r.get("id", ""),
                content=r.get("content", ""),
                category=r.get("category", ""),
                relevance=r.get("relevance", 0.0),
            ))
        return results

    # ========== Context Manager ==========

    def conversation(self) -> "ConversationContext":
        """
        Create a conversation context manager.

        Usage:
            with client.conversation() as conv:
                conv.ask("First question")
                conv.ask("Follow-up question")
        """
        return ConversationContext(self)


class ConversationContext:
    """Context manager for multi-turn conversations."""

    def __init__(self, client: CADAssistant):
        self.client = client
        self.conversation_id: Optional[str] = None
        self.messages: List[Message] = []

    def __enter__(self) -> "ConversationContext":
        conv = self.client.create_conversation()
        self.conversation_id = conv.id
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def ask(self, query: str) -> AskResponse:
        """Ask a question in this conversation."""
        response = self.client.ask(query, self.conversation_id)
        self.messages.append(Message("user", query, time.time()))
        self.messages.append(Message("assistant", response.answer, time.time()))
        return response


# Convenience function
def create_client(
    api_key: str,
    base_url: Optional[str] = None,
    **kwargs
) -> CADAssistant:
    """
    Create a CAD Assistant client.

    Args:
        api_key: Your API key
        base_url: Optional base URL
        **kwargs: Additional client options

    Returns:
        CADAssistant client instance
    """
    return CADAssistant(api_key, base_url, **kwargs)


__all__ = [
    "CADAssistant",
    "CADAssistantError",
    "RateLimitError",
    "AuthenticationError",
    "ValidationError",
    "AskResponse",
    "Conversation",
    "Message",
    "KnowledgeResult",
    "Source",
    "ConversationContext",
    "create_client",
]
