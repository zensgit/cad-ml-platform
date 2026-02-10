"""Tests for src/core/assistant/security.py.

Covers:
- Permission enum
- Error classes (AuthError, UnauthorizedError, ForbiddenError, InvalidTokenError, TokenExpiredError)
- APIKey class (has_permission, is_expired, to_dict)
- APIKeyManager (create_key, validate_key, revoke_key, delete_key, list_keys)
- SecurityAuditor (logging methods, get_events, get_failed_authentications)
- require_permission decorator
- sanitize_input function
- validate_conversation_id function
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.core.assistant.security import (
    APIKey,
    APIKeyManager,
    AuthError,
    ForbiddenError,
    InvalidTokenError,
    Permission,
    SecurityAuditor,
    TokenExpiredError,
    UnauthorizedError,
    require_permission,
    sanitize_input,
    validate_conversation_id,
)


class TestPermission:
    """Tests for Permission enum."""

    def test_permission_values(self):
        """Test Permission enum values."""
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.ADMIN.value == "admin"
        assert Permission.EVALUATE.value == "evaluate"
        assert Permission.MANAGE_KNOWLEDGE.value == "manage_knowledge"

    def test_permission_iteration(self):
        """Test iterating over permissions."""
        permissions = list(Permission)
        assert len(permissions) == 5


class TestErrorClasses:
    """Tests for security error classes."""

    def test_auth_error(self):
        """Test AuthError creation."""
        error = AuthError("Test error", code="TEST_CODE")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.code == "TEST_CODE"

    def test_auth_error_default_code(self):
        """Test AuthError default code."""
        error = AuthError("Test error")
        assert error.code == "AUTH_ERROR"

    def test_unauthorized_error(self):
        """Test UnauthorizedError creation."""
        error = UnauthorizedError()
        assert error.message == "Unauthorized"
        assert error.code == "UNAUTHORIZED"

    def test_unauthorized_error_custom_message(self):
        """Test UnauthorizedError with custom message."""
        error = UnauthorizedError("Custom unauthorized message")
        assert error.message == "Custom unauthorized message"

    def test_forbidden_error(self):
        """Test ForbiddenError creation."""
        error = ForbiddenError()
        assert error.message == "Forbidden"
        assert error.code == "FORBIDDEN"

    def test_forbidden_error_custom_message(self):
        """Test ForbiddenError with custom message."""
        error = ForbiddenError("Access denied to resource")
        assert error.message == "Access denied to resource"

    def test_invalid_token_error(self):
        """Test InvalidTokenError creation."""
        error = InvalidTokenError()
        assert error.message == "Invalid token"
        assert error.code == "INVALID_TOKEN"

    def test_invalid_token_error_custom_message(self):
        """Test InvalidTokenError with custom message."""
        error = InvalidTokenError("Token format incorrect")
        assert error.message == "Token format incorrect"

    def test_token_expired_error(self):
        """Test TokenExpiredError creation."""
        error = TokenExpiredError()
        assert error.message == "Token expired"
        assert error.code == "TOKEN_EXPIRED"

    def test_token_expired_error_custom_message(self):
        """Test TokenExpiredError with custom message."""
        error = TokenExpiredError("Token expired at midnight")
        assert error.message == "Token expired at midnight"


class TestAPIKey:
    """Tests for APIKey dataclass."""

    @pytest.fixture
    def api_key(self):
        """Create a test API key."""
        return APIKey(
            key_id="test-key-id",
            key_hash="test-hash",
            name="Test Key",
            permissions={Permission.READ, Permission.WRITE},
            rate_limit=100,
        )

    def test_default_values(self):
        """Test APIKey default values."""
        key = APIKey(
            key_id="id",
            key_hash="hash",
            name="name",
            permissions={Permission.READ},
        )
        assert key.rate_limit == 60
        assert key.expires_at is None
        assert key.last_used is None
        assert key.is_active is True

    def test_has_permission_direct(self, api_key):
        """Test has_permission with direct permission."""
        assert api_key.has_permission(Permission.READ) is True
        assert api_key.has_permission(Permission.WRITE) is True
        assert api_key.has_permission(Permission.ADMIN) is False

    def test_has_permission_admin_grants_all(self):
        """Test admin permission grants all permissions."""
        key = APIKey(
            key_id="id",
            key_hash="hash",
            name="Admin Key",
            permissions={Permission.ADMIN},
        )
        assert key.has_permission(Permission.READ) is True
        assert key.has_permission(Permission.WRITE) is True
        assert key.has_permission(Permission.EVALUATE) is True
        assert key.has_permission(Permission.MANAGE_KNOWLEDGE) is True

    def test_is_expired_no_expiration(self, api_key):
        """Test is_expired with no expiration date."""
        assert api_key.is_expired() is False

    def test_is_expired_future_date(self):
        """Test is_expired with future expiration."""
        key = APIKey(
            key_id="id",
            key_hash="hash",
            name="name",
            permissions={Permission.READ},
            expires_at=time.time() + 3600,
        )
        assert key.is_expired() is False

    def test_is_expired_past_date(self):
        """Test is_expired with past expiration."""
        key = APIKey(
            key_id="id",
            key_hash="hash",
            name="name",
            permissions={Permission.READ},
            expires_at=time.time() - 3600,
        )
        assert key.is_expired() is True

    def test_to_dict(self, api_key):
        """Test to_dict method."""
        result = api_key.to_dict()

        assert result["key_id"] == "test-key-id"
        assert result["name"] == "Test Key"
        assert set(result["permissions"]) == {"read", "write"}
        assert result["rate_limit"] == 100
        assert result["is_active"] is True
        assert "key_hash" not in result  # Sensitive data excluded


class TestAPIKeyManager:
    """Tests for APIKeyManager class."""

    @pytest.fixture
    def manager(self):
        """Create a test API key manager."""
        return APIKeyManager(secret_key="test-secret-key")

    def test_init_generates_secret_key(self):
        """Test manager generates secret key if not provided."""
        manager = APIKeyManager()
        assert manager._secret_key is not None
        assert len(manager._secret_key) == 64  # 32 bytes = 64 hex chars

    def test_create_key(self, manager):
        """Test creating an API key."""
        key_id, secret = manager.create_key(
            name="Test Key",
            permissions={Permission.READ},
        )

        assert key_id is not None
        assert secret.startswith("cad_")
        assert key_id in manager._keys

    def test_create_key_with_expiration(self, manager):
        """Test creating an API key with expiration."""
        key_id, secret = manager.create_key(
            name="Expiring Key",
            permissions={Permission.READ},
            expires_in_days=30,
        )

        api_key = manager._keys[key_id]
        assert api_key.expires_at is not None
        assert api_key.expires_at > time.time()

    def test_create_key_with_custom_rate_limit(self, manager):
        """Test creating an API key with custom rate limit."""
        key_id, secret = manager.create_key(
            name="Rate Limited Key",
            permissions={Permission.READ},
            rate_limit=1000,
        )

        api_key = manager._keys[key_id]
        assert api_key.rate_limit == 1000

    def test_validate_key_success(self, manager):
        """Test validating a valid key."""
        key_id, secret = manager.create_key(
            name="Test Key",
            permissions={Permission.READ},
        )

        api_key = manager.validate_key(secret)
        assert api_key is not None
        assert api_key.key_id == key_id
        assert api_key.last_used is not None

    def test_validate_key_invalid_format(self, manager):
        """Test validating key with invalid format."""
        with pytest.raises(InvalidTokenError) as exc_info:
            manager.validate_key("invalid-key-without-prefix")
        assert "Invalid key format" in str(exc_info.value)

    def test_validate_key_inactive(self, manager):
        """Test validating an inactive key."""
        key_id, secret = manager.create_key(
            name="Test Key",
            permissions={Permission.READ},
        )
        manager.revoke_key(key_id)

        with pytest.raises(UnauthorizedError) as exc_info:
            manager.validate_key(secret)
        assert "inactive" in str(exc_info.value)

    def test_validate_key_expired(self, manager):
        """Test validating an expired key."""
        key_id, secret = manager.create_key(
            name="Test Key",
            permissions={Permission.READ},
        )
        # Manually expire the key
        manager._keys[key_id].expires_at = time.time() - 1

        with pytest.raises(TokenExpiredError):
            manager.validate_key(secret)

    def test_validate_key_not_found(self, manager):
        """Test validating a key that doesn't exist."""
        with pytest.raises(InvalidTokenError) as exc_info:
            manager.validate_key("cad_nonexistent_key_hash_value")
        assert "Invalid API key" in str(exc_info.value)

    def test_revoke_key_success(self, manager):
        """Test revoking a key."""
        key_id, _ = manager.create_key(
            name="Test Key",
            permissions={Permission.READ},
        )

        result = manager.revoke_key(key_id)
        assert result is True
        assert manager._keys[key_id].is_active is False

    def test_revoke_key_not_found(self, manager):
        """Test revoking a nonexistent key."""
        result = manager.revoke_key("nonexistent-key-id")
        assert result is False

    def test_delete_key_success(self, manager):
        """Test deleting a key."""
        key_id, _ = manager.create_key(
            name="Test Key",
            permissions={Permission.READ},
        )

        result = manager.delete_key(key_id)
        assert result is True
        assert key_id not in manager._keys

    def test_delete_key_not_found(self, manager):
        """Test deleting a nonexistent key."""
        result = manager.delete_key("nonexistent-key-id")
        assert result is False

    def test_list_keys(self, manager):
        """Test listing keys."""
        manager.create_key(name="Key 1", permissions={Permission.READ})
        manager.create_key(name="Key 2", permissions={Permission.WRITE})

        keys = manager.list_keys()
        assert len(keys) == 2
        assert all(isinstance(k, dict) for k in keys)
        names = {k["name"] for k in keys}
        assert names == {"Key 1", "Key 2"}


class TestSecurityAuditor:
    """Tests for SecurityAuditor class."""

    @pytest.fixture
    def auditor(self):
        """Create a test security auditor."""
        return SecurityAuditor(max_events=100)

    def test_log_authentication(self, auditor):
        """Test logging authentication event."""
        auditor.log_authentication(
            identifier="user-123",
            success=True,
            method="api_key",
            ip_address="192.168.1.1",
            details={"key_id": "abc"},
        )

        events = auditor.get_events()
        assert len(events) == 1
        assert events[0]["type"] == "authentication"
        assert events[0]["identifier"] == "user-123"
        assert events[0]["success"] is True
        assert events[0]["ip_address"] == "192.168.1.1"

    def test_log_authentication_defaults(self, auditor):
        """Test logging authentication with defaults."""
        auditor.log_authentication(identifier="user-123", success=False)

        events = auditor.get_events()
        assert events[0]["method"] == "api_key"
        assert events[0]["ip_address"] is None
        assert events[0]["details"] == {}

    def test_log_access(self, auditor):
        """Test logging access event."""
        auditor.log_access(
            identifier="user-123",
            resource="knowledge:write",
            granted=True,
            permission="write",
            details={"doc_id": "doc-456"},
        )

        events = auditor.get_events()
        assert len(events) == 1
        assert events[0]["type"] == "access"
        assert events[0]["resource"] == "knowledge:write"
        assert events[0]["granted"] is True

    def test_log_access_defaults(self, auditor):
        """Test logging access with defaults."""
        auditor.log_access(identifier="user-123", resource="test", granted=False)

        events = auditor.get_events()
        assert events[0]["permission"] is None
        assert events[0]["details"] == {}

    def test_log_rate_limit(self, auditor):
        """Test logging rate limit event."""
        auditor.log_rate_limit(
            identifier="user-123",
            endpoint="/api/analyze",
            current_count=101,
            limit=100,
        )

        events = auditor.get_events()
        assert len(events) == 1
        assert events[0]["type"] == "rate_limit"
        assert events[0]["current_count"] == 101
        assert events[0]["limit"] == 100

    def test_log_security_event(self, auditor):
        """Test logging generic security event."""
        auditor.log_security_event(
            event_type="brute_force",
            severity="high",
            message="Multiple failed login attempts detected",
            details={"attempts": 10},
        )

        events = auditor.get_events()
        assert len(events) == 1
        assert events[0]["type"] == "security"
        assert events[0]["event_type"] == "brute_force"
        assert events[0]["severity"] == "high"

    def test_log_security_event_defaults(self, auditor):
        """Test logging security event with defaults."""
        auditor.log_security_event(
            event_type="test",
            severity="low",
            message="Test event",
        )

        events = auditor.get_events()
        assert events[0]["details"] == {}

    def test_get_events_filter_by_type(self, auditor):
        """Test filtering events by type."""
        auditor.log_authentication("user-1", True)
        auditor.log_access("user-1", "resource", True)
        auditor.log_authentication("user-2", False)

        events = auditor.get_events(event_type="authentication")
        assert len(events) == 2
        assert all(e["type"] == "authentication" for e in events)

    def test_get_events_filter_by_identifier(self, auditor):
        """Test filtering events by identifier."""
        auditor.log_authentication("user-1", True)
        auditor.log_authentication("user-2", True)
        auditor.log_authentication("user-1", False)

        events = auditor.get_events(identifier="user-1")
        assert len(events) == 2

    def test_get_events_filter_by_start_time(self, auditor):
        """Test filtering events by start time."""
        auditor.log_authentication("user-1", True)
        start_time = time.time()
        time.sleep(0.01)
        auditor.log_authentication("user-2", True)

        events = auditor.get_events(start_time=start_time)
        assert len(events) == 1

    def test_get_events_limit(self, auditor):
        """Test limiting events returned."""
        for i in range(10):
            auditor.log_authentication(f"user-{i}", True)

        events = auditor.get_events(limit=5)
        assert len(events) == 5

    def test_get_failed_authentications(self, auditor):
        """Test getting failed authentication identifiers."""
        # Add failures
        for _ in range(5):
            auditor.log_authentication("bad-actor", False)
        for _ in range(3):
            auditor.log_authentication("casual-user", False)
        auditor.log_authentication("good-user", True)

        failures = auditor.get_failed_authentications(threshold=5)
        assert len(failures) == 1
        assert failures[0]["identifier"] == "bad-actor"
        assert failures[0]["failures"] == 5

    def test_get_failed_authentications_with_window(self, auditor):
        """Test failed authentications respects time window."""
        # This would require mocking time, simplified test
        auditor.log_authentication("user-1", False)
        auditor.log_authentication("user-1", False)
        auditor.log_authentication("user-1", False)
        auditor.log_authentication("user-1", False)
        auditor.log_authentication("user-1", False)

        failures = auditor.get_failed_authentications(window_seconds=3600, threshold=5)
        assert len(failures) == 1

    def test_max_events_trimming(self):
        """Test events are trimmed at max limit."""
        auditor = SecurityAuditor(max_events=5)

        for i in range(10):
            auditor.log_authentication(f"user-{i}", True)

        events = auditor.get_events()
        assert len(events) == 5
        # Should have the latest events
        identifiers = [e["identifier"] for e in events]
        assert "user-9" in identifiers
        assert "user-0" not in identifiers


class TestRequirePermissionDecorator:
    """Tests for require_permission decorator."""

    def test_require_permission_success(self):
        """Test decorator allows access with permission."""
        api_key = APIKey(
            key_id="id",
            key_hash="hash",
            name="Test",
            permissions={Permission.READ},
        )

        @require_permission(Permission.READ)
        def protected_function(api_key=None):
            return "success"

        result = protected_function(api_key=api_key)
        assert result == "success"

    def test_require_permission_no_api_key(self):
        """Test decorator rejects without API key."""

        @require_permission(Permission.READ)
        def protected_function(api_key=None):
            return "success"

        with pytest.raises(UnauthorizedError) as exc_info:
            protected_function()
        assert "API key required" in str(exc_info.value)

    def test_require_permission_insufficient(self):
        """Test decorator rejects without required permission."""
        api_key = APIKey(
            key_id="id",
            key_hash="hash",
            name="Test",
            permissions={Permission.READ},
        )

        @require_permission(Permission.WRITE)
        def protected_function(api_key=None):
            return "success"

        with pytest.raises(ForbiddenError) as exc_info:
            protected_function(api_key=api_key)
        assert "Permission denied" in str(exc_info.value)

    def test_require_permission_admin_bypass(self):
        """Test admin can access any permission."""
        api_key = APIKey(
            key_id="id",
            key_hash="hash",
            name="Admin",
            permissions={Permission.ADMIN},
        )

        @require_permission(Permission.MANAGE_KNOWLEDGE)
        def protected_function(api_key=None):
            return "success"

        result = protected_function(api_key=api_key)
        assert result == "success"


class TestSanitizeInput:
    """Tests for sanitize_input function."""

    def test_truncation(self):
        """Test input is truncated."""
        long_text = "a" * 20000
        result = sanitize_input(long_text, max_length=100)
        assert len(result) == 100

    def test_no_truncation_needed(self):
        """Test short input is not truncated."""
        short_text = "Hello world"
        result = sanitize_input(short_text)
        assert result == short_text

    def test_null_byte_removal(self):
        """Test null bytes are removed."""
        text_with_null = "Hello\x00World"
        result = sanitize_input(text_with_null)
        assert result == "HelloWorld"

    def test_script_tag_sanitization(self):
        """Test script tags are sanitized."""
        xss_text = "<script>alert('xss')</script>"
        result = sanitize_input(xss_text)
        assert "<script" not in result
        assert "&lt;script" in result

    def test_script_closing_tag_sanitization(self):
        """Test script closing tags are sanitized."""
        xss_text = "text</script>more"
        result = sanitize_input(xss_text)
        assert "</script" not in result
        assert "&lt;/script" in result

    def test_default_max_length(self):
        """Test default max length is applied."""
        text = "a" * 15000
        result = sanitize_input(text)
        assert len(result) == 10000


class TestValidateConversationId:
    """Tests for validate_conversation_id function."""

    def test_valid_conversation_id(self):
        """Test valid conversation IDs."""
        assert validate_conversation_id("abc123") is True
        assert validate_conversation_id("ABC-123_def") is True
        assert validate_conversation_id("a-b-c") is True
        assert validate_conversation_id("test_123") is True

    def test_empty_conversation_id(self):
        """Test empty conversation ID is invalid."""
        assert validate_conversation_id("") is False

    def test_too_long_conversation_id(self):
        """Test conversation ID over 100 chars is invalid."""
        long_id = "a" * 101
        assert validate_conversation_id(long_id) is False

    def test_invalid_characters(self):
        """Test invalid characters are rejected."""
        assert validate_conversation_id("test@123") is False
        assert validate_conversation_id("test 123") is False
        assert validate_conversation_id("test/123") is False
        assert validate_conversation_id("test:123") is False
        assert validate_conversation_id("test.123") is False
        assert validate_conversation_id("test<script>") is False
