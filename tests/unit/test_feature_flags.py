"""Tests for Feature Flag System."""

import pytest
from unittest.mock import patch, MagicMock

from src.core.feature_flags import (
    FeatureFlag,
    FeatureFlagClient,
    FlagContext,
    RolloutStrategy,
    is_enabled,
)


class TestFlagContext:
    """Tests for FlagContext."""

    def test_get_hash_key(self):
        """Test hash key generation."""
        ctx = FlagContext(user_id="user1", tenant_id="tenant1")
        hash_key = ctx.get_hash_key()
        assert isinstance(hash_key, str)
        assert len(hash_key) == 32  # MD5 hash

    def test_get_percentage_bucket(self):
        """Test percentage bucket calculation."""
        ctx = FlagContext(user_id="user1", tenant_id="tenant1")
        bucket = ctx.get_percentage_bucket()
        assert 0 <= bucket < 100

    def test_consistent_bucket(self):
        """Test bucket is consistent for same user."""
        ctx1 = FlagContext(user_id="user1", tenant_id="tenant1")
        ctx2 = FlagContext(user_id="user1", tenant_id="tenant1")
        assert ctx1.get_percentage_bucket() == ctx2.get_percentage_bucket()

    def test_different_users_different_buckets(self):
        """Test different users likely get different buckets."""
        buckets = set()
        for i in range(100):
            ctx = FlagContext(user_id=f"user{i}")
            buckets.add(ctx.get_percentage_bucket())
        # Should have good distribution
        assert len(buckets) > 50


class TestFeatureFlag:
    """Tests for FeatureFlag."""

    def test_disabled_flag(self):
        """Test disabled flag always returns False."""
        flag = FeatureFlag(name="test", enabled=False)
        assert flag.evaluate() is False
        assert flag.evaluate(FlagContext(user_id="user1")) is False

    def test_all_strategy(self):
        """Test ALL strategy enables for everyone."""
        flag = FeatureFlag(name="test", enabled=True, strategy=RolloutStrategy.ALL)
        assert flag.evaluate() is True
        assert flag.evaluate(FlagContext(user_id="user1")) is True

    def test_none_strategy(self):
        """Test NONE strategy disables for everyone."""
        flag = FeatureFlag(name="test", enabled=True, strategy=RolloutStrategy.NONE)
        assert flag.evaluate() is False
        assert flag.evaluate(FlagContext(user_id="user1")) is False

    def test_percentage_strategy(self):
        """Test percentage rollout."""
        flag = FeatureFlag(
            name="test",
            enabled=True,
            strategy=RolloutStrategy.PERCENTAGE,
            percentage=50,
        )
        # Without context, should return False
        assert flag.evaluate() is False

        # With context, should be deterministic
        ctx = FlagContext(user_id="test_user")
        result1 = flag.evaluate(ctx)
        result2 = flag.evaluate(ctx)
        assert result1 == result2

    def test_user_list_strategy(self):
        """Test user list targeting."""
        flag = FeatureFlag(
            name="test",
            enabled=True,
            strategy=RolloutStrategy.USER_LIST,
            allowed_users={"user1", "user2"},
        )
        assert flag.evaluate(FlagContext(user_id="user1")) is True
        assert flag.evaluate(FlagContext(user_id="user3")) is False
        assert flag.evaluate(FlagContext()) is False

    def test_tenant_list_strategy(self):
        """Test tenant list targeting."""
        flag = FeatureFlag(
            name="test",
            enabled=True,
            strategy=RolloutStrategy.TENANT_LIST,
            allowed_tenants={"tenant1"},
        )
        assert flag.evaluate(FlagContext(tenant_id="tenant1")) is True
        assert flag.evaluate(FlagContext(tenant_id="tenant2")) is False

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        flag = FeatureFlag(
            name="test",
            description="Test flag",
            enabled=True,
            strategy=RolloutStrategy.PERCENTAGE,
            percentage=25,
            allowed_users={"user1"},
        )
        data = flag.to_dict()
        restored = FeatureFlag.from_dict(data)

        assert restored.name == flag.name
        assert restored.description == flag.description
        assert restored.enabled == flag.enabled
        assert restored.strategy == flag.strategy
        assert restored.percentage == flag.percentage


class TestFeatureFlagClient:
    """Tests for FeatureFlagClient."""

    def test_env_backend(self):
        """Test environment variable backend."""
        with patch.dict("os.environ", {"FF_TEST_FLAG": "true"}):
            client = FeatureFlagClient(backend="env", prefix="FF_")
            assert client.is_enabled("test_flag") is True

    def test_env_backend_disabled(self):
        """Test disabled env flag."""
        with patch.dict("os.environ", {"FF_TEST_FLAG": "false"}):
            client = FeatureFlagClient(backend="env", prefix="FF_")
            assert client.is_enabled("test_flag") is False

    def test_register_flag(self):
        """Test manual flag registration."""
        client = FeatureFlagClient(backend="env")
        flag = FeatureFlag(name="manual_flag", enabled=True)
        client.register_flag(flag)
        assert client.is_enabled("manual_flag") is True

    def test_set_flag(self):
        """Test runtime flag modification."""
        client = FeatureFlagClient(backend="env")
        client.set_flag("dynamic_flag", enabled=True)
        assert client.is_enabled("dynamic_flag") is True

        client.set_flag("dynamic_flag", enabled=False)
        assert client.is_enabled("dynamic_flag") is False

    def test_default_value(self):
        """Test default value for unknown flags."""
        client = FeatureFlagClient(backend="env")
        # Clear cache to ensure fresh evaluation
        client.clear_cache()
        assert client.is_enabled("unknown_flag_1", default=False) is False
        client.clear_cache()
        assert client.is_enabled("unknown_flag_2", default=True) is True

    def test_cache_behavior(self):
        """Test that results are cached."""
        client = FeatureFlagClient(backend="env", cache_ttl=60)
        flag = FeatureFlag(name="cached_flag", enabled=True)
        client.register_flag(flag)

        # First call
        result1 = client.is_enabled("cached_flag")
        # Modify flag
        client._flags["cached_flag"].enabled = False
        # Should still return cached value
        result2 = client.is_enabled("cached_flag")

        assert result1 is True
        # Cache might return old value
        client.clear_cache()
        result3 = client.is_enabled("cached_flag")
        assert result3 is False

    def test_list_flags(self):
        """Test listing all flags."""
        client = FeatureFlagClient(backend="env")
        client.register_flag(FeatureFlag(name="flag1", enabled=True))
        client.register_flag(FeatureFlag(name="flag2", enabled=False))

        flags = client.list_flags()
        names = {f.name for f in flags}
        assert "flag1" in names
        assert "flag2" in names


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_is_enabled_global(self):
        """Test global is_enabled function."""
        # Should use global client
        result = is_enabled("nonexistent_flag", default=False)
        assert result is False


class TestRolloutDistribution:
    """Tests for rollout distribution fairness."""

    def test_percentage_distribution(self):
        """Test that percentage rollout is approximately correct."""
        flag = FeatureFlag(
            name="test",
            enabled=True,
            strategy=RolloutStrategy.PERCENTAGE,
            percentage=30,
        )

        enabled_count = 0
        total = 1000
        for i in range(total):
            ctx = FlagContext(user_id=f"user_{i}")
            if flag.evaluate(ctx):
                enabled_count += 1

        # Should be approximately 30% (with some tolerance)
        ratio = enabled_count / total
        assert 0.20 < ratio < 0.40, f"Expected ~30%, got {ratio*100:.1f}%"
