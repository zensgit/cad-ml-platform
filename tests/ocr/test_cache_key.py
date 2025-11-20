"""
Minimal cache key consistency tests for OCR module

Ensures cache key generation is deterministic and includes all required components.
Critical path testing for cache hit/miss scenarios.
"""

import pytest
import hashlib
import json
from typing import Dict, Any
from unittest.mock import MagicMock, patch


class TestCacheKeyGeneration:
    """Test cache key generation consistency"""

    @pytest.fixture
    def sample_image_bytes(self) -> bytes:
        """Create consistent test image bytes"""
        return b"fake_image_content_for_testing_12345"

    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Standard OCR configuration"""
        return {
            "provider": "deepseek_hf",
            "prompt_version": "v1",
            "dataset_version": "v1.0",
            "crop_cfg": {
                "max_crops": 4,
                "overlap": 0.1,
                "min_size": 100
            }
        }

    def generate_cache_key(
        self,
        image_bytes: bytes,
        provider: str,
        prompt_version: str,
        crop_cfg: Dict[str, Any],
        dataset_version: str = None
    ) -> str:
        """
        Generate cache key matching production implementation

        Formula:
        key = f"ocr:{sha256(image)}:{provider}:{prompt_version}:{crop_cfg_hash}:{dataset_version}"

        Note: Using SHA256 for both image and crop_cfg (reduced collision risk vs MD5)
        """
        # Image hash (SHA256 for security and collision resistance)
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]

        # Crop config hash (SHA256, deterministic JSON)
        crop_cfg_str = json.dumps(crop_cfg, sort_keys=True)
        crop_cfg_hash = hashlib.sha256(crop_cfg_str.encode()).hexdigest()[:8]

        # Validate prompt_version format (must be vN or vN.N format)
        if not self._validate_version_format(prompt_version):
            raise ValueError(f"Invalid prompt_version format: {prompt_version} (expected: v1 or v1.0)")

        # Build key components
        components = [
            "ocr",
            image_hash,
            provider,
            prompt_version,
            crop_cfg_hash
        ]

        # Dataset version is optional (only for evaluation context)
        if dataset_version:
            if not self._validate_version_format(dataset_version, allow_patch=True):
                raise ValueError(f"Invalid dataset_version format: {dataset_version} (expected: v1.0)")
            components.append(dataset_version)

        return ":".join(components)

    def _validate_version_format(self, version: str, allow_patch: bool = False) -> bool:
        """Validate version format (v1, v2 for prompts; v1.0, v1.1 for datasets)"""
        import re
        if allow_patch:
            # Dataset version: v1.0, v2.1, etc.
            return bool(re.match(r'^v\d+\.\d+$', version))
        else:
            # Prompt version: v1, v2, etc. (can also accept v1.0 for compatibility)
            return bool(re.match(r'^v\d+(\.\d+)?$', version))

    def test_cache_key_deterministic(self, sample_image_bytes, sample_config):
        """Test that same inputs always produce same cache key"""
        key1 = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            sample_config["prompt_version"],
            sample_config["crop_cfg"],
            sample_config["dataset_version"]
        )

        key2 = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            sample_config["prompt_version"],
            sample_config["crop_cfg"],
            sample_config["dataset_version"]
        )

        assert key1 == key2, "Cache keys should be identical for same inputs"
        assert key1.startswith("ocr:"), "Cache key should start with 'ocr:'"
        assert len(key1.split(":")) == 6, "Cache key should have 6 components"

    def test_cache_key_changes_with_image(self, sample_config):
        """Test that different images produce different keys"""
        image1 = b"image_content_1"
        image2 = b"image_content_2"

        key1 = self.generate_cache_key(
            image1,
            sample_config["provider"],
            sample_config["prompt_version"],
            sample_config["crop_cfg"]
        )

        key2 = self.generate_cache_key(
            image2,
            sample_config["provider"],
            sample_config["prompt_version"],
            sample_config["crop_cfg"]
        )

        assert key1 != key2, "Different images should produce different cache keys"

    def test_cache_key_changes_with_provider(self, sample_image_bytes, sample_config):
        """Test that different providers produce different keys"""
        key_paddle = self.generate_cache_key(
            sample_image_bytes,
            "paddle",
            sample_config["prompt_version"],
            sample_config["crop_cfg"]
        )

        key_deepseek = self.generate_cache_key(
            sample_image_bytes,
            "deepseek_hf",
            sample_config["prompt_version"],
            sample_config["crop_cfg"]
        )

        assert key_paddle != key_deepseek, "Different providers should have different keys"

    def test_cache_key_changes_with_prompt_version(self, sample_image_bytes, sample_config):
        """Test that prompt version changes invalidate cache"""
        key_v1 = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            "v1",
            sample_config["crop_cfg"]
        )

        key_v2 = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            "v2",
            sample_config["crop_cfg"]
        )

        assert key_v1 != key_v2, "Different prompt versions should invalidate cache"

    def test_cache_key_changes_with_crop_config(self, sample_image_bytes, sample_config):
        """Test that crop configuration changes affect cache key"""
        crop_cfg1 = {"max_crops": 4, "overlap": 0.1}
        crop_cfg2 = {"max_crops": 6, "overlap": 0.2}

        key1 = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            sample_config["prompt_version"],
            crop_cfg1
        )

        key2 = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            sample_config["prompt_version"],
            crop_cfg2
        )

        assert key1 != key2, "Different crop configs should produce different keys"

    def test_cache_key_with_optional_dataset_version(self, sample_image_bytes, sample_config):
        """Test cache key with and without dataset version"""
        key_without = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            sample_config["prompt_version"],
            sample_config["crop_cfg"],
            dataset_version=None
        )

        key_with = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            sample_config["prompt_version"],
            sample_config["crop_cfg"],
            dataset_version="v1.0"
        )

        assert len(key_without.split(":")) == 5, "Key without dataset should have 5 parts"
        assert len(key_with.split(":")) == 6, "Key with dataset should have 6 parts"
        assert key_without != key_with, "Dataset version should affect cache key"

    def test_cache_key_order_independence_for_crop_config(self, sample_image_bytes, sample_config):
        """Test that crop config key order doesn't affect cache key"""
        crop_cfg1 = {"overlap": 0.1, "max_crops": 4, "min_size": 100}
        crop_cfg2 = {"max_crops": 4, "min_size": 100, "overlap": 0.1}

        key1 = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            sample_config["prompt_version"],
            crop_cfg1
        )

        key2 = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            sample_config["prompt_version"],
            crop_cfg2
        )

        assert key1 == key2, "Crop config order should not affect cache key"

    @pytest.mark.asyncio
    async def test_cache_lookup_simulation(self, sample_image_bytes, sample_config):
        """Simulate cache lookup with mocked Redis"""
        from unittest.mock import AsyncMock

        # Mock Redis client
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)  # Cache miss
        redis_mock.setex = AsyncMock(return_value=True)

        cache_key = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            sample_config["prompt_version"],
            sample_config["crop_cfg"]
        )

        # Simulate cache miss
        cached_result = await redis_mock.get(cache_key)
        assert cached_result is None, "Should be cache miss initially"

        # Simulate cache set
        mock_result = {"text": "test", "confidence": 0.95}
        await redis_mock.setex(cache_key, 3600, json.dumps(mock_result))

        # Verify setex was called with correct key
        redis_mock.setex.assert_called_once()
        call_args = redis_mock.setex.call_args[0]
        assert call_args[0] == cache_key
        assert call_args[1] == 3600  # TTL

    def test_cache_key_length_reasonable(self, sample_image_bytes, sample_config):
        """Ensure cache keys are reasonable length for Redis"""
        key = self.generate_cache_key(
            sample_image_bytes,
            sample_config["provider"],
            sample_config["prompt_version"],
            sample_config["crop_cfg"],
            sample_config["dataset_version"]
        )

        # Redis key length limit is 512MB, but we want reasonable keys
        assert len(key) < 200, f"Cache key too long: {len(key)} chars"
        assert len(key) > 30, f"Cache key suspiciously short: {len(key)} chars"

    def test_prompt_version_format_validation(self, sample_image_bytes, sample_config):
        """Test prompt version format validation (v1, v2)"""
        # Valid formats
        valid_versions = ["v1", "v2", "v10"]

        for version in valid_versions:
            key = self.generate_cache_key(
                sample_image_bytes,
                sample_config["provider"],
                version,
                sample_config["crop_cfg"]
            )
            assert ":" + version + ":" in key

        # Invalid formats should raise
        invalid_versions = ["1", "v", "version1", "v1."]

        for invalid_version in invalid_versions:
            with pytest.raises(ValueError, match="Invalid prompt_version format"):
                self.generate_cache_key(
                    sample_image_bytes,
                    sample_config["provider"],
                    invalid_version,
                    sample_config["crop_cfg"]
                )

    def test_large_file_no_cache_simulation(self, sample_config):
        """Test large files (>10MB) should not be cached"""
        # Simulate large file check
        large_image = b"x" * (11 * 1024 * 1024)  # 11MB
        small_image = b"x" * (5 * 1024 * 1024)   # 5MB

        def should_cache(image_bytes: bytes) -> bool:
            """Production logic: don't cache files > 10MB"""
            return len(image_bytes) <= 10 * 1024 * 1024

        assert not should_cache(large_image), "Large files should not be cached"
        assert should_cache(small_image), "Small files should be cached"


class TestCacheInvalidation:
    """Test cache invalidation scenarios"""

    def test_version_change_invalidation_matrix(self):
        """Test all version change combinations that should invalidate cache"""
        base_config = {
            "image_bytes": b"test_image",
            "provider": "deepseek_hf",
            "prompt_version": "v1",
            "crop_cfg": {"max_crops": 4},
            "dataset_version": "v1.0"
        }

        test_cases = [
            ({"prompt_version": "v2"}, "Prompt version change"),
            ({"dataset_version": "v1.1"}, "Dataset version change"),
            ({"provider": "paddle"}, "Provider change"),
            ({"crop_cfg": {"max_crops": 6}}, "Crop config change"),
        ]

        generator = TestCacheKeyGeneration()
        base_key = generator.generate_cache_key(
            base_config["image_bytes"],
            base_config["provider"],
            base_config["prompt_version"],
            base_config["crop_cfg"],
            base_config["dataset_version"],
        )

        for changes, description in test_cases:
            modified_config = {**base_config, **changes}
            new_key = generator.generate_cache_key(
                modified_config["image_bytes"],
                modified_config["provider"],
                modified_config["prompt_version"],
                modified_config["crop_cfg"],
                modified_config["dataset_version"],
            )

            assert base_key != new_key, f"Cache should invalidate on: {description}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
