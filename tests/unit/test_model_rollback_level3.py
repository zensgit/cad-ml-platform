"""Tests for model rollback level 3 functionality.

Verifies that:
1. Snapshot shifting works correctly (PREV -> PREV2 -> PREV3)
2. Level 3 rollback is triggered when levels 1 and 2 are exhausted
3. Model state is correctly restored from level 3 snapshot
4. get_model_info() correctly detects rollback level 3
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch, MagicMock

import pytest


# Disable opcode scanning for these tests to allow valid models to load
@pytest.fixture(autouse=True)
def disable_opcode_scan():
    """Disable opcode scanning for rollback tests."""
    with patch.dict(os.environ, {"MODEL_OPCODE_SCAN": "0"}):
        yield


class MockValidModel:
    """Valid sklearn-like model for testing."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.data = [1, 2, 3]

    def predict(self, X):
        return ["A"] * len(X)


class MockInvalidModel:
    """Invalid model without predict method."""

    def __init__(self):
        self.data = "invalid"


def create_model_file(model: object, path: Path, protocol: int = 4) -> str:
    """Create a pickle file with the given model.

    Returns:
        SHA256 hash prefix (16 chars) of the file
    """
    import hashlib
    with path.open("wb") as f:
        pickle.dump(model, f, protocol=protocol)

    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return file_hash


def create_invalid_pickle(path: Path) -> None:
    """Create a file that will fail to load as valid model (missing predict)."""
    # Create a valid pickle file but with an invalid model (no predict method)
    invalid_model = {"data": "invalid", "no_predict": True}
    with path.open("wb") as f:
        pickle.dump(invalid_model, f, protocol=4)


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def reset_classifier_state():
    """Reset classifier module state before and after each test."""
    import src.ml.classifier as clf

    # Save original state
    original_state = {
        "_MODEL": clf._MODEL,
        "_MODEL_HASH": clf._MODEL_HASH,
        "_MODEL_VERSION": clf._MODEL_VERSION,
        "_MODEL_PATH": clf._MODEL_PATH,
        "_MODEL_LOADED_AT": clf._MODEL_LOADED_AT,
        "_MODEL_LAST_ERROR": clf._MODEL_LAST_ERROR,
        "_MODEL_LOAD_SEQ": clf._MODEL_LOAD_SEQ,
        "_MODEL_PREV": clf._MODEL_PREV,
        "_MODEL_PREV_HASH": clf._MODEL_PREV_HASH,
        "_MODEL_PREV_VERSION": clf._MODEL_PREV_VERSION,
        "_MODEL_PREV_PATH": clf._MODEL_PREV_PATH,
        "_MODEL_PREV2": clf._MODEL_PREV2,
        "_MODEL_PREV2_HASH": clf._MODEL_PREV2_HASH,
        "_MODEL_PREV2_VERSION": clf._MODEL_PREV2_VERSION,
        "_MODEL_PREV2_PATH": clf._MODEL_PREV2_PATH,
        "_MODEL_PREV3": clf._MODEL_PREV3,
        "_MODEL_PREV3_HASH": clf._MODEL_PREV3_HASH,
        "_MODEL_PREV3_VERSION": clf._MODEL_PREV3_VERSION,
        "_MODEL_PREV3_PATH": clf._MODEL_PREV3_PATH,
    }

    # Reset to clean state
    clf._MODEL = None
    clf._MODEL_HASH = None
    clf._MODEL_VERSION = "test"
    clf._MODEL_PATH = Path("test.pkl")
    clf._MODEL_LOADED_AT = None
    clf._MODEL_LAST_ERROR = None
    clf._MODEL_LOAD_SEQ = 0
    clf._MODEL_PREV = None
    clf._MODEL_PREV_HASH = None
    clf._MODEL_PREV_VERSION = None
    clf._MODEL_PREV_PATH = None
    clf._MODEL_PREV2 = None
    clf._MODEL_PREV2_HASH = None
    clf._MODEL_PREV2_VERSION = None
    clf._MODEL_PREV2_PATH = None
    clf._MODEL_PREV3 = None
    clf._MODEL_PREV3_HASH = None
    clf._MODEL_PREV3_VERSION = None
    clf._MODEL_PREV3_PATH = None

    yield

    # Restore original state
    for key, value in original_state.items():
        setattr(clf, key, value)


class TestSnapshotShifting:
    """Test that snapshot shifting works correctly."""

    def test_first_reload_no_prev3(self, temp_model_dir, reset_classifier_state):
        """First reload should not create PREV3."""
        import src.ml.classifier as clf

        # Create and load first model
        model1_path = temp_model_dir / "model1.pkl"
        create_model_file(MockValidModel("v1"), model1_path)

        clf._MODEL = MockValidModel("initial")
        clf._MODEL_HASH = "initialhash"
        clf._MODEL_VERSION = "initial"

        result = clf.reload_model(str(model1_path), expected_version="v1")

        assert result["status"] == "success"
        assert clf._MODEL_PREV is not None
        assert clf._MODEL_PREV2 is None
        assert clf._MODEL_PREV3 is None

    def test_second_reload_creates_prev2_no_prev3(self, temp_model_dir, reset_classifier_state):
        """Second reload should create PREV2 but not PREV3."""
        import src.ml.classifier as clf

        # Setup initial model
        clf._MODEL = MockValidModel("v0")
        clf._MODEL_HASH = "hash_v0"
        clf._MODEL_VERSION = "v0"

        # First reload
        model1_path = temp_model_dir / "model1.pkl"
        create_model_file(MockValidModel("v1"), model1_path)
        clf.reload_model(str(model1_path), expected_version="v1")

        # Second reload
        model2_path = temp_model_dir / "model2.pkl"
        create_model_file(MockValidModel("v2"), model2_path)
        clf.reload_model(str(model2_path), expected_version="v2")

        assert clf._MODEL_PREV is not None
        assert clf._MODEL_PREV2 is not None
        assert clf._MODEL_PREV3 is None

        # Verify the chain
        assert clf._MODEL_PREV.name == "v1"
        assert clf._MODEL_PREV2.name == "v0"

    def test_third_reload_creates_prev3(self, temp_model_dir, reset_classifier_state):
        """Third reload should create PREV3."""
        import src.ml.classifier as clf

        # Setup initial model
        clf._MODEL = MockValidModel("v0")
        clf._MODEL_HASH = "hash_v0"
        clf._MODEL_VERSION = "v0"

        # Three reloads
        for i in range(1, 4):
            model_path = temp_model_dir / f"model{i}.pkl"
            create_model_file(MockValidModel(f"v{i}"), model_path)
            clf.reload_model(str(model_path), expected_version=f"v{i}")

        # After 3 reloads, we should have all snapshot levels
        assert clf._MODEL is not None
        assert clf._MODEL_PREV is not None
        assert clf._MODEL_PREV2 is not None
        assert clf._MODEL_PREV3 is not None

        # Verify the chain: current=v3, PREV=v2, PREV2=v1, PREV3=v0
        assert clf._MODEL.name == "v3"
        assert clf._MODEL_PREV.name == "v2"
        assert clf._MODEL_PREV2.name == "v1"
        assert clf._MODEL_PREV3.name == "v0"

    def test_fourth_reload_shifts_all(self, temp_model_dir, reset_classifier_state):
        """Fourth reload should shift all snapshots, losing the oldest."""
        import src.ml.classifier as clf

        # Setup initial model
        clf._MODEL = MockValidModel("v0")
        clf._MODEL_HASH = "hash_v0"
        clf._MODEL_VERSION = "v0"

        # Four reloads
        for i in range(1, 5):
            model_path = temp_model_dir / f"model{i}.pkl"
            create_model_file(MockValidModel(f"v{i}"), model_path)
            clf.reload_model(str(model_path), expected_version=f"v{i}")

        # Verify: current=v4, PREV=v3, PREV2=v2, PREV3=v1 (v0 is lost)
        assert clf._MODEL.name == "v4"
        assert clf._MODEL_PREV.name == "v3"
        assert clf._MODEL_PREV2.name == "v2"
        assert clf._MODEL_PREV3.name == "v1"


class TestLevel3Rollback:
    """Test level 3 rollback scenarios.

    Note: The reload_model implementation shifts snapshots BEFORE the try block:
    - PREV3 = PREV2 (old)
    - PREV2 = PREV (old)
    - PREV = _MODEL (current)

    On failure, rollback uses the NEW PREV (which was the old current model).
    This means rollback always goes back to what was current before the reload attempt.

    For level 3 rollback to trigger, we need:
    - PREV = None (after shifting)
    - PREV2 = None (after shifting)
    - PREV3 = something (after shifting)
    """

    def test_level3_rollback_on_fourth_failure(self, temp_model_dir, reset_classifier_state):
        """Level 3 rollback triggers when PREV and PREV2 become None after shifting.

        Setup: We need PREV3 to have a value AFTER shifting:
        - _MODEL = None → becomes new PREV (None)
        - PREV = None → becomes new PREV2 (None)
        - PREV2 = v0 → becomes new PREV3 (v0)
        """
        import src.ml.classifier as clf

        v0_model = MockValidModel("v0")
        # Setup so that after shifting, PREV3 will have v0
        clf._MODEL = None  # Will become PREV (None)
        clf._MODEL_PREV = None  # Will become PREV2 (None)
        clf._MODEL_PREV2 = v0_model  # Will become PREV3 (v0)
        clf._MODEL_PREV2_HASH = "hash_v0"
        clf._MODEL_PREV2_VERSION = "v0"
        clf._MODEL_PREV2_PATH = temp_model_dir / "v0.pkl"
        clf._MODEL_PREV3 = None  # Will be overwritten

        # Create invalid model to trigger rollback
        invalid_path = temp_model_dir / "invalid.pkl"
        create_invalid_pickle(invalid_path)

        result = clf.reload_model(str(invalid_path), expected_version="vX")

        assert result["status"] == "rollback_level3"
        assert clf._MODEL == v0_model
        assert clf._MODEL.name == "v0"

    def test_level3_rollback_after_consecutive_failures(self, temp_model_dir, reset_classifier_state):
        """Test level 3 rollback after 3 consecutive reload failures.

        The shifting happens before each reload attempt, so we track the chain
        as it evolves through each failure.
        """
        import src.ml.classifier as clf

        # Setup initial model chain: v0 -> v1 -> v2 -> v3 (current)
        v0_model = MockValidModel("v0")
        v1_model = MockValidModel("v1")
        v2_model = MockValidModel("v2")
        v3_model = MockValidModel("v3")

        clf._MODEL = v3_model
        clf._MODEL_HASH = "hash_v3"
        clf._MODEL_VERSION = "v3"

        clf._MODEL_PREV = v2_model
        clf._MODEL_PREV_HASH = "hash_v2"
        clf._MODEL_PREV_VERSION = "v2"

        clf._MODEL_PREV2 = v1_model
        clf._MODEL_PREV2_HASH = "hash_v1"
        clf._MODEL_PREV2_VERSION = "v1"

        clf._MODEL_PREV3 = v0_model
        clf._MODEL_PREV3_HASH = "hash_v0"
        clf._MODEL_PREV3_VERSION = "v0"

        # First failure:
        # After shifting: PREV=v3, PREV2=v2, PREV3=v1
        # Rollback to PREV (v3) - restores to what was current
        invalid1 = temp_model_dir / "invalid1.pkl"
        create_invalid_pickle(invalid1)
        result1 = clf.reload_model(str(invalid1), expected_version="v4")

        assert result1["status"] == "rollback"
        assert clf._MODEL.name == "v3"  # Restored to previous current

        # After first failure, state is:
        # _MODEL=v3, PREV=v3, PREV2=v2, PREV3=v1
        # Clear PREV to force level 2 rollback on next failure
        clf._MODEL_PREV = None

        # Second failure:
        # After shifting: PREV=v3 (current), PREV2=None (old PREV), PREV3=v2 (old PREV2)
        # PREV is not None, so rollback to level 1
        invalid2 = temp_model_dir / "invalid2.pkl"
        create_invalid_pickle(invalid2)
        result2 = clf.reload_model(str(invalid2), expected_version="v5")

        # This will actually rollback to PREV (v3) since _MODEL was v3
        assert result2["status"] == "rollback"
        assert clf._MODEL.name == "v3"

        # For level 2 rollback to trigger, we need PREV=None after shifting
        # Set _MODEL=None so PREV becomes None after shifting
        clf._MODEL = None
        clf._MODEL_PREV = None  # Will become PREV2
        clf._MODEL_PREV2 = v1_model  # Will become PREV3

        invalid3 = temp_model_dir / "invalid3.pkl"
        create_invalid_pickle(invalid3)
        result3 = clf.reload_model(str(invalid3), expected_version="v6")

        # After shifting: PREV=None, PREV2=None, PREV3=v1
        assert result3["status"] == "rollback_level3"
        assert clf._MODEL.name == "v1"

    def test_no_rollback_when_prev3_empty(self, temp_model_dir, reset_classifier_state):
        """Test error status when all rollback levels are exhausted."""
        import src.ml.classifier as clf

        # Setup with no rollback targets
        clf._MODEL = None
        clf._MODEL_PREV = None
        clf._MODEL_PREV2 = None
        clf._MODEL_PREV3 = None

        # Create invalid model
        invalid_path = temp_model_dir / "invalid.pkl"
        create_invalid_pickle(invalid_path)

        result = clf.reload_model(str(invalid_path), expected_version="vX")

        assert result["status"] == "error"
        assert "error" in result


class TestGetModelInfoLevel3:
    """Test get_model_info() correctly detects level 3 rollback."""

    def test_get_model_info_no_rollback(self, reset_classifier_state):
        """Test get_model_info with no rollback."""
        import src.ml.classifier as clf

        clf._MODEL = MockValidModel("current")
        clf._MODEL_PREV = MockValidModel("prev")

        info = clf.get_model_info()

        assert info["rollback_level"] == 0
        assert info["rollback_reason"] is None
        assert info["has_prev"] is True
        assert info["has_prev2"] is False
        assert info["has_prev3"] is False

    def test_get_model_info_level1_rollback(self, reset_classifier_state):
        """Test get_model_info detects level 1 rollback."""
        import src.ml.classifier as clf

        model = MockValidModel("v1")
        clf._MODEL = model
        clf._MODEL_PREV = model  # Same object = rolled back

        info = clf.get_model_info()

        assert info["rollback_level"] == 1
        assert "previous model" in info["rollback_reason"].lower()

    def test_get_model_info_level2_rollback(self, reset_classifier_state):
        """Test get_model_info detects level 2 rollback."""
        import src.ml.classifier as clf

        model = MockValidModel("v2")
        clf._MODEL = model
        clf._MODEL_PREV = None
        clf._MODEL_PREV2 = model  # Same object = rolled back to level 2

        info = clf.get_model_info()

        assert info["rollback_level"] == 2
        assert "level 2" in info["rollback_reason"].lower()

    def test_get_model_info_level3_rollback(self, reset_classifier_state):
        """Test get_model_info detects level 3 rollback."""
        import src.ml.classifier as clf

        model = MockValidModel("v3")
        clf._MODEL = model
        clf._MODEL_PREV = None
        clf._MODEL_PREV2 = None
        clf._MODEL_PREV3 = model  # Same object = rolled back to level 3

        info = clf.get_model_info()

        assert info["rollback_level"] == 3
        assert "level 3" in info["rollback_reason"].lower()

    def test_get_model_info_has_prev3(self, reset_classifier_state):
        """Test get_model_info correctly reports has_prev3."""
        import src.ml.classifier as clf

        clf._MODEL = MockValidModel("current")
        clf._MODEL_PREV = MockValidModel("prev1")
        clf._MODEL_PREV2 = MockValidModel("prev2")
        clf._MODEL_PREV3 = MockValidModel("prev3")

        info = clf.get_model_info()

        assert info["has_prev"] is True
        assert info["has_prev2"] is True
        assert info["has_prev3"] is True
        assert info["rollback_level"] == 0  # Not currently rolled back


class TestSnapshotVersionTracking:
    """Test that version/hash information is preserved across snapshots."""

    def test_version_preserved_in_prev3(self, temp_model_dir, reset_classifier_state):
        """Test that version is correctly tracked in PREV3."""
        import src.ml.classifier as clf

        # Setup initial model with version
        clf._MODEL = MockValidModel("v0")
        clf._MODEL_HASH = "hash_initial"
        clf._MODEL_VERSION = "version_initial"
        clf._MODEL_PATH = temp_model_dir / "initial.pkl"

        # Perform 3 reloads
        for i in range(1, 4):
            model_path = temp_model_dir / f"model{i}.pkl"
            create_model_file(MockValidModel(f"v{i}"), model_path)
            clf.reload_model(str(model_path), expected_version=f"version_{i}")

        # Verify PREV3 has the original version
        assert clf._MODEL_PREV3_VERSION == "version_initial"
        assert clf._MODEL_PREV3_HASH == "hash_initial"

    def test_hash_preserved_through_rollback(self, temp_model_dir, reset_classifier_state):
        """Test that hash is correctly restored during level 3 rollback.

        Setup: After shifting, PREV3 will have v0's hash:
        - _MODEL=None -> PREV=None
        - PREV=None -> PREV2=None
        - PREV2=v0 -> PREV3=v0
        """
        import src.ml.classifier as clf

        v0_model = MockValidModel("v0")
        # Setup so PREV2 becomes PREV3 after shifting
        clf._MODEL = None
        clf._MODEL_PREV = None
        clf._MODEL_PREV2 = v0_model
        clf._MODEL_PREV2_HASH = "known_hash_v0"
        clf._MODEL_PREV2_VERSION = "v0"
        clf._MODEL_PREV2_PATH = temp_model_dir / "v0.pkl"
        clf._MODEL_PREV3 = None  # Will be overwritten

        # Trigger rollback with invalid file
        invalid_path = temp_model_dir / "invalid.pkl"
        create_invalid_pickle(invalid_path)

        result = clf.reload_model(str(invalid_path))

        assert result["status"] == "rollback_level3"
        assert clf._MODEL_HASH == "known_hash_v0"


class TestRollbackLogging:
    """Test that rollback operations are properly logged."""

    def test_level3_rollback_logs_correctly(self, temp_model_dir, reset_classifier_state, caplog):
        """Test that level 3 rollback produces correct log output.

        Setup: PREV2=v0 will become PREV3 after shifting.
        """
        import src.ml.classifier as clf
        import logging

        v0_model = MockValidModel("v0")
        # Setup so PREV2 becomes PREV3 after shifting
        clf._MODEL = None
        clf._MODEL_PREV = None
        clf._MODEL_PREV2 = v0_model
        clf._MODEL_PREV2_HASH = "hash_v0"
        clf._MODEL_PREV2_VERSION = "v0"
        clf._MODEL_PREV2_PATH = temp_model_dir / "v0.pkl"
        clf._MODEL_PREV3 = None

        # Create invalid model
        invalid_path = temp_model_dir / "invalid.pkl"
        create_invalid_pickle(invalid_path)

        with caplog.at_level(logging.ERROR):
            result = clf.reload_model(str(invalid_path))

        assert result["status"] == "rollback_level3"
        # Check that level 3 rollback was logged
        assert any("level 3" in record.message.lower() for record in caplog.records)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_prev3_path(self, temp_model_dir, reset_classifier_state):
        """Test rollback when PREV3 path is None.

        Setup: PREV2 becomes PREV3 after shifting, with no path.
        """
        import src.ml.classifier as clf

        v0_model = MockValidModel("v0")
        clf._MODEL = None
        clf._MODEL_PREV = None
        clf._MODEL_PREV2 = v0_model
        clf._MODEL_PREV2_HASH = "hash_v0"
        clf._MODEL_PREV2_VERSION = "v0"
        clf._MODEL_PREV2_PATH = None  # No path
        clf._MODEL_PREV3 = None

        invalid_path = temp_model_dir / "invalid.pkl"
        create_invalid_pickle(invalid_path)

        result = clf.reload_model(str(invalid_path))

        # Should still work, using current _MODEL_PATH as fallback
        assert result["status"] == "rollback_level3"

    def test_prev3_version_none_fallback(self, temp_model_dir, reset_classifier_state):
        """Test fallback when PREV3 version is None.

        Setup: PREV2 becomes PREV3 after shifting, with no version.
        """
        import src.ml.classifier as clf

        v0_model = MockValidModel("v0")
        clf._MODEL = None
        clf._MODEL_VERSION = "current_version"
        clf._MODEL_PREV = None
        clf._MODEL_PREV2 = v0_model
        clf._MODEL_PREV2_HASH = "hash_v0"
        clf._MODEL_PREV2_VERSION = None  # No version
        clf._MODEL_PREV2_PATH = temp_model_dir / "v0.pkl"
        clf._MODEL_PREV3 = None

        invalid_path = temp_model_dir / "invalid.pkl"
        create_invalid_pickle(invalid_path)

        result = clf.reload_model(str(invalid_path))

        assert result["status"] == "rollback_level3"
        # Should fallback to current version
        assert clf._MODEL_VERSION == "current_version"

    def test_multiple_reloads_then_rollback(self, temp_model_dir, reset_classifier_state):
        """Test complex scenario: multiple successful reloads then failure.

        After each reload, snapshot shifting occurs:
        - Before reload: current=v0
        - After reload 1: current=v1, PREV=v0
        - After reload 2: current=v2, PREV=v1, PREV2=v0
        - After reload 3: current=v3, PREV=v2, PREV2=v1, PREV3=v0
        - After reload 4: current=v4, PREV=v3, PREV2=v2, PREV3=v1
        - After reload 5: current=v5, PREV=v4, PREV2=v3, PREV3=v2

        On failure attempt:
        - Shifting happens: PREV=v5, PREV2=v4, PREV3=v3
        - Rollback to PREV (v5)
        """
        import src.ml.classifier as clf

        # Setup initial
        clf._MODEL = MockValidModel("v0")
        clf._MODEL_VERSION = "v0"
        clf._MODEL_HASH = "h0"

        # 5 successful reloads (v1, v2, v3, v4, v5)
        for i in range(1, 6):
            model_path = temp_model_dir / f"model{i}.pkl"
            create_model_file(MockValidModel(f"v{i}"), model_path)
            result = clf.reload_model(str(model_path), expected_version=f"v{i}")
            assert result["status"] == "success"

        # Verify current state: v5, PREV=v4, PREV2=v3, PREV3=v2
        assert clf._MODEL.name == "v5"
        assert clf._MODEL_PREV.name == "v4"
        assert clf._MODEL_PREV2.name == "v3"
        assert clf._MODEL_PREV3.name == "v2"

        # Now fail - shifting will make PREV=v5, then rollback to PREV (v5)
        invalid_path = temp_model_dir / "invalid.pkl"
        create_invalid_pickle(invalid_path)

        result = clf.reload_model(str(invalid_path))

        # After shifting: PREV=v5 (old current), so rollback restores to v5
        assert result["status"] == "rollback"
        assert clf._MODEL.name == "v5"  # Restored to what was current before attempt
